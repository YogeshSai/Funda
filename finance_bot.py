"""
finance_bot.py
--------------
Core logic for FundFinder: an AI-powered mutual fund analytics engine.
(See original module docstring for full background -- unchanged here.)

FIX (duplicate "same fund, different option" cards): _OPTION_PHRASES
below now also strips the generic word "option" itself. Scheme names
like "... Direct Plan - Growth Option" vs "... Direct Plan - IDCW" only
had "growth"/"idcw" stripped previously, leaving a stray leftover
"option" word behind in the Growth variant's key but not the IDCW
variant's -- e.g. "uti equity savings fund direct plan option" vs
"uti equity savings fund direct plan". Those two different keys meant
group_funds_with_variants()/dedup_funds_keep_direct() treated them as
TWO different underlying funds instead of two options of the SAME one,
which is exactly the "UTI Equity Savings Fund - Direct Plan - Growth
Option" / "... - Direct Plan - IDCW" duplicate the site was showing.
Adding "option" to the phrase-strip list collapses both back onto the
same identity key, so the site now shows one fund card with "Growth"
and "IDCW" listed as its Available Plans & Options instead of as two
separate top-level funds.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field
from urllib.parse import quote

import pandas as pd

from rapidfuzz import fuzz

from nlp_utils import (
    AMCMatcher,
    best_fuzzy_match,
    extract_number_word,
    fuzzy_ratio,
    ranked_fuzzy_matches,
)
from llm_fallback import parse_fund_verdict

DATA_FILENAME = "MF_Risk_Metrics.xlsx"
SHEET_NAME = "Risk Metrics"

DEFAULT_DATA_FILENAME = DATA_FILENAME
DEFAULT_SHEET_NAME = SHEET_NAME


def _data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, DATA_FILENAME)


def _fix_mojibake(s):
    if not isinstance(s, str) or "â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


BASIC_COLS = [
    "Scheme Code", "Scheme Name", "AMC (Fund House)", "Sub Category",
    "Asset Class", "ELSS", "Latest NAV Date", "Latest NAV",
]

HORIZONS = ["1D", "1W", "1M", "6M", "1Y", "3Y", "5Y", "10Y", "SI"]
METRIC_SUFFIXES = [
    "AbsoluteReturn", "CAGR", "Volatility", "MaxDrawdown", "Sharpe", "Sortino",
    "DownsideDev", "VaR95", "Calmar", "RollMean", "RollMin", "RollMax",
]
PEER_PCTILE_COLS = [
    "3Y_CAGR_PeerPctile", "3Y_Sharpe_PeerPctile", "3Y_Sortino_PeerPctile",
    "3Y_Calmar_PeerPctile", "3Y_MaxDrawdown_PeerPctile", "3Y_Volatility_PeerPctile",
    "3Y_VaR95_PeerPctile", "3Y_DownsideDev_PeerPctile",
]
SCORE_COLS = ["Composite_Score", "Peer_Rank"]

TOP_N_METRIC_SPECS = [
    ("1D_AbsoluteReturn", None, "1D Obs. Return"),
    ("6M_AbsoluteReturn", None, "6M Obs. Return"),
    ("1Y_AbsoluteReturn", "1Y_CAGR", "1Y Obs. Return"),
    ("3Y_AbsoluteReturn", "3Y_CAGR", "3Y Obs. Return"),
    ("5Y_AbsoluteReturn", "5Y_CAGR", "5Y Obs. Return"),
]

PERCENT_COLS = {
    "1D_AbsoluteReturn", "6M_AbsoluteReturn", "1Y_AbsoluteReturn",
    "3Y_AbsoluteReturn", "5Y_AbsoluteReturn", "1Y_CAGR", "3Y_CAGR", "5Y_CAGR",
}

FRIENDLY_LABELS = {
    "AbsoluteReturn": "Absolute Return",
    "CAGR": "CAGR (Annualised Return)",
    "Volatility": "Volatility (Std. Dev.)",
    "MaxDrawdown": "Max Drawdown",
    "Sharpe": "Sharpe Ratio",
    "Sortino": "Sortino Ratio",
    "DownsideDev": "Downside Deviation",
    "VaR95": "Value at Risk (95%)",
    "Calmar": "Calmar Ratio",
    "RollMean": "Rolling Return (Mean)",
    "RollMin": "Rolling Return (Min)",
    "RollMax": "Rolling Return (Max)",

}

SUBCAT_MATCH_THRESHOLD = 0.6

_ELSS_VARIANT_RE = re.compile(
    r"elss(\s*-?\s*tax\s*saver(\s*fund)?)?", re.IGNORECASE
)


def _canonicalize_subcat(raw):
    if not isinstance(raw, str):
        return raw
    if "elss" in raw.lower():
        return _ELSS_VARIANT_RE.sub("ELSS", raw)
    return raw


# ----------------------------------------------------------------------
# De-duplicating "same fund, different plan/option" rows
# ----------------------------------------------------------------------
# See module docstring FIX note above re: the added "option" phrase.
_OPTION_PHRASES = [
    "payout & re-investment of income distribution cum capital withdrawal option",
    "payout and re-investment of income distribution cum capital withdrawal option",
    "income distribution cum capital withdrawal option",
    "idcw", "dividend", "growth", "payout", "reinvestment", "bonus",
    # Generic trailing/qualifying word that shows up alongside the option
    # name itself (e.g. "Growth Option", "Dividend Option", "IDCW
    # Option") -- stripped separately from the option-name words above so
    # a scheme with "Growth Option" in its name and its sibling with a
    # bare "IDCW" (no trailing "Option" word) still collapse to the same
    # identity key instead of surviving dedup as two different funds.
    "option",
]
_OPTION_KEYWORDS = _OPTION_PHRASES

_FREQUENCY_PHRASES = [
    "half yearly", "half-yearly", "fortnightly", "quarterly",
    "monthly", "weekly", "daily", "annual", "periodic",
]

_DASH_CHARS_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")


def _fund_dedup_key(name: str) -> str:
    text = str(name).lower()
    text = _DASH_CHARS_RE.sub("-", text)
    for phrase in _OPTION_PHRASES + _FREQUENCY_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[()/]", " ", text)
    text = re.sub(r"[-,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_growth_variant(name: str) -> bool:
    return "growth" in str(name).lower()


def dedup_funds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Scheme Name" not in df.columns:
        return df
    work = df.copy()
    work["_dedup_key"] = work["Scheme Name"].apply(_fund_dedup_key)
    work["_is_growth"] = work["Scheme Name"].apply(_is_growth_variant)
    work = work.sort_values("_is_growth", ascending=False, kind="stable")
    work = work.drop_duplicates(subset="_dedup_key", keep="first")
    return work.drop(columns=["_dedup_key", "_is_growth"])


_PLAN_PHRASES = ["direct plan", "regular plan", "direct", "regular"]


def _is_direct_variant(name: str) -> bool:
    return "direct" in str(name).lower()


def _fund_identity_key(name: str) -> str:
    text = _fund_dedup_key(name)
    for phrase in _PLAN_PHRASES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[-,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_funds_keep_direct(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Scheme Name" not in df.columns:
        return df
    work = df.copy()
    work["_identity_key"] = work["Scheme Name"].apply(_fund_identity_key)
    work["_is_direct"] = work["Scheme Name"].apply(_is_direct_variant)
    work["_is_growth"] = work["Scheme Name"].apply(_is_growth_variant)
    work = work.sort_values(
        ["_is_direct", "_is_growth"], ascending=[False, False], kind="stable"
    )
    work = work.drop_duplicates(subset="_identity_key", keep="first")
    return work.drop(columns=["_identity_key", "_is_direct", "_is_growth"])


_IDCW_HINT_RE = re.compile(
    r"idcw|income distribution|dividend", re.IGNORECASE
)
_FREQUENCY_RE = re.compile(
    r"\b(half[\s-]?yearly|fortnightly|quarterly|monthly|weekly|daily|annual|periodic)\b",
    re.IGNORECASE,
)
_IDCW_FULL_NAME_RE = re.compile(
    r"income\s+distribution\s+cum\s+capital\s+withdrawal", re.IGNORECASE
)


def describe_variant_label(name: str) -> str:
    text = str(name)
    lower = text.lower()
    plan = "Direct Plan" if "direct" in lower else ("Regular Plan" if "regular" in lower else "")

    if _IDCW_HINT_RE.search(lower):
        freq_match = _FREQUENCY_RE.search(lower)
        freq = freq_match.group(1).title() if freq_match else ""
        option = (freq + " IDCW").strip()
    elif "growth" in lower:
        option = "Growth"
    else:
        option = "Other"

    return f"{plan} - {option}" if plan else option


def describe_variant_fields(name: str) -> dict:
    text = str(name)
    lower = text.lower()

    plan = "Direct" if "direct" in lower else ("Regular" if "regular" in lower else "")

    if _IDCW_FULL_NAME_RE.search(lower):
        option = "Income Distribution cum Capital Withdrawal"
    elif "idcw" in lower:
        option = "IDCW"
    elif "dividend" in lower:
        option = "Dividend"
    elif "growth" in lower:
        option = "Growth"
    else:
        option = "Other"

    frequency = ""
    if option in ("IDCW", "Dividend", "Income Distribution cum Capital Withdrawal"):
        freq_match = _FREQUENCY_RE.search(lower)
        if freq_match:
            frequency = freq_match.group(1).title()

    return {"plan": plan, "option": option, "frequency": frequency}


def group_funds_with_variants(df: pd.DataFrame) -> list[dict]:
    if df.empty or "Scheme Name" not in df.columns:
        return [{"primary": row, "variants": []} for _, row in df.iterrows()]

    work = df.copy()
    work["_identity_key"] = work["Scheme Name"].apply(_fund_identity_key)
    work["_is_direct"] = work["Scheme Name"].apply(_is_direct_variant)
    work["_is_growth"] = work["Scheme Name"].apply(_is_growth_variant)

    groups = []
    for _key, group in work.groupby("_identity_key", sort=False):
        group = group.sort_values(
            ["_is_direct", "_is_growth"], ascending=[False, False], kind="stable"
        )
        rows = [r.drop(labels=["_identity_key", "_is_direct", "_is_growth"]) for _, r in group.iterrows()]
        primary, *variants = rows
        groups.append({"primary": primary, "variants": variants})
    return groups


_WRAPPER_PHRASES = ["close ended schemes", "open ended schemes"]


def clean_subcat_label(raw: str) -> str:
    if not raw:
        return raw
    text = str(raw)
    for phrase in _WRAPPER_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or str(raw)


_SCHEME_TYPE_PREFIXES = [
    "income/debt oriented schemes",
    "exchange traded funds etfs",
    "overseas fund of funds",
    "solution oriented scheme",
    "debt schemes", "debt scheme",
    "equity schemes", "equity scheme",
    "hybrid schemes", "hybrid scheme",
    "index funds",
    "other scheme",
]


def strip_scheme_type_prefix(label: str) -> str:
    text = str(label)
    text_l = text.lower()
    for prefix in _SCHEME_TYPE_PREFIXES:
        if text_l.startswith(prefix):
            rest = text[len(prefix):].lstrip(" -")
            return rest or text
    return text


def subcat_browse_label(raw: str) -> str:
    return strip_scheme_type_prefix(clean_subcat_label(raw))


def _normalize_category_text(text: str) -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"\bfunds\b", "fund", t)
    t = re.sub(r"\b(large|mid|small|multi|flexi)cap\b", r"\1 cap", t)
    words = t.split(" ")
    for i, w in enumerate(words):
        if w and w != "fund" and 3 <= len(w) <= 6 and fuzz.ratio(w, "fund") >= 65:
            words[i] = "fund"
    t = " ".join(words)
    t = re.sub(r"\s+", " ", t).strip()
    return t

_CATEGORY_BOILERPLATE_WORDS = {
    "equity", "debt", "hybrid", "scheme", "schemes", "open", "ended", "fund",
}


def _category_core_tokens(label_l: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", label_l)
        if w not in _CATEGORY_BOILERPLATE_WORDS
    }


_LABEL_MERGE_THRESHOLD = 0.87


def _merge_similar_labels(by_label: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    ordered = sorted(by_label.keys(), key=lambda l: -by_label[l][1])
    merged: dict[str, tuple[str, int]] = {}
    claimed: set[str] = set()
    for label in ordered:
        if label in claimed:
            continue
        raw, count = by_label[label]
        for other in ordered:
            if other == label or other in claimed:
                continue
            if fuzzy_ratio(label.lower(), other.lower()) >= _LABEL_MERGE_THRESHOLD:
                other_raw, other_count = by_label[other]
                if other_count > count:
                    raw, count = other_raw, other_count
                claimed.add(other)
        claimed.add(label)
        merged[label] = (raw, count)
    return merged


_ASSET_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Solution Oriented", ["solution oriented"]),
    ("Index ETF", ["index fund", "exchange traded fund", "etf"]),
    ("Hybrid", ["hybrid scheme"]),
    ("Equity", ["equity scheme", "elss"]),
    ("Debt", ["debt scheme", "income/debt oriented", "il&fs", "idf", "income"]),
    ("Other", ["other scheme", "fund of funds"]),
]


def _infer_asset_type(raw_subcat: str) -> str | None:
    text = str(raw_subcat).lower()
    for asset_type, keywords in _ASSET_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return asset_type
    return None


def _build_asset_type_subcat_map(
    df: pd.DataFrame, asset_types: list[str]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    buckets: dict[str, dict[str, tuple[str, int]]] = {a: {} for a in asset_types}
    raws_by_label: dict[str, dict[str, list[str]]] = {a: {} for a in asset_types}

    counts = df.groupby(["Asset Class", "Sub Category"]).size()
    for (asset_class, raw_subcat), count in counts.items():
        asset_type = _infer_asset_type(raw_subcat) or asset_class
        by_label = buckets.setdefault(asset_type, {})
        label_raws = raws_by_label.setdefault(asset_type, {})
        label = subcat_browse_label(raw_subcat)
        label_raws.setdefault(label, []).append(raw_subcat)
        current = by_label.get(label)
        if current is None or count > current[1]:
            by_label[label] = (raw_subcat, count)

    result: dict[str, list[str]] = {}
    raw_to_canonical: dict[str, str] = {}
    for asset_type, by_label in buckets.items():
        merged = _merge_similar_labels(by_label)
        raws = sorted({raw for raw, _ in merged.values()}, key=subcat_browse_label)
        result[asset_type] = raws

        label_to_canonical_raw: dict[str, str] = {}
        for surviving_label, (canon_raw, _count) in merged.items():
            label_to_canonical_raw[surviving_label] = canon_raw
        for label in by_label:
            if label not in label_to_canonical_raw:
                for surviving_label, canon_raw in list(label_to_canonical_raw.items()):
                    if fuzzy_ratio(label.lower(), surviving_label.lower()) >= _LABEL_MERGE_THRESHOLD:
                        label_to_canonical_raw[label] = canon_raw
                        break
                else:
                    label_to_canonical_raw[label] = by_label[label][0]
        for label, raw_list in raws_by_label[asset_type].items():
            canon_raw = label_to_canonical_raw.get(label, by_label[label][0])
            for raw in raw_list:
                raw_to_canonical[raw] = canon_raw

    return result, raw_to_canonical


def _fund_link(name: str) -> str:
    safe_name = html.escape(str(name))
    return f'<a href="?fund={quote(str(name))}" target="_self">{safe_name}</a>'


_SUMMARY_PCT_SUFFIXES = {
    "AbsoluteReturn", "CAGR", "Volatility", "MaxDrawdown",
    "DownsideDev", "VaR95", "RollMean", "RollMin", "RollMax",
}
_SUMMARY_HORIZONS = ["3Y"]
_SUMMARY_SUFFIXES = ["CAGR", "Volatility", "MaxDrawdown", "Sharpe", "Sortino", "Calmar"]


def _fund_metrics_text(row: pd.Series) -> str:
    lines = [f"Fund: {row.get('Scheme Name', 'Unknown')}"]
    if "Sub Category" in row.index:
        lines.append(f"Category: {clean_subcat_label(row['Sub Category'])}")

    for horizon in _SUMMARY_HORIZONS:
        parts = []
        for suffix in _SUMMARY_SUFFIXES:
            col = f"{horizon}_{suffix}"
            if col not in row.index or pd.isna(row[col]):
                continue
            v = row[col]
            if suffix in _SUMMARY_PCT_SUFFIXES:
                parts.append(f"{suffix}={v * 100:.2f}%")
            else:
                parts.append(f"{suffix}={v:.2f}")
        if parts:
            lines.append("Past 3 years: " + ", ".join(parts))

    pctile_present = [c for c in PEER_PCTILE_COLS if c in row.index and not pd.isna(row[c])]
    if pctile_present:
        parts = []
        for c in pctile_present:
            label = c.replace("3Y_", "").replace("_PeerPctile", "")
            parts.append(f"{label}={row[c] * 100:.0f}th pctile")
        lines.append("3Y peer percentile ranks: " + ", ".join(parts))

    for c in SCORE_COLS:
        if c in row.index and not pd.isna(row[c]):
            lines.append(f"{c.replace('_', ' ')}: {row[c]:.2f}")

    return "\n".join(lines)


class FundNotFoundError(Exception):
    pass


@dataclass
class FinanceBot:
    df: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        self.load_data()
        self.pending: dict | None = None

    def load_data(self) -> None:
        path = _data_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. Make sure "
                f"'{DATA_FILENAME}' is in the same folder as "
                f"finance_bot.py (sheet: '{SHEET_NAME}')."
            )
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
        df.columns = [c.strip() for c in df.columns]
        required = {"Scheme Name", "Sub Category", "Composite_Score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].map(_fix_mojibake)

        for col in ("Sub Category", "Asset Class"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

        if "Sub Category" in df.columns:
            df["Sub Category"] = df["Sub Category"].apply(_canonicalize_subcat)

        self.df = df
        self._scheme_names = df["Scheme Name"].astype(str).tolist()
        self._sub_categories = sorted(df["Sub Category"].dropna().unique().tolist())

        if "Asset Class" in df.columns:
            self._asset_types = sorted(df["Asset Class"].dropna().unique().tolist())
            self._asset_type_to_subcats, self._subcat_canonical_map = _build_asset_type_subcat_map(
                df, self._asset_types
            )
        else:
            self._asset_types = ["All Funds"]
            self._asset_type_to_subcats = {"All Funds": self._sub_categories}
            self._subcat_canonical_map = {sc: sc for sc in self._sub_categories}

        if "AMC (Fund House)" in df.columns:
            amc_values = df["AMC (Fund House)"].dropna().astype(str).unique().tolist()
        else:
            amc_values = []
        self._amc_matcher = AMCMatcher(amc_values)

        self._amc_first_words: set[str] = set(self._amc_matcher._brand_words)

    @property
    def sub_categories(self) -> list[str]:
        return self._sub_categories

    @property
    def asset_types(self) -> list[str]:
        return self._asset_types

    @property
    def asset_type_to_subcats(self) -> dict[str, list[str]]:
        return self._asset_type_to_subcats

    @property
    def subcat_canonical_map(self) -> dict[str, str]:
        return self._subcat_canonical_map

    def fund_count(self) -> int:
        return len(self.df)

    def _extract_amc_and_rest(self, query: str) -> tuple[str | None, str]:
        return self._amc_matcher.extract(query)

    def _funds_for_amc(self, subset: pd.DataFrame, amc_text: str) -> pd.DataFrame:
        amc_col = "AMC (Fund House)"
        if amc_col not in subset.columns or subset.empty:
            return subset.iloc[0:0]
        amc_values = subset[amc_col].dropna().astype(str).unique().tolist()
        lower_to_original: dict[str, str] = {}
        for v in amc_values:
            lower_to_original.setdefault(v.lower(), v)
        amc_values_lower = list(lower_to_original.keys())
        best_lower, score = best_fuzzy_match(
            amc_text.lower(), amc_values_lower, scorer=fuzz.token_set_ratio
        )
        if best_lower is None or score < 0.6:
            return subset.iloc[0:0]
        best = lower_to_original[best_lower]
        return subset[subset[amc_col] == best]

    _CLOSE_ENDED_MARKER = "close ended schemes"

    def _default_subcat_pool(self, query_norm: str, pool: list[str]) -> list[str]:
        if "close" in query_norm and "end" in query_norm:
            return pool
        open_only = [sc for sc in pool if self._CLOSE_ENDED_MARKER not in sc.lower()]
        return open_only or pool

    def best_sub_category_match(
        self, query: str, candidates: list[str] | None = None
    ) -> tuple[str | None, float]:
        q = _normalize_category_text(query)
        if not q:
            return None, 0.0

        pool = candidates if candidates is not None else self._sub_categories
        if not pool:
            return None, 0.0
        pool = self._default_subcat_pool(q, pool)

        best_sc, best_score = None, 0.0
        exact_matches: list[str] = []
        for sc in pool:
            label_l = _normalize_category_text(clean_subcat_label(sc))
            if label_l == q:
                exact_matches.append(sc)
                continue

            score = fuzzy_ratio(q, label_l)
            q_tokens = set(re.findall(r"[a-z0-9]+", q))
            q_core = q_tokens - _CATEGORY_BOILERPLATE_WORDS
            core_tokens = _category_core_tokens(label_l)
            if core_tokens and q_core:
                overlap = q_core & core_tokens
                if overlap:
                    jaccard = len(overlap) / len(q_core | core_tokens)
                    coverage = len(overlap) / len(core_tokens)
                    if q_core <= core_tokens or coverage >= 0.5:
                        score = max(score, 0.6 + 0.35 * jaccard)
            if q in label_l or label_l.replace(" fund", "").strip() in q:
                score = max(score, 0.7)

            if score > best_score:
                best_sc, best_score = sc, score

        if exact_matches:
            best_exact = max(
                exact_matches,
                key=lambda sc: int((self.df["Sub Category"] == sc).sum()),
            )
            return best_exact, 1.0

        return best_sc, best_score

    def match_sub_categories(self, query: str, limit: int = 3) -> list[str]:
        q = _normalize_category_text(query)
        if not q:
            return []
        pool = self._default_subcat_pool(q, self._sub_categories)
        scored = []
        for sc in pool:
            label_l = _normalize_category_text(clean_subcat_label(sc))
            score = fuzzy_ratio(q, label_l)
            if q in label_l or label_l.replace(" fund", "").strip() in q:
                score = max(score, 0.85)
            scored.append((sc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [sc for sc, sc_score in scored[:limit] if sc_score > 0]

    def get_fund_group(self, scheme_name: str) -> tuple[pd.Series, list[pd.Series]]:
        match = self.df[self.df["Scheme Name"] == scheme_name]
        if match.empty:
            raise FundNotFoundError(f"No fund matching '{scheme_name}' found in the dataset.")
        row = match.iloc[0]
        key = _fund_identity_key(scheme_name)
        same_family = self.df[self.df["Scheme Name"].apply(_fund_identity_key) == key]
        siblings = [r for _, r in same_family.iterrows() if r["Scheme Name"] != scheme_name]
        return row, siblings

    def match_fund(self, query: str) -> pd.Series:
        q = query.strip().lower()
        if not q:
            raise FundNotFoundError("No fund name given.")

        exact = self.df[self.df["Scheme Name"].str.lower() == q]
        if len(exact):
            return exact.iloc[0]

        contains = self.df[self.df["Scheme Name"].str.lower().str.contains(re.escape(q), na=False)]
        if len(contains):
            contains = contains.assign(_len=contains["Scheme Name"].str.len()).sort_values("_len")
            return contains.iloc[0]

        best, score = best_fuzzy_match(query, self._scheme_names)
        if best is not None and score >= 0.45:
            return self.df[self.df["Scheme Name"] == best].iloc[0]

        raise FundNotFoundError(f"No fund matching '{query}' found in the dataset.")

    def match_funds_multi(self, query: str, n: int = 5, score_cutoff: float = 0.4) -> pd.DataFrame:
        q = query.strip().lower()
        contains = self.df[self.df["Scheme Name"].str.lower().str.contains(re.escape(q), na=False)]
        if len(contains):
            return contains.head(n)
        matches = ranked_fuzzy_matches(query, self._scheme_names, limit=n, score_cutoff=score_cutoff)
        names = [name for name, _score in matches]
        return self.df[self.df["Scheme Name"].isin(names)]

    def match_funds_ranked(self, query: str, n: int = 6) -> pd.DataFrame:
        q = (query or "").strip()
        if not q:
            return self.df.iloc[0:0]

        matches = ranked_fuzzy_matches(q, self._scheme_names, limit=n, score_cutoff=0.55)
        if not matches:
            return self.df.iloc[0:0]
        ordered_names = [name for name, _score in matches]
        rows = self.df[self.df["Scheme Name"].isin(ordered_names)]
        rows = rows.drop_duplicates(subset="Scheme Name")
        rank = {name: i for i, name in enumerate(ordered_names)}
        rows = rows.assign(_rank=rows["Scheme Name"].map(rank)).sort_values("_rank")
        return rows.drop(columns="_rank").head(n)

    def search_suggestions(self, query: str, limit: int = 8) -> list[dict]:
        q = (query or "").strip()
        if not q:
            return []

        scored: list[tuple[float, dict]] = []

        lower_to_name: dict[str, str] = {}
        for name in self._scheme_names:
            lower_to_name.setdefault(name.lower(), name)
        fund_matches = ranked_fuzzy_matches(
            q.lower(), list(lower_to_name.keys()), limit=limit,
            scorer=fuzz.token_set_ratio, score_cutoff=0.4,
        )
        for name_lower, score in fund_matches:
            name = lower_to_name[name_lower]
            scored.append((score, {"type": "fund", "label": name, "value": name}))

        q_norm = _normalize_category_text(q)
        for sc in self._sub_categories:
            label = subcat_browse_label(sc)
            label_norm = _normalize_category_text(label)
            score = fuzzy_ratio(q_norm, label_norm)
            if q_norm in label_norm:
                score = max(score, 0.75)
            if score >= 0.4:
                scored.append((score, {"type": "category", "label": label, "value": sc}))

        best_by_key: dict[tuple[str, str], tuple[float, dict]] = {}
        for score, entry in scored:
            key = (entry["type"], entry["label"])
            current = best_by_key.get(key)
            if current is None or score > current[0]:
                best_by_key[key] = (score, entry)

        ranked = sorted(best_by_key.values(), key=lambda t: t[0], reverse=True)
        return [entry for _score, entry in ranked[:limit]]

    def top_funds(
        self, sub_category: str, n: int = 5, sort_by: str = "Peer_Rank",
        amc: str | None = None,
    ) -> pd.DataFrame:
        subset = self.df[self.df["Sub Category"] == sub_category].copy()
        if subset.empty:
            return subset

        if amc:
            subset = self._funds_for_amc(subset, amc)
            if subset.empty:
                return subset

        subset = dedup_funds_keep_direct(subset)

        amc_col = "AMC (Fund House)" if ("AMC (Fund House)" in subset.columns and not amc) else None

        if "Peer_Rank" in subset.columns:
            subset["Peer_Rank"] = pd.to_numeric(subset["Peer_Rank"], errors="coerce")
            subset = subset.dropna(subset=["Peer_Rank"])
            subset = subset.sort_values(["Peer_Rank", "Composite_Score"], ascending=[True, False])
            if amc_col:
                subset = subset.drop_duplicates(subset=amc_col, keep="first")
            return subset.head(n)
        if sort_by not in subset.columns:
            sort_by = "Composite_Score"
        subset = subset.sort_values(sort_by, ascending=False)
        if amc_col:
            subset = subset.drop_duplicates(subset=amc_col, keep="first")
        return subset.head(n)

    def _resolve_top_n_metric_cols(self, subset: pd.DataFrame) -> list[tuple[str, str]]:
        resolved = []
        for primary, fallback, label in TOP_N_METRIC_SPECS:
            if primary in subset.columns:
                resolved.append((primary, label))
            elif fallback and fallback in subset.columns:
                resolved.append((fallback, label))
        return resolved

    def format_top_funds(self, sub_category: str, n: int = 5, amc: str | None = None) -> str:
        subset = self.top_funds(sub_category, n=n, amc=amc)
        cat_label = clean_subcat_label(sub_category)
        if subset.empty:
            if amc:
                return f"I couldn't find any **{amc}** funds in **{cat_label}**."
            return f"I couldn't find any funds in **{cat_label}**."

        metric_cols = self._resolve_top_n_metric_cols(subset)
        heading = f"### Top performing funds in **{cat_label}**"
        if amc:
            heading += f" — **{amc}**"
        heading += f" ({len(subset)} funds)\n"

        cards = ['<div class="ff-fundlist">']
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            name_html = _fund_link(row["Scheme Name"])
            chips = []
            for col, label in metric_cols:
                period = label.split(" ", 1)[0]
                v = row[col]
                sign_cls = ""
                if pd.isna(v):
                    disp = "—"
                elif isinstance(v, float):
                    sign_cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                    disp = f"{v * 100:.2f}%" if col in PERCENT_COLS else f"{v:.2f}"
                else:
                    disp = html.escape(str(v))
                chips.append(
                    '<div class="ff-metric">'
                    f'<span class="ff-metric-label">{html.escape(period)}</span>'
                    f'<span class="ff-metric-value {sign_cls}">{disp}</span>'
                    "</div>"
                )
            cards.append(
                '<div class="ff-fundcard">'
                '<div class="ff-fundcard-head">'
                f'<span class="ff-rank">{i}</span>'
                f'<span class="ff-fundname">{name_html}</span>'
                "</div>"
                f'<div class="ff-metric-grid">{"".join(chips)}</div>'
                "</div>"
            )
        cards.append("</div>")

        out = [heading, "".join(cards)]
        out.append(
            "\n_Ranked by Composite Score (blends 3Y return, risk-adjusted "
            "return, drawdown & volatility percentile vs. category peers). "
            "Tap a fund name for the full metric sheet._"
        )
        return "\n".join(out)

    def format_fund_profile(self, row: pd.Series, fund_summarizer=None, variants: list[pd.Series] | None = None) -> str:
        def fmt(v, is_percent=False):
            if pd.isna(v):
                return "—"
            if isinstance(v, float):
                if is_percent:
                    return f"{v * 100:.2f}%"
                return f"{v:.2f}"
            return str(v)

        out = [f"## {row['Scheme Name']}\n"]

        for c in BASIC_COLS:
            if c == "Scheme Name" or c not in row.index:
                continue
            v = row[c]
            if pd.isna(v):
                continue
            if c == "ELSS":
                if str(v).strip().lower() in ("true", "yes", "y", "1", "1.0"):
                    out.append("- **ELSS:** Yes")
                continue
            label = "Sub Category" if c == "Sub Category" else c
            value = clean_subcat_label(str(v)) if c == "Sub Category" else fmt(v)
            out.append(f"- **{label}:** {value}")
        out.append("")

        out.append("**Performance & Risk by Horizon**")
        out.append('<div class="ff-hint">Tap a period to see the full breakdown.</div>')
        out.append("")
        out.append('<div class="ff-horizons">')
        for horizon in HORIZONS:
            present = [
                f"{horizon}_{s}" for s in METRIC_SUFFIXES
                if f"{horizon}_{s}" in row.index and not pd.isna(row[f"{horizon}_{s}"])
            ]
            if not present:
                continue

            cagr_col, abs_col = f"{horizon}_CAGR", f"{horizon}_AbsoluteReturn"
            headline_is_cagr = cagr_col in row.index and not pd.isna(row.get(cagr_col))
            headline_val = row.get(cagr_col) if headline_is_cagr else row.get(abs_col)
            has_headline = headline_val is not None and not pd.isna(headline_val)
            headline = (
                f"{fmt(headline_val, is_percent=True)}" + (" p.a." if headline_is_cagr else "")
                if has_headline else "—"
            )
            sign_cls = ""
            if has_headline:
                sign_cls = "pos" if headline_val > 0 else ("neg" if headline_val < 0 else "")

            out.append('<details class="ff-horizon">')
            out.append(
                "<summary>"
                f'<span class="ff-h-period">{horizon}</span>'
                '<span class="ff-h-return-wrap">'
                '<span class="ff-h-return-label">Return</span>'
                f'<span class="ff-h-return {sign_cls}">{headline}</span>'
                "</span>"
                "</summary>"
            )
            out.append("")
            for c in present:
                suffix = c.split("_", 1)[1]
                label = FRIENDLY_LABELS.get(suffix, suffix)
                is_pct = suffix in (
                    "AbsoluteReturn", "CAGR", "Volatility", "MaxDrawdown",
                    "DownsideDev", "VaR95", "RollMean", "RollMin", "RollMax",
                )
                out.append(f"- {label}: {fmt(row[c], is_percent=is_pct)}")
            out.append("")
            out.append("</details>")
        out.append("</div>")
        out.append("")

        composite = row.get("Composite_Score")
        peer_rank = row.get("Peer_Rank")
        if not pd.isna(composite) or not pd.isna(peer_rank):
            bits = []
            if not pd.isna(composite):
                bits.append(f"**Composite Score:** {fmt(composite)}")
            if not pd.isna(peer_rank):
                bits.append(f"**Peer Rank:** #{int(peer_rank)}")
            out.append(" · ".join(bits))

        all_rows = [row] + list(variants or [])
        if len(all_rows) > 1:
            out.append("")
            out.append("**Available Plans & Options**")
            out.append('<div class="ff-hint">Same underlying fund — pick the Plan/Option you hold or want to invest in.</div>')
            out.append("")

            by_plan: dict[str, list[tuple[pd.Series, dict]]] = {}
            for r in all_rows:
                fields = describe_variant_fields(r["Scheme Name"])
                plan_key = fields["plan"] or "Other"
                by_plan.setdefault(plan_key, []).append((r, fields))

            plan_order = [p for p in ("Direct", "Regular") if p in by_plan]
            plan_order += [p for p in by_plan if p not in plan_order]

            out.append('<div class="ff-plans">')
            for plan_key in plan_order:
                out.append('<div class="ff-plan-group">')
                out.append(f'<div class="ff-plan-title">{html.escape(plan_key)} Plan</div>')
                for r, fields in by_plan[plan_key]:
                    option = fields["option"]
                    freq = fields["frequency"]
                    option_label = f"{freq} {option}" if freq else option
                    is_current = r["Scheme Name"] == row["Scheme Name"]
                    nav_val = r.get("Latest NAV")
                    nav_bit = f" — NAV {fmt(nav_val)}" if not pd.isna(nav_val) else ""
                    current_tag = ' <span class="ff-current-tag">(viewing)</span>' if is_current else ""
                    out.append(
                        '<div class="ff-plan-option">'
                        f'<span class="ff-plan-option-label">{html.escape(option_label)}</span>'
                        f'<span class="ff-plan-option-nav">{nav_bit}</span>'
                        f'{current_tag}'
                        "</div>"
                    )
                out.append("</div>")
            out.append("</div>")

        out.append("")
        out.append("**AI Verdict**")
        summary_text = fund_summarizer(_fund_metrics_text(row)) if fund_summarizer else None
        if summary_text:
            badge, body = parse_fund_verdict(summary_text)
            if badge:
                out.append(badge)
                out.append("")
            out.append(body)
        else:
            out.append(
                "_AI verdict is temporarily unavailable. Set GROQ_API_KEY to "
                "enable an automated invest/avoid read for every fund._"
            )

        return "\n".join(out)

    def _render_prompt(self, heading: str) -> str:
        return f"### {heading}\n\n_Reply with its number or name._"

    def pending_options_payload(self) -> list[dict] | None:
        if not self.pending:
            return None
        stage = self.pending["stage"]
        options = self.pending.get("options", [])
        clean = stage == "await_sub_category"
        return [
            {
                "index": i,
                "label": subcat_browse_label(opt) if clean else opt,
                "value": opt,
            }
            for i, opt in enumerate(options, start=1)
        ]

    def start_asset_type_flow(self) -> str:
        self.pending = {"stage": "await_asset_type", "options": self._asset_types}
        return self._render_prompt("Which Asset Type are you interested in?")

    def _resolve_choice(self, query: str, options: list[str]) -> str | None:
        q = query.strip()
        if q.isdigit():
            idx = int(q) - 1
            if 0 <= idx < len(options):
                return options[idx]
            return None
        best, score = self.best_sub_category_match(q, candidates=options)
        return best if score >= 0.3 else None

    def _handle_pending(self, query: str, fund_summarizer=None) -> str | None:
        if not self.pending:
            return None

        stage = self.pending["stage"]

        if stage == "await_asset_type":
            choice = self._resolve_choice(query, self.pending["options"])
            if choice is None:
                return (
                    "I didn't quite catch that. " +
                    self._render_prompt("Please pick an Asset Type")
                )
            subcats = self._asset_type_to_subcats.get(choice, [])
            self.pending = {
                "stage": "await_sub_category",
                "asset_type": choice,
                "options": subcats,
            }
            if not subcats:
                self.pending = None
                return f"No Sub Categories found under **{choice}**."
            return self._render_prompt(f"Great — within {choice}, which Sub Category?")

        if stage == "await_sub_category":
            options = self.pending["options"]
            choice = self._resolve_choice(query, options)
            if choice is None:
                choice, score = self.best_sub_category_match(query)
                if choice is None or score < SUBCAT_MATCH_THRESHOLD:
                    return (
                        "I didn't quite catch that. " +
                        self._render_prompt("Please pick a Sub Category")
                    )
            self.pending = None
            return self.format_top_funds(choice, n=5)

        if stage == "await_fund_choice":
            options = self.pending["options"]
            choice = self._resolve_choice(query, options)
            if choice is None:
                return (
                    "I didn't quite catch that. " +
                    self._render_prompt("Please pick a fund")
                )
            self.pending = None
            match = self.df[self.df["Scheme Name"] == choice]
            if match.empty:
                return f"I couldn't find '{choice}' in the dataset."
            row, variants = self.get_fund_group(choice)
            return self.format_fund_profile(row, fund_summarizer=fund_summarizer, variants=variants)

        self.pending = None
        return None

    TOP_PATTERNS = [
        r"\btop\s*(\d+)?\s*(?:performing|rated|funds?)\b.*?\bin\b\s*(.+)",
        r"\bbest\s*(\d+)?\s*funds?\b.*?\bin\b\s*(.+)",
        r"\btop\s*(\d+)?\s*(.+?)\s*funds?\b",
    ]

    def detect_intent(self, query: str) -> tuple[str, dict]:
        q = query.strip()

        amc, q_wo_amc = self._extract_amc_and_rest(q)
        ql = q_wo_amc.lower()

        for pat in self.TOP_PATTERNS:
            m = re.search(pat, ql)
            if m:
                groups = m.groups()
                n = 5
                cat_text = None
                for g in groups:
                    if g and g.isdigit():
                        n = int(g)
                    elif g:
                        cat_text = g
                if cat_text is None:
                    word_n = extract_number_word(ql)
                    if word_n:
                        n = word_n
                if cat_text:
                    cat_text = re.sub(r"\bfunds?\b", "", cat_text).strip(" ?.!")
                    return "top_funds", {"category_text": cat_text, "n": n, "amc": amc}

        if any(kw in ql for kw in ["tell me about", "info on", "information about",
                                    "details of", "details on", "about the fund",
                                    "how is", "how's"]):
            for trigger in ["tell me about", "info on", "information about",
                             "details of", "details on", "about the fund", "how is", "how's"]:
                if trigger in ql:
                    idx_full = q.lower().index(trigger) + len(trigger)
                    return "fund_info", {"fund_text": q[idx_full:].strip(" ?.!")}

        if any(kw in ql for kw in ["show categories", "browse funds", "show funds",
                                    "which categories", "list categories",
                                    "top funds", "show me funds"]) and not self.match_sub_categories(q_wo_amc):
            return "browse", {}

        generic_question = bool(re.match(
            r"^(what|why|how|explain|define|meaning of|difference between)\b", ql
        ))

        if not generic_question:
            best_sc, score = self.best_sub_category_match(q_wo_amc)
            if best_sc and score >= SUBCAT_MATCH_THRESHOLD:
                return "top_funds", {"category_text": q_wo_amc, "n": 5, "amc": amc}
        else:
            best_sc, score = None, 0.0

        if not generic_question:
            if score < 0.45:
                candidate = self.match_funds_multi(q, n=1, score_cutoff=0.55)
                if not candidate.empty:
                    return "fund_info", {"fund_text": q}

        return "unknown", {"raw": q}

    def respond(self, query: str, llm_fallback=None, fund_summarizer=None) -> str:
        pending_response = self._handle_pending(query, fund_summarizer=fund_summarizer)
        if pending_response is not None:
            return pending_response

        intent, params = self.detect_intent(query)

        if intent == "browse":
            return self.start_asset_type_flow()

        if intent == "top_funds":
            best_sc, score = self.best_sub_category_match(params["category_text"])
            if not best_sc or score < SUBCAT_MATCH_THRESHOLD:
                return self.start_asset_type_flow()
            amc = params.get("amc")
            return self.format_top_funds(best_sc, n=params.get("n", 5), amc=amc)

        if intent == "fund_info":
            fund_text = params["fund_text"]
            exact = self.df[
                self.df["Scheme Name"].str.lower() == fund_text.strip().lower()
            ]
            if not exact.empty:
                row, variants = self.get_fund_group(exact.iloc[0]["Scheme Name"])
                return self.format_fund_profile(row, fund_summarizer=fund_summarizer, variants=variants)
            candidates = self.match_funds_ranked(fund_text, n=6)
            if candidates.empty:
                return f"I couldn't find a fund matching '{fund_text}' in the dataset."
            if len(candidates) == 1:
                row, variants = self.get_fund_group(candidates.iloc[0]["Scheme Name"])
                return self.format_fund_profile(row, fund_summarizer=fund_summarizer, variants=variants)
            self.pending = {
                "stage": "await_fund_choice",
                "options": candidates["Scheme Name"].tolist(),
            }
            return self._render_prompt(f"A few funds match \"{fund_text}\" — which one?")

        if llm_fallback is not None:
            return llm_fallback(query)

        return (
            "I can help with two things right now:\n\n"
            "1. **Top performing funds** — search a category, or type "
            "something like *\"top 10 mid cap funds\"*\n"
            "2. **Fund details** — e.g. *\"HDFC Flexi Cap Fund\"*\n\n"
            + self.start_asset_type_flow()
        )
