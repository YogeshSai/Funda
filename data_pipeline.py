"""
data_pipeline.py
-----------------
Turns the raw "Risk Metrics" sheet of MF_Risk_Metrics.xlsx into a
compact list of fund records ready to embed in fundfinder.html.

FIX: _OPTION_PHRASES now also strips the bare word "option" (see the
matching note in finance_bot.py) so a "... Growth Option" row and its
"... IDCW" sibling collapse to the same dedup key here too, instead of
surviving as two separate top-level fund records.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

SHEET_NAME = "Risk Metrics"

_PEER_PCTILE_COLS = [
    "3Y_CAGR_PeerPctile", "3Y_Sharpe_PeerPctile", "3Y_Sortino_PeerPctile",
    "3Y_Calmar_PeerPctile", "3Y_MaxDrawdown_PeerPctile", "3Y_Volatility_PeerPctile",
    "3Y_VaR95_PeerPctile", "3Y_DownsideDev_PeerPctile",
]

_RETURN_HORIZONS = ["1D", "6M", "1Y", "3Y", "5Y"]

_ELSS_VARIANT_RE = re.compile(r"elss(\s*-?\s*tax\s*saver(\s*fund)?)?", re.IGNORECASE)

_WRAPPER_PHRASES = ["close ended schemes", "open ended schemes"]

_SCHEME_TYPE_PREFIXES = [
    "income/debt oriented schemes", "exchange traded funds etfs",
    "overseas fund of funds", "solution oriented scheme",
    "debt schemes", "debt scheme", "equity schemes", "equity scheme",
    "hybrid schemes", "hybrid scheme", "index funds", "other scheme",
]

_ASSET_TYPE_KEYWORDS = [
    ("Solution Oriented", ["solution oriented"]),
    ("Index ETF", ["index fund", "exchange traded fund", "etf"]),
    ("Hybrid", ["hybrid scheme"]),
    ("Equity", ["equity scheme", "elss"]),
    ("Debt", ["debt scheme", "income/debt oriented", "il&fs", "idf", "income"]),
    ("Other", ["other scheme", "fund of funds"]),
]

_OPTION_PHRASES = [
    "payout & re-investment of income distribution cum capital withdrawal option",
    "payout and re-investment of income distribution cum capital withdrawal option",
    "income distribution cum capital withdrawal option",
    "idcw", "dividend", "growth", "payout", "reinvestment", "bonus",
    # Generic qualifying word left over after stripping the option name
    # itself (e.g. "Growth Option") -- without this, "... Growth Option"
    # and its "... IDCW" (no trailing "Option" word) sibling produced
    # different dedup keys and survived as two separate fund records.
    "option",
]


def _fix_mojibake(s):
    if not isinstance(s, str) or "\u00e2" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def _canonicalize_subcat(raw):
    if not isinstance(raw, str):
        return raw
    if "elss" in raw.lower():
        return _ELSS_VARIANT_RE.sub("ELSS", raw)
    return raw


def _clean_subcat_label(raw: str) -> str:
    text = str(raw)
    for phrase in _WRAPPER_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = text.replace("(", "").replace(")", "")
    return re.sub(r"\s+", " ", text).strip(" -") or str(raw)


def _strip_scheme_type_prefix(label: str) -> str:
    text_l = label.lower()
    for prefix in _SCHEME_TYPE_PREFIXES:
        if text_l.startswith(prefix):
            rest = label[len(prefix):].lstrip(" -")
            return rest or label
    return label


def _subcat_browse_label(raw: str) -> str:
    return _strip_scheme_type_prefix(_clean_subcat_label(raw))


def _infer_asset_type(raw_subcat: str, fallback: str) -> str:
    text = str(raw_subcat).lower()
    for asset_type, keywords in _ASSET_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return asset_type
    return fallback or "Other"


def _fund_dedup_key(name: str) -> str:
    text = str(name).lower()
    for phrase in _OPTION_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[()]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _is_growth_variant(name: str) -> bool:
    return "growth" in str(name).lower()


def _safe_round(v, nd=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), nd)


def build_fund_records(xlsx_bytes: bytes) -> list[dict]:
    """Reads the Risk Metrics sheet from raw .xlsx bytes and returns a
    list of clean, rankable fund record dicts (JSON-serialisable)."""
    df = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=SHEET_NAME)
    df.columns = [c.strip() for c in df.columns]

    required = {"Scheme Name", "Sub Category", "Composite_Score", "Peer_Rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sheet is missing required columns: {missing}")

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(_fix_mojibake)
    for col in ("Sub Category", "Asset Class"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    df["Sub Category"] = df["Sub Category"].apply(_canonicalize_subcat)

    df = df.dropna(subset=["Composite_Score", "Peer_Rank", "Scheme Name", "Sub Category"])

    df["_dedup_key"] = df["Scheme Name"].apply(_fund_dedup_key)
    df["_is_growth"] = df["Scheme Name"].apply(_is_growth_variant)
    df["Peer_Rank"] = pd.to_numeric(df["Peer_Rank"], errors="coerce")
    df = df.sort_values(["_is_growth", "Peer_Rank"], ascending=[False, True])
    df = df.drop_duplicates(subset="_dedup_key", keep="first")

    records = []
    for _, row in df.iterrows():
        raw_subcat = row["Sub Category"]
        asset_type = _infer_asset_type(raw_subcat, row.get("Asset Class"))
        label = _subcat_browse_label(raw_subcat)

        returns = {h: _safe_round(row.get(f"{h}_AbsoluteReturn")) for h in _RETURN_HORIZONS}
        if all(v is None for v in returns.values()):
            continue

        pctiles = [row.get(c) for c in _PEER_PCTILE_COLS if pd.notna(row.get(c))]
        peer_pctile = round(float(np.mean(pctiles)) * 100) if pctiles else None

        records.append({
            "name": str(row["Scheme Name"]).strip(),
            "amc": str(row.get("AMC (Fund House)") or "").strip(),
            "assetType": asset_type,
            "subCategoryRaw": raw_subcat,
            "subCategoryLabel": label,
            "nav": _safe_round(row.get("Latest NAV"), 2),
            "returns": returns,
            "risk": {
                "vol": _safe_round(row.get("3Y_Volatility")),
                "mdd": _safe_round(row.get("3Y_MaxDrawdown")),
                "sharpe": _safe_round(row.get("3Y_Sharpe"), 2),
                "sortino": _safe_round(row.get("3Y_Sortino"), 2),
                "calmar": _safe_round(row.get("3Y_Calmar"), 2),
            },
            "peerPctile": peer_pctile,
            "compositeScore": _safe_round(row.get("Composite_Score"), 1),
            "peerRank": int(row["Peer_Rank"]) if pd.notna(row["Peer_Rank"]) else None,
        })

    return records
