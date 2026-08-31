"""
finance_bot.py
--------------
Core logic for FundFinder: an AI-powered mutual fund analytics engine.

Responsibilities:
  1. Load & validate the fund dataset from the fixed local file
     "MF_Risk_Metrics.xlsx" (sheet "Risk Metrics"), located in the same
     folder as this script. This is the ONLY data source the bot will ever
     read from -- there is no upload path, and the loader does not accept
     an alternate file or sheet name.
  2. Answer "top performing funds in <Sub Category>" queries, matching the
     user's category text against the dataset's real Sub Category values by
     highest similarity score (no need to type it exactly), and -- if an
     AMC / fund-house name is present in the query (e.g. "HDFC Small cap
     funds") -- filtering results down to just that AMC.
  3. Answer "tell me about <Scheme Name>" queries with the full metric sheet.
  4. Lightweight intent + entity extraction: regex for coarse intent
     shape ("top N ... in ...") combined with local, offline NLP
     (nlp_utils.py: spaCy tokenization + a dataset-driven AMC matcher,
     and rapidfuzz for fuzzy string scoring) for pulling the AMC name and
     the true category/fund text out of free-form phrasing. No LLM call
     is required for these two core features, and nothing in this path
     makes a network request.
  5. A guided "Asset Type -> Sub Category" category directory: Sub
     Category options are always shown to the user with clean, friendly
     labels (see `clean_subcat_label` / `subcat_browse_label`) rather
     than raw dataset strings.
  6. A combined fund + category search-suggestion ranker
     (`search_suggestions`) that powers the search box's typeahead in
     the website UI (app.py) -- one box that can resolve either straight
     to a fund's profile or to a category's top-10 list.
  7. Optional LLM fallback (Groq) for free-form finance questions that
     aren't a direct top-N or fund-lookup request, and a separate,
     always-attempted per-fund "AI Verdict" summary. See llm_fallback.py.

Sub Category canonicalization
------------------------------
The raw dataset sometimes tags the *same* SEBI category under two
different spellings -- most notably ELSS appearing both as plain
"...(ELSS)" and as "...(ELSS Tax Saver)" / "...(ELSS Tax Saver Fund)".
Since every downstream feature (the sub-category matcher, the Asset
Type -> Sub Category browse map, and top_funds()) keys off the raw
"Sub Category" column verbatim, two spellings of the same category
would otherwise survive as two separate entries everywhere -- the
category directory, the guided-flow buttons, and top-N results. See
`_canonicalize_subcat`, which is applied once at load time so every
consumer downstream sees a single merged value instead.

NLP layer
---------
See nlp_utils.py for the local, offline NLP helpers this file uses:
  - AMCMatcher: recognizes an AMC / fund-house name anywhere in a query
    (built from the dataset's own "AMC (Fund House)" column -- never a
    hardcoded list) and strips it out, so "HDFC Small cap funds" splits
    cleanly into amc="HDFC" and rest="Small cap funds" instead of the
    AMC name silently diluting -- or being dropped from -- the category
    match.
  - best_fuzzy_match / ranked_fuzzy_matches: rapidfuzz-based fuzzy
    scoring (replaces the previous difflib.SequenceMatcher calls).
    rapidfuzz's token_sort_ratio / token_set_ratio compare bags of
    words rather than raw character sequences, so word reordering and
    a few extra/missing words -- exactly what stripping (or failing to
    strip) an AMC prefix introduces -- no longer tank the score.
This is a fully local/offline NLP layer: spaCy + rapidfuzz only, no
external API calls, no added latency or cost per query.
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

# ----------------------------------------------------------------------
# Fixed dataset location -- this is the single, static source of data.
# There is intentionally no way to point the bot at a different file,
# a different sheet, or an uploaded workbook.
# ----------------------------------------------------------------------

DATA_FILENAME = "MF_Risk_Metrics.xlsx"
SHEET_NAME = "Risk Metrics"

# Backward-compatible aliases (older app.py versions import these names).
# The dataset itself is still fixed/static either way -- these are just
# read-only names pointing at the same constants above.
DEFAULT_DATA_FILENAME = DATA_FILENAME
DEFAULT_SHEET_NAME = SHEET_NAME


def _data_path() -> str:
    """Resolve MF_Risk_Metrics.xlsx sitting next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, DATA_FILENAME)


# A handful of cells in the source spreadsheet have UTF-8 text (curly
# apostrophes, en-dashes) that got saved through Latin-1/cp1252 at some
# point, so they show up as mojibake -- e.g. "Childrenâ€™s Fund" instead
# of "Children's Fund", or "â€"" instead of an en-dash in a couple of
# Scheme Names. Repaired once here at load time (see load_data() below)
# so every downstream table, link, and AI-summary input is clean, rather
# than patching each display function separately.
def _fix_mojibake(s):
    if not isinstance(s, str) or "â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Doesn't round-trip cleanly -- leave it as-is rather than risk
        # mangling a string that just happens to contain "â".
        return s


# ----------------------------------------------------------------------
# Column groupings used for pretty-printing a fund's full profile
# ----------------------------------------------------------------------
# NOTE: these are kept in sync with the actual columns produced by
# mf_risk_metrics.py. There is no TER / cost data in that workbook, so
# there is deliberately no "Costs" section here -- rendering one would
# just show blank/"--" rows for every fund.

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

# ----------------------------------------------------------------------
# Top-N table columns -- Observed (absolute, non-annualised) Return at
# each horizon, NOT CAGR, and no Peer_Rank column shown (Peer_Rank is
# still what the table is ORDERED by -- see top_funds() -- it's just not
# rendered as a column anymore).
#
# Each entry is (source_col, fallback_col_or_None, display_label). The
# fallback is only used if the primary "Obs return" column isn't present
# in the loaded sheet for that horizon, so the table still degrades
# gracefully instead of silently dropping a horizon.
# ----------------------------------------------------------------------
TOP_N_METRIC_SPECS = [
    ("1D_AbsoluteReturn", None, "1D Obs. Return"),
    ("6M_AbsoluteReturn", None, "6M Obs. Return"),
    ("1Y_AbsoluteReturn", "1Y_CAGR", "1Y Obs. Return"),
    ("3Y_AbsoluteReturn", "3Y_CAGR", "3Y Obs. Return"),
    ("5Y_AbsoluteReturn", "5Y_CAGR", "5Y Obs. Return"),
]

# These columns are stored as fractions (0.04 == 4%) and are rendered with
# a trailing '%' in the Top-N table.
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

# Minimum similarity score (0-1) required to auto-accept a free-text
# Sub Category match without falling back to the category directory.
#
# NOTE: this used to be 0.35, which is far too low for token_sort_ratio
# on short strings -- plain conversational words with no category intent
# at all were scoring above it purely by coincidental shared letters,
# e.g. "who" vs "Growth" ~0.44 and "clear" vs "Income" ~0.36, both of
# which silently returned a top-funds table instead of falling through
# to "unknown"/LLM fallback. Genuine (even typo'd) category queries score
# much higher -- "smal cap" vs "Small Cap Fund" ~0.73, "incom" vs
# "Income" ~0.91 -- so 0.6 comfortably keeps those while rejecting noise.
SUBCAT_MATCH_THRESHOLD = 0.6

# ----------------------------------------------------------------------
# Sub Category canonicalization -- merge known duplicate raw spellings
# ----------------------------------------------------------------------
# The dataset can tag the SAME SEBI category under different raw
# "Sub Category" strings. Left unmerged, this shows up as duplicate
# entries in the category directory and splits a single category's
# funds across two separate _funds() results. This runs once at load
# time (see FinanceBot.load_data), before _sub_categories / the Asset
# Type map are built, so every downstream consumer sees one merged
# value.
#
# Known duplicate: ELSS vs. "ELSS Tax Saver" / "ELSS Tax Saver Fund" --
# same category, two spellings. Add further phrase-merge rules here if
# more duplicates like this turn up in the sheet.
_ELSS_VARIANT_RE = re.compile(
    r"elss(\s*-?\s*tax\s*saver(\s*fund)?)?", re.IGNORECASE
)


def _canonicalize_subcat(raw):
    """Collapse known duplicate raw Sub Category spellings onto one
    canonical value. Preserves whatever wrapper the row already has
    ('Open Ended Schemes(...)' etc.) -- only the inner category phrase
    is normalized, so matching against the rest of the dataset (and the
    wrapper-stripping helpers below) still works unchanged."""
    if not isinstance(raw, str):
        return raw
    if "elss" in raw.lower():
        return _ELSS_VARIANT_RE.sub("ELSS", raw)
    return raw


# ----------------------------------------------------------------------
# De-duplicating "same fund, different plan/option" rows
# ----------------------------------------------------------------------
# The dataset often has one row per (Scheme, Plan, Option) combination --
# e.g. "WhiteOak Capital Large Cap Fund Direct Plan Growth" and
# "WhiteOak Capital Large Cap Fund Direct Plan IDCW" are the *same*
# underlying fund/portfolio, just different payout options, and end up
# with identical (or near-identical) return/risk metrics. We collapse
# these down to a single row -- preferring the Growth variant -- before
# ranking/displaying "top funds".
_OPTION_PHRASES = [
    # Longer, descriptive variants FIRST -- these are the newer full-length
    # payout-option names funds have started using instead of the short
    # "IDCW" / "Dividend" suffix, e.g. "SBI Contra Fund - Direct Plan -
    # Income Distribution cum Capital Withdrawal Option (IDCW)". Stripping
    # only the bare word "idcw" leaves the rest of this phrase behind,
    # which then fails to match the Growth variant's key and lets both
    # rows survive dedup -- checked here explicitly to avoid that.
    "payout & re-investment of income distribution cum capital withdrawal option",
    "payout and re-investment of income distribution cum capital withdrawal option",
    "income distribution cum capital withdrawal option",
    "idcw", "dividend", "growth", "payout", "reinvestment", "bonus",
]
# Backward-compatible alias.
_OPTION_KEYWORDS = _OPTION_PHRASES

# ----------------------------------------------------------------------
# IDCW payout-frequency qualifiers -- "Daily IDCW", "Weekly IDCW",
# "Monthly IDCW", "Quarterly IDCW", etc. are all still the SAME
# underlying fund/portfolio as its Growth variant; only how often any
# income distribution gets paid out differs. Previously only the bare
# word "idcw"/"dividend" was stripped out of the dedup key, which left
# "daily"/"weekly"/"monthly"/... sitting in the key -- so
# "HSBC Banking and PSU Debt Fund - Direct Daily IDCW",
# "... Direct Monthly IDCW" and "... Direct Weekly IDCW" each produced a
# DIFFERENT key from each other AND from "... Direct Growth", and none of
# them collapsed together. That's why the site kept showing 3-4 rows for
# what is really one purchasable fund. Stripped out here, alongside the
# option words above, so every frequency variant collapses onto the same
# identity as its Growth sibling.
_FREQUENCY_PHRASES = [
    "half yearly", "half-yearly", "fortnightly", "quarterly",
    "monthly", "weekly", "daily", "annual", "periodic",
]

# ----------------------------------------------------------------------
# Dash-character normalization for dedup keys
# ----------------------------------------------------------------------
# Scheme Names in the sheet are NOT consistent about which dash
# character separates "Fund - Direct Plan - Growth" -- most rows use a
# plain ASCII hyphen ("-"), but a fair number use a typographic en dash
# ("\u2013"), em dash ("\u2014"), or (rarely) a true minus sign
# ("\u2212") instead, sometimes even mixing them within the same name
# (e.g. "Motilal Oswal Contra Fund - Direct \u2013 Growth"). Left
# unnormalized, _fund_dedup_key/_fund_identity_key below strip the
# *word* between dashes (e.g. "growth") but leave a stray non-ASCII
# dash character sitting in the key -- so two rows naming the exact same
# underlying fund/plan/option ended up with DIFFERENT keys purely
# because of which dash glyph was typed, and silently survived dedup as
# two (or three) separate "funds" instead of collapsing into one with
# its variants nested underneath. Normalizing every dash-like character
# to a plain "-" before any other processing fixes this at the source.
_DASH_CHARS_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")


def _fund_dedup_key(name: str) -> str:
    """Normalized identity for a fund, with the plan-option phrase (Growth /
    IDCW / Income Distribution cum Capital Withdrawal Option / Dividend /
    ...) AND any IDCW payout-frequency qualifier (Daily/Weekly/Monthly/...)
    stripped out, so different options -- and different payout frequencies
    of the same option -- of the same underlying fund all collapse to the
    same key. 'Direct'/'Regular Plan' is deliberately kept, since those ARE
    genuinely different funds/TERs."""
    text = str(name).lower()
    # Normalize dash variants FIRST (see _DASH_CHARS_RE above) so the
    # separator-collapsing step near the end of this function catches
    # every dash glyph the sheet might use, not just the ASCII hyphen.
    text = _DASH_CHARS_RE.sub("-", text)
    for phrase in _OPTION_PHRASES + _FREQUENCY_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    # "(" ")" and "/" are all separators that can surround a stripped
    # option phrase -- e.g. "IDCW (Payout / Reinvestment)" leaves a
    # stray "/" behind once "payout" and "reinvestment" are removed by
    # the loop above. Left in place, that stray "/" -- like the
    # unnormalized dash issue above -- produced a different key than the
    # same fund's Growth row and kept the two from collapsing. All three
    # characters are folded into a space here, alongside the dash
    # separator collapsing on the next line.
    text = re.sub(r"[()/]", " ", text)
    # Removing an option phrase leaves a stray separator (usually a "-")
    # behind wherever the phrase used to sit -- e.g. "Fund - IDCW - Direct
    # Plan" becomes "fund -   - direct plan" after stripping "idcw", while
    # "Fund - Direct Plan - Growth" becomes "fund - direct plan -  " after
    # stripping "growth". Only trimming LEADING/TRAILING dashes (the old
    # behaviour) left an interior "-" in the first case but not the
    # second, so the two keys came out different even though they name
    # the same underlying fund -- and IDCW rows survived dedup instead of
    # collapsing into their Growth counterpart. Stripping every dash/comma
    # separator (not just the ones at the ends) before collapsing
    # whitespace makes the key independent of where in the name the
    # option phrase happened to appear.
    text = re.sub(r"[-,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_growth_variant(name: str) -> bool:
    return "growth" in str(name).lower()


def dedup_funds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that represent the same underlying fund under
    different plan-options (Growth/IDCW/Dividend/...) down to one row
    each, keeping the Growth variant when one is present. Direct and
    Regular plans are deliberately kept as separate rows here -- they
    have different expense ratios and are genuinely different
    investable products. See dedup_funds_keep_direct() below for the
    site's actual display behaviour, which collapses those too."""
    if df.empty or "Scheme Name" not in df.columns:
        return df
    work = df.copy()
    work["_dedup_key"] = work["Scheme Name"].apply(_fund_dedup_key)
    work["_is_growth"] = work["Scheme Name"].apply(_is_growth_variant)
    # Growth rows sort first, so drop_duplicates(keep="first") keeps them.
    work = work.sort_values("_is_growth", ascending=False, kind="stable")
    work = work.drop_duplicates(subset="_dedup_key", keep="first")
    return work.drop(columns=["_dedup_key", "_is_growth"])


# ----------------------------------------------------------------------
# Further collapsing Direct vs Regular plan down to a single row per
# fund -- the site shows only ONE row per underlying fund, preferring
# the Direct plan (lower expense ratio, what most retail investors
# should be looking at). A fund that only has a Regular plan on record
# (no Direct option exists for it) still shows up via that Regular
# row -- it's never dropped outright, only de-prioritized.
# ----------------------------------------------------------------------
_PLAN_PHRASES = ["direct plan", "regular plan", "direct", "regular"]


def _is_direct_variant(name: str) -> bool:
    return "direct" in str(name).lower()


def _fund_identity_key(name: str) -> str:
    """Like _fund_dedup_key, but with the Direct/Regular plan phrase
    ALSO stripped out, so different plans of the same underlying fund
    collapse to one identity. Used by dedup_funds_keep_direct() and
    group_funds_with_variants() -- dedup_funds() above still needs
    Direct/Regular kept apart for callers that want both plans
    distinguished."""
    text = _fund_dedup_key(name)
    for phrase in _PLAN_PHRASES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[-,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_funds_keep_direct(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each underlying fund (across BOTH payout option and
    plan type) down to a single row, preferring: Direct plan over
    Regular, and within that, Growth over IDCW/Dividend. A fund that
    only exists as a Regular-plan row in the sheet is still kept (via
    that row) rather than dropped -- preference, not a filter."""
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


# ----------------------------------------------------------------------
# Grouping a fund with its other purchasable Plan/Option variants, for
# display as "Additional investment options" underneath the single
# primary (Direct + Growth, where available) listing -- rather than
# dedup_funds_keep_direct()'s behaviour of silently dropping every
# variant but the primary.
# ----------------------------------------------------------------------
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
    """Short 'Plan - Option' label for a variant row, derived from its
    Scheme Name (e.g. 'Direct Plan - Monthly IDCW', 'Regular Plan -
    Growth'), for display under an "Additional investment options"
    sub-heading."""
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
    """Structured breakdown of a Scheme Name into its three purchasable
    dimensions, for display as separate "Plan / Option / IDCW frequency"
    fields rather than one flattened label string:

      - plan:      "Direct" | "Regular" | "" (unknown)
      - option:    "Growth" | "IDCW" | "Dividend" |
                   "Income Distribution cum Capital Withdrawal" | "Other"
      - frequency: "Daily" | "Weekly" | "Monthly" | "Quarterly" | ... | ""
                   (only ever set when option is an IDCW-type payout)

    Used by format_fund_profile() (grouped "Available Plans & Options"
    section) and by app.py's build_fund_records() (per-fund "plans" list
    sent to the website) so both surfaces show the same breakdown instead
    of re-deriving it themselves ad hoc.
    """
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
    """Group rows by underlying fund identity (ignoring Plan and
    Option/payout-frequency), choosing one primary row per group --
    Direct plan preferred, then Growth option, matching
    dedup_funds_keep_direct()'s own preference order -- and returning
    every OTHER row in that group as a "variant" instead of silently
    dropping it. Used by the site to show one primary fund card/profile
    (Direct Growth wherever it exists) with any other purchasable
    Plan/Option combinations (Regular plan, IDCW payout frequencies,
    etc.) nested underneath as "Additional investment options".

    Returns a list of {"primary": <pd.Series>, "variants": [<pd.Series>, ...]}
    dicts, one per underlying fund.
    """
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


# ----------------------------------------------------------------------
# Sub Category label cleanup
# ----------------------------------------------------------------------
# Raw dataset values look like "Open Ended Schemes(Debt Scheme - Banking
# and PSU Fund)" or "Close Ended Schemes(ELSS)". Matching still runs
# against the raw values (they're what's actually in the dataset), but
# anything shown to the user -- in headings, tables, or directory/
# suggestion labels -- goes through this cleaner first.
_WRAPPER_PHRASES = ["close ended schemes", "open ended schemes"]


def clean_subcat_label(raw: str) -> str:
    """Human-friendly Sub Category label with the 'Close/Open Ended
    Schemes' wrapper text and surrounding parentheses stripped out.

    'Open Ended Schemes(Debt Scheme - Banking and PSU Fund)' ->
    'Debt Scheme - Banking and PSU Fund'
    'Close Ended Schemes(ELSS)' -> 'ELSS'
    """
    if not raw:
        return raw
    text = str(raw)
    for phrase in _WRAPPER_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or str(raw)


# Scheme-type prefixes stripped only for DIRECTORY display (see
# subcat_browse_label below) -- kept separate from clean_subcat_label
# because the Top-N table header and a fund's full profile still want
# the scheme-type shown (they aren't rendered inside an Asset-Type-
# scoped list, so "Contra Fund" alone would lose useful context there).
# Longer/more specific phrases are listed first so e.g. "equity schemes"
# matches before the shorter "equity scheme" would partially match it.
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
    """Remove a leading 'Debt Scheme - ' / 'Equity Scheme - ' / ... style
    prefix from an already wrapper-cleaned Sub Category label. Falls
    back to the original label if nothing would be left after stripping
    (e.g. a bare 'Equity Scheme' with no specific fund type), so an
    option is never shown blank."""
    text = str(label)
    text_l = text.lower()
    for prefix in _SCHEME_TYPE_PREFIXES:
        if text_l.startswith(prefix):
            rest = text[len(prefix):].lstrip(" -")
            return rest or text
    return text


def subcat_browse_label(raw: str) -> str:
    """Display label for a Sub Category when it's shown as an option
    inside an Asset-Type-scoped directory list: both the 'Open/Close
    Ended Schemes' wrapper AND the redundant scheme-type prefix are
    stripped, since the Asset Type is already implied by which list the
    option is in -- e.g. 'Equity Scheme - Contra Fund' -> 'Contra Fund'
    when it's already under the "Equity" list. Used as the dedup key
    when building that list too, so a scheme-prefixed and a bare variant
    that reduce to the same short label (e.g. "Equity Scheme - ELSS" and
    "ELSS") collapse to one option instead of showing as two identical
    entries."""
    return strip_scheme_type_prefix(clean_subcat_label(raw))


# ----------------------------------------------------------------------
# Free-text category matching helpers
# ----------------------------------------------------------------------
def _normalize_category_text(text: str) -> str:
    """Lowercase + collapse whitespace + singularize the standalone word
    'funds' -> 'fund' so a plural user query ('small cap funds') and a
    singular dataset label ('Small Cap Fund') aren't treated as different
    strings by an exact-substring check purely over the trailing 's'.

    Also typo-corrects a word that's a near-miss of 'fund'/'funds' (e.g.
    'fumds', 'fnud') back to 'fund'. Without this, a single-letter typo
    on that one word was enough to knock the whole query below both the
    exact-substring boost AND the fuzzy-match threshold in
    best_sub_category_match() -- so e.g. 'large cap fumds' failed to
    resolve as a category at all and fell through to the single-fund
    fallback in detect_intent(), which then confidently (and wrongly)
    matched a specific fund whose name happened to contain 'Large' and
    'Cap' and 'Fund', instead of showing the Large Cap Fund category list.

    Also splits a market-cap size word run into 'midcap' -> 'mid cap'
    etc. so a no-space query still tokenizes the same way the dataset's
    "<Size> Cap Fund" labels do -- the token-set comparison in
    best_sub_category_match() otherwise sees one unmatched 'midcap' token
    instead of the two tokens ('mid', 'cap') it needs to line up against
    the category label, and silently scores too low to match at all.
    """
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

# Generic wrapper/connective words stripped out before comparing a query's
# words against a category label's words in best_sub_category_match()'s
# substring boost -- these appear in most/all cap-size (and debt/hybrid)
# category labels, so they carry no distinguishing signal and would
# otherwise let two DIFFERENT categories that both happen to contain the
# query as a literal substring (e.g. "mid cap" is a substring of both
# "Mid Cap Fund" and "Large & Mid Cap Fund") tie at the same flat boost.
_CATEGORY_BOILERPLATE_WORDS = {
    "equity", "debt", "hybrid", "scheme", "schemes", "open", "ended", "fund",
}


def _category_core_tokens(label_l: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", label_l)
        if w not in _CATEGORY_BOILERPLATE_WORDS
    }


# ----------------------------------------------------------------------
# Near-duplicate category-label merging
# ----------------------------------------------------------------------
# Even after clean_subcat_label / subcat_browse_label strip the known
# wrapper/prefix text, the sheet itself sometimes spells the SAME SEBI
# category two different ways with no shared machine-detectable pattern
# -- e.g. "Banking and PSU Fund" vs "Banking and PSU Debt Fund". Unlike
# the ELSS case (_canonicalize_subcat), there's no fixed phrase to merge
# on, so this is handled with a fuzzy pass over the cleaned labels
# instead: any two labels within an Asset Type that are near-identical
# (share nearly all their words) get folded into one directory entry,
# keeping whichever raw value is backed by more funds as the canonical
# one. Threshold is intentionally high (0.87) so it only catches
# wording variants of the same category, not genuinely different but
# similarly-worded categories (e.g. "Large Cap Fund" vs "Large & Mid Cap
# Fund" scores well below this).
_LABEL_MERGE_THRESHOLD = 0.87


def _merge_similar_labels(by_label: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    # Biggest group first, so a small mis-spelled variant folds into the
    # dominant spelling rather than the other way around.
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


# ----------------------------------------------------------------------
# Asset Type -> Sub Category mapping for the category directory.
# ----------------------------------------------------------------------
# Two separate problems showed up here:
#
#   1. DUPLICATES: the raw sheet can contain multiple different raw Sub
#      Category strings that all clean down to the *same* display label
#      -- most commonly an "Open Ended Schemes(X)" row and a "Close Ended
#      Schemes(X)" row both collapsing to just "X" via
#      clean_subcat_label(). Building the directory from raw unique
#      values let both survive as separate list entries, so the same
#      label appeared twice under one Asset Type. (A related case --
#      ELSS vs. "ELSS Tax Saver" -- is handled earlier, upstream of this,
#      by _canonicalize_subcat() in load_data(), since those two raw
#      strings don't even clean down to the same text on their own. A
#      further case -- near-duplicate but not identical labels, e.g.
#      "Banking and PSU Fund" vs "Banking and PSU Debt Fund" -- is
#      handled by the fuzzy _merge_similar_labels() pass below.)
#
#   2. MISPLACEMENT: grouping by the sheet's "Asset Class" column trusts
#      that column to be tagged correctly per row. In practice it isn't
#      -- e.g. many "Debt Scheme - ..." / "Hybrid Scheme - ..." /
#      "Solution Oriented Scheme - ..." rows turned out to be tagged
#      Asset Class = "Equity", which then made ALL of those categories
#      surface under the "Equity" bucket instead of their real one.
#
# The fix for both: classify each Sub Category by the SEBI scheme-type
# phrase already embedded in its own text ("Debt Scheme - ...",
# "Equity Scheme - ...", "Hybrid Scheme - ...", "Index Funds - ...",
# "Solution Oriented Scheme - ...", "Other Scheme - ..."), which is
# self-describing and doesn't depend on a separate column that can be
# mistagged. The "Asset Class" column is used only as a fallback for the
# handful of legacy labels ("ELSS", "Growth", "Income", ...) that carry
# no scheme-type phrase of their own.
_ASSET_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Solution Oriented", ["solution oriented"]),
    ("Index ETF", ["index fund", "exchange traded fund", "etf"]),
    ("Hybrid", ["hybrid scheme"]),
    ("Equity", ["equity scheme", "elss"]),
    ("Debt", ["debt scheme", "income/debt oriented", "il&fs", "idf", "income"]),
    ("Other", ["other scheme", "fund of funds"]),
]


def _infer_asset_type(raw_subcat: str) -> str | None:
    """Classify a raw Sub Category string by the scheme-type phrase
    embedded in its own text. Returns None if no known phrase is found,
    so the caller can fall back to the row's "Asset Class" value."""
    text = str(raw_subcat).lower()
    for asset_type, keywords in _ASSET_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return asset_type
    return None


def _build_asset_type_subcat_map(
    df: pd.DataFrame, asset_types: list[str]
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build Asset Type -> [Sub Category, ...] for the category
    directory. Each cleaned display label is collapsed to a single
    representative raw value per Asset Type (fixes exact duplicates),
    near-duplicate labels are further folded together (fixes wording
    variants like "Banking and PSU Fund" vs "Banking and PSU Debt
    Fund"), and each Sub Category is bucketed by its own embedded
    scheme-type phrase rather than the sheet's "Asset Class" column
    (fixes misplacement).

    Also returns raw_to_canonical: every distinct raw Sub Category value
    that appears in the dataset, mapped to whichever raw value ended up
    representing its (possibly merged) directory entry. Callers that
    re-key a fund's own row onto the directory's canonical value (e.g.
    app.py, so the site's client-side category grouping matches this
    directory) need this FULL mapping -- not just the raws that ended up
    in the final directory list -- otherwise a fund whose raw label was
    the "losing" side of a near-duplicate merge silently keeps its own,
    un-merged raw value instead of joining the merged category."""
    # asset_type -> {cleaned_label: (raw_value_with_max_count, fund_count)}
    buckets: dict[str, dict[str, tuple[str, int]]] = {a: {} for a in asset_types}
    # asset_type -> {cleaned_label: [every raw value that cleans to this label]}
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

        # merged only keeps the SURVIVING labels; walk every label that
        # was folded into a survivor (including ones merged away by
        # _merge_similar_labels, which aren't keys in `merged` at all)
        # and point every raw value under every label at the survivor's
        # canonical raw.
        label_to_canonical_raw: dict[str, str] = {}
        for surviving_label, (canon_raw, _count) in merged.items():
            label_to_canonical_raw[surviving_label] = canon_raw
        for label in by_label:
            if label not in label_to_canonical_raw:
                # This label was folded into some other (surviving) label
                # by _merge_similar_labels -- find which one it maps to by
                # re-running the same similarity check against survivors.
                for surviving_label, canon_raw in list(label_to_canonical_raw.items()):
                    if fuzzy_ratio(label.lower(), surviving_label.lower()) >= _LABEL_MERGE_THRESHOLD:
                        label_to_canonical_raw[label] = canon_raw
                        break
                else:
                    # Shouldn't happen, but fall back to its own raw
                    # rather than dropping the fund from the map.
                    label_to_canonical_raw[label] = by_label[label][0]
        for label, raw_list in raws_by_label[asset_type].items():
            canon_raw = label_to_canonical_raw.get(label, by_label[label][0])
            for raw in raw_list:
                raw_to_canonical[raw] = canon_raw

    return result, raw_to_canonical


def _fund_link(name: str) -> str:
    """Raw HTML anchor (not markdown syntax) so we can force target="_self" --
    plain markdown '[text](url)' links get target="_blank" forced on them by
    Streamlit's renderer, which would open a new tab instead of resolving in
    the same page."""
    safe_name = html.escape(str(name))
    return f'<a href="?fund={quote(str(name))}" target="_self">{safe_name}</a>'


# ----------------------------------------------------------------------
# Compact plain-text metrics dump fed to the AI risk-summary call (see
# llm_fallback.get_fund_risk_summarizer()). Deliberately not the full
# rendered markdown profile -- just the handful of numbers that actually
# drive a risk read: multi-horizon return/CAGR, the standard risk/
# risk-adjusted-return metrics, peer percentile ranks, and the overall
# composite score/rank. Percent-style fields are pre-converted here too,
# so the LLM sees "16.03%" rather than a raw 0.1603 it would have to
# reinterpret itself.
# ----------------------------------------------------------------------
_SUMMARY_PCT_SUFFIXES = {
    "AbsoluteReturn", "CAGR", "Volatility", "MaxDrawdown",
    "DownsideDev", "VaR95", "RollMean", "RollMin", "RollMax",
}
# AI verdict is scoped to 3-year metrics only -- the fund profile page
# still shows 1Y/3Y/5Y for reference, but the summary/verdict call gets
# just the 3Y horizon (return/CAGR, volatility, drawdown, risk-adjusted
# ratios) plus the already-3Y-based peer percentile ranks and Composite
# Score, so the model can't lean on a strong/weak 1Y or 5Y number.
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
        # Conversation state kept only for the legacy multi-step
        # "which fund did you mean" disambiguation flow (see
        # detect_intent / respond below); the website UI drives
        # navigation itself and doesn't rely on this for category
        # browsing anymore.
        self.pending: dict | None = None

    # ------------------------------------------------------------------
    # Data loading -- always the fixed static file/sheet, no overrides.
    # ------------------------------------------------------------------
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

        # Repair mojibake baked into the source cells (see _fix_mojibake)
        # before anything else touches the text -- whitespace-trim,
        # canonicalization, and every downstream table/link/AI-summary
        # input should all see the corrected text.
        #
        # NOTE: pandas 3.x defaults text columns to its own "string" dtype
        # rather than "object" -- checking `dtype == object` here silently
        # skipped every column and let the mojibake straight through, so
        # this uses is_string_dtype() to catch both.
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].map(_fix_mojibake)

        # Strip stray leading/trailing whitespace from the text fields the
        # category directory groups on. Untrimmed whitespace (e.g. a sheet
        # value of "Equity " next to "Equity") would otherwise be treated
        # as a distinct value and produce the same duplicate/misplaced
        # symptom as the Open/Close-Ended wrapper issue below.
        for col in ("Sub Category", "Asset Class"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)

        # Merge known duplicate Sub Category spellings (e.g. plain "ELSS"
        # vs. "ELSS Tax Saver" / "ELSS Tax Saver Fund") onto one canonical
        # value. This MUST run before _sub_categories / the Asset Type map
        # are built below, since everything downstream keys off this
        # column verbatim.
        if "Sub Category" in df.columns:
            df["Sub Category"] = df["Sub Category"].apply(_canonicalize_subcat)

        self.df = df
        self._scheme_names = df["Scheme Name"].astype(str).tolist()
        self._sub_categories = sorted(df["Sub Category"].dropna().unique().tolist())

        # Build Asset Type -> [Sub Category, ...] mapping for the
        # category directory.
        if "Asset Class" in df.columns:
            self._asset_types = sorted(df["Asset Class"].dropna().unique().tolist())
            self._asset_type_to_subcats, self._subcat_canonical_map = _build_asset_type_subcat_map(
                df, self._asset_types
            )
        else:
            # No Asset Class column -> treat everything as one bucket.
            self._asset_types = ["All Funds"]
            self._asset_type_to_subcats = {"All Funds": self._sub_categories}
            self._subcat_canonical_map = {sc: sc for sc in self._sub_categories}

        # Local NLP: AMC / fund-house matcher, built from the dataset's own
        # "AMC (Fund House)" column values -- never a hardcoded list -- so
        # a query like "HDFC Small cap funds" can have "HDFC" recognized
        # and split off from the category text instead of silently
        # diluting (or being dropped from) the fuzzy category match. See
        # nlp_utils.AMCMatcher and _extract_amc_and_rest() below.
        if "AMC (Fund House)" in df.columns:
            amc_values = df["AMC (Fund House)"].dropna().astype(str).unique().tolist()
        else:
            amc_values = []
        self._amc_matcher = AMCMatcher(amc_values)

        # Kept for backward compatibility with any external code that
        # imported this attribute directly; AMCMatcher now does the actual
        # extraction work.
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
        """Every raw Sub Category value seen in the dataset, mapped to
        the raw value that represents its (possibly near-duplicate-
        merged) entry in asset_type_to_subcats. Use this to re-key a
        fund's own raw Sub Category before grouping/filtering by
        category, so funds whose raw label was the "losing" side of a
        merge (e.g. "Banking and PSU Debt Fund" folded into "Banking and
        PSU Fund") still land in the merged category instead of a
        leftover, no-longer-listed one."""
        return self._subcat_canonical_map

    def fund_count(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    # AMC extraction -- pulls a recognized AMC / fund-house name out of
    # free text and returns (amc_or_None, remaining_text). Delegates to
    # nlp_utils.AMCMatcher, which is seeded from the dataset itself.
    # ------------------------------------------------------------------
    def _extract_amc_and_rest(self, query: str) -> tuple[str | None, str]:
        return self._amc_matcher.extract(query)

    def _funds_for_amc(self, subset: pd.DataFrame, amc_text: str) -> pd.DataFrame:
        """Filter `subset` down to rows whose AMC (Fund House) best matches
        `amc_text` (fuzzy, since a query's "HDFC" needs to match a full
        column value like "HDFC Mutual Fund"). Returns the filtered rows,
        or an empty frame if no AMC in the subset scores above threshold."""
        amc_col = "AMC (Fund House)"
        if amc_col not in subset.columns or subset.empty:
            return subset.iloc[0:0]
        amc_values = subset[amc_col].dropna().astype(str).unique().tolist()
        # rapidfuzz's scorers are CASE-SENSITIVE ('Quant' vs 'quant' scores
        # far lower than a same-case comparison would suggest). The dataset
        # has at least one AMC name stored fully lowercase ("quant Mutual
        # Fund", vs. e.g. "Quantum Mutual Fund" or "HDFC Mutual Fund"), so
        # matching the raw (title-cased, as typed/extracted) query text
        # against raw AMC values let a query for "Quant" score HIGHER
        # against the unrelated "Quantum Mutual Fund" (case matches) than
        # against the actually-intended "quant Mutual Fund" (case
        # mismatch) -- silently returning the wrong AMC's funds, or an
        # empty result if the wrong AMC's rows don't overlap the requested
        # category. Lowercase both sides purely for scoring; the ORIGINAL
        # (correctly-cased) amc_values entries are still what's returned
        # and used to filter, via the lower->original lookup map.
        lower_to_original: dict[str, str] = {}
        for v in amc_values:
            lower_to_original.setdefault(v.lower(), v)
        amc_values_lower = list(lower_to_original.keys())
        # token_set_ratio (not the default token_sort_ratio) here: the
        # query is a short brand word ("HDFC") being matched against a
        # much longer full legal name ("HDFC Mutual Fund"). token_set_ratio
        # scores a query whose words are a subset of the candidate's words
        # highly regardless of the length difference; token_sort_ratio
        # penalizes that length mismatch and would wrongly score this low
        # (verified: 'HDFC' vs 'HDFC Mutual Fund' -> 40% on token_sort_ratio
        # but 100% on token_set_ratio).
        best_lower, score = best_fuzzy_match(
            amc_text.lower(), amc_values_lower, scorer=fuzz.token_set_ratio
        )
        if best_lower is None or score < 0.6:
            return subset.iloc[0:0]
        best = lower_to_original[best_lower]
        return subset[subset[amc_col] == best]

    # ------------------------------------------------------------------
    # Sub-category matching -- fuzzy, highest-score based (no exact text
    # required). Matching runs against the *cleaned* display label (the
    # "Open/Close Ended Schemes" wrapper stripped out), not the raw
    # dataset string. Comparing against the raw value let its wrapper text
    # dilute the old character-sequence ratio -- a long, correct raw value
    # like "Open Ended Schemes(Equity Scheme - Small Cap Fund)" could
    # score WORSE against a short query than an unrelated but
    # coincidentally short raw label with no scheme-type prefix (e.g.
    # "Open Ended Schemes(IL&FS Mutual Fund IDF)"), purely because of
    # string-length mismatch, not actual relevance. The raw Sub Category
    # value is still what gets returned/used for filtering -- only the
    # comparison text changes.
    #
    # Scoring uses rapidfuzz's token_sort_ratio (nlp_utils.fuzzy_ratio) --
    # a bag-of-words comparison, unlike difflib's raw character-sequence
    # ratio -- so a query with extra words still scores well against the
    # right label as long as the important words match.
    # Returns the best matching (raw) Sub Category and its score.
    # ------------------------------------------------------------------
    # Sub Categories wrapped as "Close Ended Schemes(...)" are legacy /
    # matured buckets that are no longer open for fresh investment (e.g.
    # decades-old fixed-tenure ELSS or Growth/Income plans). A query like
    # "ELSS" almost always means the actively-investable, open-ended ELSS
    # Tax Saver funds -- but that category's cleaned label is "Equity
    # Scheme - ELSS", NOT a bare "ELSS", so a plain "ELSS" query only
    # exact-string-matches the closed-ended bucket ("Close Ended
    # Schemes(ELSS)" cleans to just "ELSS"). Left unhandled, that exact
    # match (score 1.0) beat the open-ended category's substring-boosted
    # score (0.85) purely by string-matching luck -- so "Quant ELSS"
    # resolved to the closed-ended ELSS bucket, which doesn't contain
    # quant Mutual Fund's (or almost any AMC's) ELSS fund at all, and the
    # bot reported no results even though the fund exists.
    #
    # Fix: exclude closed-ended sub-categories from the default match
    # pool -- unless the query itself explicitly asks for "close(d)
    # ended" funds, or excluding them would leave nothing to match at all
    # (some categories genuinely only exist as closed-ended).
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
                # Don't return on the FIRST exact match -- several raw Sub
                # Category values can clean/normalize down to the same
                # label (e.g. "Open Ended Schemes(Growth)" and, before the
                # closed-ended filter above, "Close Ended Schemes(Growth)"
                # both clean to just "Growth"). Collect every exact match
                # and break ties below instead of taking whichever happens
                # to be first.
                exact_matches.append(sc)
                continue

            score = fuzzy_ratio(q, label_l)
            # Boost substring / core-word matches (e.g. "large cap" inside
            # "Large Cap Fund", or "small cap fund" -- after pluralization
            # is normalized above -- inside "Small Cap Fund"). Scaled by
            # how much of the label's own distinguishing (non-boilerplate)
            # vocabulary the query actually covers -- a flat boost here
            # previously gave "Mid Cap Fund" and "Large & Mid Cap Fund"
            # the exact same score for a "mid cap" query (it's a literal
            # substring of both), so the tie silently fell to whichever
            # category happened to come first in the dataset instead of
            # the one that's actually the better match.
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
            # Lower-confidence fallback boost for the plain substring case,
            # in case tokenization (hyphens, "&", etc.) kept the two sides
            # from lining up as cleanly as the core-token check above wants.
            if q in label_l or label_l.replace(" fund", "").strip() in q:
                score = max(score, 0.7)

            if score > best_score:
                best_sc, best_score = sc, score

        if exact_matches:
            # Same tie-break heuristic as _build_asset_type_subcat_map():
            # prefer whichever raw value is backed by the most funds in
            # the dataset, so a big category always wins over a small one
            # with the same cleaned label.
            best_exact = max(
                exact_matches,
                key=lambda sc: int((self.df["Sub Category"] == sc).sum()),
            )
            return best_exact, 1.0

        return best_sc, best_score

    def match_sub_categories(self, query: str, limit: int = 3) -> list[str]:
        """Kept for backward compatibility: returns a short ranked list
        of raw Sub Category values. Uses the same cleaned-label / plural-
        normalized comparison as best_sub_category_match() so results here
        stay consistent with what a single lookup would pick."""
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

    # ------------------------------------------------------------------
    # Fund name matching
    # ------------------------------------------------------------------
    def get_fund_group(self, scheme_name: str) -> tuple[pd.Series, list[pd.Series]]:
        """Given an exact Scheme Name, return (row, siblings) where
        siblings is every OTHER row in the full dataset that represents a
        different Plan/Option/IDCW-frequency of the SAME underlying fund
        (matched via _fund_identity_key, the same identity key
        group_funds_with_variants() uses). Lets format_fund_profile()
        show the full "Available Plans & Options" breakdown for a fund
        looked up through the conversational respond()/detect_intent path,
        not just the website's own JSON-record path (app.py already does
        this grouping itself via group_funds_with_variants())."""
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
        """Return several close candidates (used when an exact pick is ambiguous)."""
        q = query.strip().lower()
        contains = self.df[self.df["Scheme Name"].str.lower().str.contains(re.escape(q), na=False)]
        if len(contains):
            return contains.head(n)
        matches = ranked_fuzzy_matches(query, self._scheme_names, limit=n, score_cutoff=score_cutoff)
        names = [name for name, _score in matches]
        return self.df[self.df["Scheme Name"].isin(names)]

    def match_funds_ranked(self, query: str, n: int = 6) -> pd.DataFrame:
        """Rank every fund in the dataset by similarity to a free-text
        search and return up to n rows, best match first. Used so a fund
        search always surfaces every close candidate for the user to pick
        from -- instead of silently auto-selecting whichever single match
        scores highest, which can guess wrong when several funds share
        very similar names (different AMCs' "... ELSS Tax Saver Fund",
        Direct vs Regular plan, Growth vs IDCW, etc.).

        Uses rapidfuzz's token_set_ratio, which treats the query and each
        scheme name as bags of words -- so a short query like "hdfc flexi
        cap" scores well against the much longer "HDFC Flexi Cap Fund -
        Direct Plan - Growth" without needing a manual substring-boost
        special case."""
        q = (query or "").strip()
        if not q:
            return self.df.iloc[0:0]

        # A higher bar than the Sub Category matcher's (0.35): fund names
        # share a lot of boilerplate ("Fund", "Direct Plan", "Growth"), so
        # a loose cutoff pulls in unrelated AMCs' funds just because they're
        # in the same category (e.g. searching "SBI Flexi Cap Fund" would
        # otherwise also surface every other house's Flexi Cap fund). 0.55
        # keeps genuine near-duplicates (same fund, different plan/option)
        # while dropping same-category noise.
        matches = ranked_fuzzy_matches(q, self._scheme_names, limit=n, score_cutoff=0.55)
        if not matches:
            return self.df.iloc[0:0]
        ordered_names = [name for name, _score in matches]
        # Preserve rank order (rapidfuzz already sorts best-first, but
        # DataFrame.isin() doesn't preserve it) and dedupe if the same
        # scheme name appears more than once in the raw data.
        rows = self.df[self.df["Scheme Name"].isin(ordered_names)]
        rows = rows.drop_duplicates(subset="Scheme Name")
        rank = {name: i for i, name in enumerate(ordered_names)}
        rows = rows.assign(_rank=rows["Scheme Name"].map(rank)).sort_values("_rank")
        return rows.drop(columns="_rank").head(n)

    # ------------------------------------------------------------------
    # Search-box typeahead -- combined fund + category suggestions
    # ------------------------------------------------------------------
    def search_suggestions(self, query: str, limit: int = 8) -> list[dict]:
        """Ranked suggestions for the website's search box, mixing
        matching fund names and Sub Categories so one box can jump
        straight to either a fund's profile or a category's top-10 list
        -- the way a Coin/Kite-style instrument search box works.

        Each entry: {"type": "fund"|"category", "label": display text,
        "value": the exact string to look up (raw Scheme Name for a
        fund, raw Sub Category for a category)}.
        """
        q = (query or "").strip()
        if not q:
            return []

        scored: list[tuple[float, dict]] = []

        # rapidfuzz's scorers are case-sensitive, so a lowercase-typed
        # query like "quant" would otherwise score poorly against a
        # candidate whose brand word is capitalized differently (e.g.
        # "Quant ELSS Tax Saver Fund" vs the AMC's own literal "quant
        # Mutual Fund" casing elsewhere in the sheet). Score in
        # lowercase, then map back to the original (correctly-cased)
        # scheme name for display/lookup.
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

        # Dedupe categories/funds that share a display label (keep the
        # highest-scoring raw value for each), then take the overall top
        # `limit` across both types, best score first -- funds and
        # categories are interleaved by relevance rather than grouped, so
        # the single best match (of either kind) always shows up first.
        best_by_key: dict[tuple[str, str], tuple[float, dict]] = {}
        for score, entry in scored:
            key = (entry["type"], entry["label"])
            current = best_by_key.get(key)
            if current is None or score > current[0]:
                best_by_key[key] = (score, entry)

        ranked = sorted(best_by_key.values(), key=lambda t: t[0], reverse=True)
        return [entry for _score, entry in ranked[:limit]]

    # ------------------------------------------------------------------
    # Top-N funds in a sub-category
    # ------------------------------------------------------------------
    def top_funds(
        self, sub_category: str, n: int = 5, sort_by: str = "Peer_Rank",
        amc: str | None = None,
    ) -> pd.DataFrame:
        subset = self.df[self.df["Sub Category"] == sub_category].copy()
        if subset.empty:
            return subset

        # Filter to a specific AMC / fund house before dedup/ranking, if
        # one was recognized in the query (e.g. "HDFC Small cap funds").
        if amc:
            subset = self._funds_for_amc(subset, amc)
            if subset.empty:
                return subset

        subset = dedup_funds_keep_direct(subset)

        # Only one fund per AMC in the displayed set -- if two funds from the
        # same fund house both qualify, keep just the better-ranked one
        # rather than showing the AMC twice. Skipped when the caller has
        # already filtered to a single AMC (amc is set), since collapsing
        # to "one per AMC" there would incorrectly cut an AMC's results
        # down to a single fund.
        #
        # This IS a backfill: the table always fills to n funds (subject to
        # there being n distinct-AMC funds in the category at all) by
        # walking down the ascending Peer_Rank / Composite_Score order past
        # rank n if dropping same-AMC and same-underlying-fund duplicates
        # left the top-n band short. Peer_Rank itself is not shown as a
        # column in the rendered table (see format_top_funds /
        # TOP_N_METRIC_SPECS) -- it's used here purely as the backend
        # ordering, ascending (best rank first).
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

    # ------------------------------------------------------------------
    # Formatting: top-N table -> markdown
    # ------------------------------------------------------------------
    def _resolve_top_n_metric_cols(self, subset: pd.DataFrame) -> list[tuple[str, str]]:
        """For each entry in TOP_N_METRIC_SPECS, pick whichever of
        (primary, fallback) column actually exists in this dataset and
        pair it with its display label. Horizons with neither column
        present are skipped entirely rather than rendered blank."""
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

        # Rendered as a stack of cards instead of a wide markdown table --
        # a 6+ column table only fits a phone screen by squeezing text or
        # forcing horizontal scroll, neither of which is a good mobile
        # experience. Each fund becomes one card: rank + name up top, then
        # its return figures as small label/value chips that wrap onto as
        # many rows as the screen needs, so nothing is ever clipped or
        # scrolled sideways. Colour-coding the sign (green/red) also makes
        # the numbers scannable at a glance instead of requiring a careful
        # read of the '-' sign. Relies on the .ff-fundlist/.ff-fundcard/
        # .ff-metric CSS in app.py and on pages rendering with
        # unsafe_allow_html=True (see app.py).
        cards = ['<div class="ff-fundlist">']
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            name_html = _fund_link(row["Scheme Name"])
            chips = []
            for col, label in metric_cols:
                period = label.split(" ", 1)[0]  # "1D Obs. Return" -> "1D"
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

    # ------------------------------------------------------------------
    # Formatting: full fund profile -> markdown
    # ------------------------------------------------------------------
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

        # Basic Information -- bullet list of BASIC_COLS fields (Scheme
        # Code, AMC, Sub Category, Asset Class, NAV + date). A 2-column
        # Field/Value table would force a fixed-width left column that
        # can crowd a narrow phone screen, where a bullet list just wraps.
        for c in BASIC_COLS:
            if c == "Scheme Name" or c not in row.index:
                continue
            v = row[c]
            if pd.isna(v):
                continue
            # The ELSS flag is only worth showing when it's actually
            # True -- False just means "not an ELSS fund", which is true
            # for the vast majority of funds and reads as confusing,
            # content-free noise ("ELSS: False") on every non-ELSS
            # fund's profile. Skip the line entirely in that case.
            if c == "ELSS":
                # Compared via str(v) rather than "v is True" -- pandas
                # often stores this column as numpy.bool_ (or, if the
                # sheet has any blank cells in the column, as plain
                # object/string values), and numpy.bool_(True) is NOT
                # Python's True by identity, which silently dropped the
                # "Yes" line for every genuinely-ELSS fund read from a
                # real spreadsheet even though the value was truthy.
                if str(v).strip().lower() in ("true", "yes", "y", "1", "1.0"):
                    out.append("- **ELSS:** Yes")
                continue
            label = "Sub Category" if c == "Sub Category" else c
            value = clean_subcat_label(str(v)) if c == "Sub Category" else fmt(v)
            out.append(f"- **{label}:** {value}")
        out.append("")

        # Every horizon (1D through SI) and every metric -- each horizon
        # is a tap-to-expand <details> section instead of a permanently-
        # open table, so nine stacked tables don't turn into one long
        # wall of scrolling on a phone. The <summary> line shows the
        # headline return so a reader can scan all horizons at a glance
        # and only expand the ones they want the full risk breakdown
        # for. Relies on pages rendering with unsafe_allow_html=True.
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

            # Custom-styled <details>/<summary> instead of the bare browser
            # default (a tiny triangle glyph that's easy to miss and gives
            # no hover/tap feedback) -- the .ff-horizon CSS in app.py turns
            # this into a card with a clear "Return" label, a colour-coded
            # value, and a chevron that visibly rotates on open, so it
            # reads as an obviously tappable row instead of plain text.
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

        # Available Plans & Options -- EVERY purchasable Plan/Option/
        # frequency combination of this underlying fund, including the
        # one currently being viewed (row itself), grouped by Plan so
        # the structure reads as:
        #   Direct Plan
        #     - Growth
        #     - Monthly IDCW
        #   Regular Plan
        #     - Growth
        #     - Income Distribution cum Capital Withdrawal (Quarterly)
        # rather than a flat "Additional investment options" list that
        # only covered the OTHER rows and omitted the primary's own
        # plan/option from the same structured breakdown.
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

            # Direct first (what most retail investors want), then
            # Regular, then anything unclassified.
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

        # AI Verdict is always rendered as a standard section on every
        # fund's profile -- not just when a summarizer happens to be
        # wired up -- so it reads as a permanent, expected part of every
        # fund page rather than a feature that sometimes silently isn't
        # there. If no GROQ_API_KEY is configured, the section still
        # shows, with a plain note instead of a generated verdict.
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

    # ------------------------------------------------------------------
    # Legacy guided "Asset Type -> Sub Category" flow, retained only for
    # the free-text disambiguation prompt used by respond()/detect_intent
    # (e.g. "a few funds match X -- which one?"). The website UI (app.py)
    # drives category browsing itself via asset_type_to_subcats and does
    # not depend on this state machine.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Intent detection (rule based, no LLM required) + local NLP entity
    # extraction (AMC name, category text, spelled-out counts). Retained
    # for advanced free-text queries typed directly into the search box
    # (e.g. "top 10 mid cap funds").
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Main entry point (kept for free-text queries typed into the
    # search box that don't come from a clicked suggestion).
    # ------------------------------------------------------------------
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
