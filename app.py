"""
app.py
------
Streamlit entry point for FundFinder.

The site itself (search box with live, as-you-type suggestions,
category browser, fund profile pages) lives in fundfinder.html, a
single self-contained HTML/CSS/JS file sitting next to this script.
It stays a plain HTML/JS page -- rather than being rebuilt as
st.text_input + st.button widgets -- because Streamlit reruns the
whole script on every widget interaction, which is too slow/round-
trippy for instant, character-by-character search suggestions. This
file's job is to (1) load & shape the real fund data using
finance_bot.FinanceBot, (2) inject that data as JSON into the
fundfinder.html marker, and (3) embed the resulting page via
st.components.v1.html so the browser runs it exactly as it would
standalone.

Folder layout expected:
    app.py                <- this file
    fundfinder.html       <- the website
    finance_bot.py        <- data loading / shaping logic
    nlp_utils.py
    llm_fallback.py
    MF_Risk_Metrics.xlsx
    requirements.txt

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push all files (plus requirements.txt) to a GitHub repo, then
    point Community Cloud at app.py -- no other setup needed.
"""
import json
import math
import pathlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from finance_bot import FinanceBot, clean_subcat_label, dedup_funds, subcat_browse_label

st.set_page_config(page_title="FundFinder", page_icon="📈", layout="wide")

# Streamlit wraps every page in its own chrome (top padding, a hidden
# hamburger menu, a "Made with Streamlit" footer). Hidden here so the
# embedded site can use the full browser window like a normal
# standalone website rather than sitting inside a framed widget.
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
      iframe {width: 100%; border: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

HTML_PATH = pathlib.Path(__file__).parent / "fundfinder.html"
FUND_DATA_START_MARKER = "/*__FUND_DATA_JSON__*/"
FUND_DATA_END_MARKER = "/*__END_FUND_DATA_JSON__*/"

# Horizons the site's fund cards / profile pages show returns for.
RETURN_HORIZONS = ["1D", "6M", "1Y", "3Y", "5Y"]
# For horizons where the sheet only has an annualised CAGR (no observed
# absolute return), fall back to CAGR rather than showing a blank.
RETURN_FALLBACK_COL = {
    "1Y": "1Y_CAGR",
    "3Y": "3Y_CAGR",
    "5Y": "5Y_CAGR",
}


def _num_or_none(v):
    """Convert a pandas scalar to a JSON-safe float or None."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _build_asset_type_lookup(bot: FinanceBot) -> dict:
    """Invert FinanceBot's Asset Type -> [Sub Category,...] map so every
    raw Sub Category value in the dataset can be looked up directly.
    Falls back to the row's own Asset Class for any Sub Category that
    the directory-building step didn't end up assigning (e.g. when two
    raw values collapse to the same cleaned label and only one is kept
    as that label's representative)."""
    lookup = {}
    for asset_type, subcats in bot.asset_type_to_subcats.items():
        for sc in subcats:
            lookup[sc] = asset_type
    return lookup


def build_fund_records(bot: FinanceBot) -> list[dict]:
    df = dedup_funds(bot.df.copy())
    asset_type_lookup = _build_asset_type_lookup(bot)

    records = []
    for _, row in df.iterrows():
        sub_cat_raw = row.get("Sub Category")
        if pd.isna(sub_cat_raw):
            continue

        returns = {}
        for horizon in RETURN_HORIZONS:
            col = f"{horizon}_AbsoluteReturn"
            if col not in row.index or pd.isna(row.get(col)):
                col = RETURN_FALLBACK_COL.get(horizon)
            returns[horizon] = _num_or_none(row.get(col)) if col else None

        peer_pctile_col = "3Y_CAGR_PeerPctile"
        peer_pctile = _num_or_none(row.get(peer_pctile_col))
        if peer_pctile is not None:
            peer_pctile = round(peer_pctile * 100)

        peer_rank = _num_or_none(row.get("Peer_Rank"))

        records.append({
            "name": str(row.get("Scheme Name", "")),
            "amc": str(row.get("AMC (Fund House)", "")) if not pd.isna(row.get("AMC (Fund House)")) else "",
            "assetType": asset_type_lookup.get(sub_cat_raw, str(row.get("Asset Class", "Other"))),
            "subCategoryRaw": str(sub_cat_raw),
            "subCategoryLabel": subcat_browse_label(sub_cat_raw),
            "nav": _num_or_none(row.get("Latest NAV")),
            "returns": returns,
            "risk": {
                "vol": _num_or_none(row.get("3Y_Volatility")),
                "mdd": _num_or_none(row.get("3Y_MaxDrawdown")),
                "sharpe": _num_or_none(row.get("3Y_Sharpe")),
                "sortino": _num_or_none(row.get("3Y_Sortino")),
                "calmar": _num_or_none(row.get("3Y_Calmar")),
            },
            "peerPctile": peer_pctile,
            "compositeScore": _num_or_none(row.get("Composite_Score")),
            "peerRank": int(peer_rank) if peer_rank is not None else None,
        })
    return records


@st.cache_data(show_spinner=False)
def load_fund_json() -> str:
    bot = FinanceBot()
    records = build_fund_records(bot)
    # Escape "</" so a stray "</script>"-like substring in any fund name
    # can't break out of the inline <script> tag it's embedded in.
    return json.dumps(records, ensure_ascii=False).replace("</", "<\\/")


try:
    html_code = HTML_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    st.error(
        "Couldn't find fundfinder.html next to app.py. Make sure both "
        "files are in the same folder before running `streamlit run app.py`."
    )
    st.stop()

start_idx = html_code.find(FUND_DATA_START_MARKER)
end_idx = html_code.find(FUND_DATA_END_MARKER)
if start_idx == -1 or end_idx == -1:
    st.error(
        "Couldn't find the fund-data markers in fundfinder.html -- the "
        "page's JS expects "
        f"'{FUND_DATA_START_MARKER}...{FUND_DATA_END_MARKER}' so app.py "
        "can inject real data in their place."
    )
    st.stop()

try:
    fund_json = load_fund_json()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ValueError as e:
    st.error(f"Dataset problem: {e}")
    st.stop()

marker_content_start = start_idx + len(FUND_DATA_START_MARKER)
html_code = (
    html_code[:marker_content_start]
    + fund_json
    + html_code[end_idx:]
)

# height is generous and fixed, with scrolling enabled inside the
# iframe -- the page's own content (category directory, expanded fund
# lists, etc.) scrolls within that frame the same way it would in a
# normal browser tab.
components.html(html_code, height=2400, scrolling=True)
