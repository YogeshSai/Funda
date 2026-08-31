"""
app.py
------
Streamlit entry point for FundFinder.
"""
import json
import math
import pathlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from finance_bot import (
    FinanceBot,
    clean_subcat_label,
    describe_variant_fields,
    describe_variant_label,
    group_funds_with_variants,
    subcat_browse_label,
)

st.set_page_config(page_title="MIRA-AI", page_icon="👑", layout="wide")

# Streamlit wraps every page in its own chrome (top padding, a hidden
# hamburger menu, a "Made with Streamlit" footer). Hidden here so the
# embedded site can use the full browser window like a normal
# standalone website rather than sitting inside a framed widget.
#
# NOTE: these were previously hidden with `visibility: hidden`, which
# keeps an element's box in the page's layout flow -- it just makes the
# box invisible, it does NOT collapse the space that box reserves. That
# left Streamlit's own header/footer chrome silently reserving a strip
# of blank space above/below our embedded iframe on every page, which
# read as "extra empty space at the bottom" especially on shorter pages
# like Browse by Category. `display: none` removes the element from
# layout entirely, so no reserved space is left behind.
st.markdown(
    """
    <style>
      #MainMenu, header, footer {display: none !important;}
      .block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
      iframe {width: 100%; border: none; display: block;}
    </style>
    """,
    unsafe_allow_html=True,
)

HTML_PATH = pathlib.Path(__file__).parent / "fundfinder.html"
FUND_DATA_START_MARKER = "/*__FUND_DATA_JSON__*/"
FUND_DATA_END_MARKER = "/*__END_FUND_DATA_JSON__*/"

RETURN_HORIZONS = ["1D", "6M", "1Y", "3Y", "5Y"]
RETURN_FALLBACK_COL = {
    "1Y": "1Y_CAGR",
    "3Y": "3Y_CAGR",
    "5Y": "5Y_CAGR",
}

TOP_N_PER_CATEGORY = 5


def _num_or_none(v):
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


def _build_category_lookups(bot: FinanceBot) -> dict:
    asset_type_by_raw: dict = {}
    for asset_type, subcats in bot.asset_type_to_subcats.items():
        for sc in subcats:
            asset_type_by_raw[sc] = asset_type
    return asset_type_by_raw


def build_fund_records(bot: FinanceBot) -> list[dict]:
    groups = group_funds_with_variants(bot.df.copy())
    asset_type_by_raw = _build_category_lookups(bot)
    canonical_map = bot.subcat_canonical_map

    def _returns_for(row) -> dict:
        returns = {}
        for horizon in RETURN_HORIZONS:
            col = f"{horizon}_AbsoluteReturn"
            if col not in row.index or pd.isna(row.get(col)):
                col = RETURN_FALLBACK_COL.get(horizon)
            returns[horizon] = _num_or_none(row.get(col)) if col else None
        return returns

    def _plan_entry(r, is_current: bool) -> dict:
        fields = describe_variant_fields(r.get("Scheme Name", ""))
        return {
            "name": str(r.get("Scheme Name", "")),
            "plan": fields["plan"],
            "option": fields["option"],
            "frequency": fields["frequency"],
            "nav": _num_or_none(r.get("Latest NAV")),
            "isCurrent": is_current,
        }

    records = []
    for group in groups:
        row = group["primary"]
        sub_cat_raw = row.get("Sub Category")
        if pd.isna(sub_cat_raw):
            continue

        sub_cat_raw = canonical_map.get(sub_cat_raw, sub_cat_raw)
        label = subcat_browse_label(sub_cat_raw)

        peer_pctile_col = "3Y_CAGR_PeerPctile"
        peer_pctile = _num_or_none(row.get(peer_pctile_col))
        if peer_pctile is not None:
            peer_pctile = round(peer_pctile * 100)

        peer_rank = _num_or_none(row.get("Peer_Rank"))

        additional_options = [
            {
                "label": describe_variant_label(v.get("Scheme Name", "")),
                "name": str(v.get("Scheme Name", "")),
                "nav": _num_or_none(v.get("Latest NAV")),
            }
            for v in group["variants"]
        ]

        plans = [_plan_entry(row, is_current=True)] + [
            _plan_entry(v, is_current=False) for v in group["variants"]
        ]

        records.append({
            "name": str(row.get("Scheme Name", "")),
            "amc": str(row.get("AMC (Fund House)", "")) if not pd.isna(row.get("AMC (Fund House)")) else "",
            "assetType": asset_type_by_raw.get(sub_cat_raw, str(row.get("Asset Class", "Other"))),
            "subCategoryRaw": str(sub_cat_raw),
            "subCategoryLabel": label,
            "nav": _num_or_none(row.get("Latest NAV")),
            "returns": _returns_for(row),
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
            "additionalOptions": additional_options,
            "plans": plans,
        })
    return records


@st.cache_data(show_spinner=False)
def load_fund_json() -> str:
    bot = FinanceBot()
    records = build_fund_records(bot)
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

components.html(html_code, height=1100, scrolling=False)
