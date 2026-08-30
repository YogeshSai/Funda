"""
app.py
------
Streamlit frontend for FundFinder — a search-first mutual fund
analytics website, styled after the simple, clean instrument-search
layout of Zerodha Coin: a single search box up top with live
suggestions, a category directory below it, and a fund page / category
results page that opens when you pick something. No chat, no message
history — just search, browse, and read.

Run:
    streamlit run app.py
"""

import streamlit as st

from finance_bot import FinanceBot, clean_subcat_label, subcat_browse_label
from llm_fallback import get_llm_fallback, get_fund_risk_summarizer

st.set_page_config(page_title="FundFinder", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------
# Style — light, plain, high-contrast. Coin-style: white background,
# a single green accent, thin borders instead of shadows, generous
# whitespace. No chat bubbles, no dark theme.
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --ff-bg: #FFFFFF;
    --ff-surface: #F7F8FA;
    --ff-surface-2: #EEF1F4;
    --ff-border: #E3E6EA;
    --ff-text: #1A1F27;
    --ff-text-muted: #6B7280;
    --ff-accent: #00875A;
    --ff-accent-soft: #E6F4EE;
    --ff-radius: 10px;
}

[data-testid="stAppViewContainer"] { background: var(--ff-bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppViewContainer"] * { color: var(--ff-text); }

.main .block-container {
    max-width: 860px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

@media (max-width: 680px) {
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    div[data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
    .main .block-container { padding-left: 0.9rem; padding-right: 0.9rem; }
}

/* ---- Brand header ---- */
.ff-brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.2rem; }
.ff-brand .ff-mark {
    width: 34px; height: 34px; border-radius: 8px;
    background: var(--ff-accent); color: #FFFFFF !important;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem;
}
.ff-brand .ff-title { font-size: 1.35rem; font-weight: 700; color: var(--ff-text) !important; }
.ff-sub { color: var(--ff-text-muted) !important; font-size: 0.92rem; margin: 0.2rem 0 1.4rem 0; }

/* ---- Search box ---- */
[data-testid="stTextInput"] input {
    border: 1.5px solid var(--ff-border) !important;
    border-radius: 999px !important;
    padding: 0.75rem 1.2rem !important;
    font-size: 1rem !important;
    background: var(--ff-surface) !important;
    color: var(--ff-text) !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--ff-accent) !important;
    box-shadow: 0 0 0 3px var(--ff-accent-soft) !important;
}

/* ---- Plain flat buttons everywhere: no gradients, no shadows ---- */
div[data-testid="stButton"] button {
    width: 100%;
    text-align: left;
    border-radius: var(--ff-radius);
    border: 1px solid var(--ff-border);
    background: #FFFFFF !important;
    color: var(--ff-text) !important;
    padding: 0.55rem 0.9rem;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--ff-accent);
    background: var(--ff-accent-soft) !important;
    color: var(--ff-accent) !important;
}

/* Search-suggestion rows: compact, list-like, not full button chrome */
.ff-suggestions div[data-testid="stButton"] button {
    border: none;
    border-bottom: 1px solid var(--ff-border);
    border-radius: 0;
    background: #FFFFFF !important;
    padding: 0.65rem 0.4rem;
    font-weight: 400;
}
.ff-suggestions div[data-testid="stButton"] button:hover {
    background: var(--ff-surface) !important;
    color: var(--ff-accent) !important;
}
.ff-suggestions-box {
    border: 1px solid var(--ff-border);
    border-radius: var(--ff-radius);
    margin-top: 0.5rem;
    overflow: hidden;
    background: #FFFFFF;
}

/* ---- Category directory ---- */
.ff-section-label {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--ff-text-muted) !important; font-weight: 700;
    margin: 1.8rem 0 0.7rem 0;
}
details.ff-asset-group {
    border: 1px solid var(--ff-border);
    border-radius: var(--ff-radius);
    margin-bottom: 0.6rem;
    overflow: hidden;
    background: #FFFFFF;
}
details.ff-asset-group summary {
    list-style: none;
    cursor: pointer;
    padding: 0.8rem 1rem;
    font-weight: 600;
    display: flex; align-items: center; gap: 0.6rem;
}
details.ff-asset-group summary::-webkit-details-marker { display: none; }
details.ff-asset-group summary:hover { background: var(--ff-surface); }
details.ff-asset-group .ff-count {
    margin-left: auto; font-weight: 400; font-size: 0.8rem; color: var(--ff-text-muted) !important;
}

/* ---- Back link ---- */
.ff-back div[data-testid="stButton"] button {
    width: auto; border: none; background: transparent !important;
    color: var(--ff-accent) !important; font-size: 0.85rem; font-weight: 600;
    padding: 0.2rem 0;
}
.ff-back div[data-testid="stButton"] button:hover { background: transparent !important; text-decoration: underline; }

/* ---- Top-funds card list ---- */
.ff-fundlist { margin: 0.6rem 0 0.8rem 0; }
.ff-fundcard {
    background: #FFFFFF;
    border: 1px solid var(--ff-border);
    border-radius: var(--ff-radius);
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}
.ff-fundcard-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; }
.ff-rank {
    flex: none; width: 24px; height: 24px; border-radius: 50%;
    background: var(--ff-surface); border: 1px solid var(--ff-border);
    color: var(--ff-text-muted) !important;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700;
}
.ff-fundcard:first-child .ff-rank { color: var(--ff-accent) !important; border-color: var(--ff-accent); }
.ff-fundname { flex: 1 1 auto; min-width: 0; }
.ff-fundname a { color: var(--ff-text) !important; font-weight: 600; font-size: 0.95rem; text-decoration: none !important; }
.ff-fundname a:hover { color: var(--ff-accent) !important; text-decoration: underline !important; }
.ff-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(72px, 1fr)); gap: 0.5rem 0.4rem; }
.ff-metric { display: flex; flex-direction: column; gap: 0.15rem; background: var(--ff-surface); border-radius: 8px; padding: 0.35rem 0.5rem; }
.ff-metric-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ff-text-muted) !important; font-weight: 600; }
.ff-metric-value { font-size: 0.88rem; font-weight: 700; color: var(--ff-text) !important; }
.ff-metric-value.pos { color: #0F8A4B !important; }
.ff-metric-value.neg { color: #D14343 !important; }

/* ---- Performance & Risk horizons ---- */
.ff-hint { color: var(--ff-text-muted) !important; font-size: 0.8rem; font-style: italic; margin: 0.1rem 0 0.5rem 0; }
details.ff-horizon {
    background: #FFFFFF; border: 1px solid var(--ff-border);
    border-radius: var(--ff-radius); margin-bottom: 0.45rem; overflow: hidden;
}
details.ff-horizon summary {
    list-style: none; cursor: pointer; display: flex; align-items: center; gap: 0.6rem;
    padding: 0.65rem 0.9rem; font-weight: 600;
}
details.ff-horizon summary::-webkit-details-marker { display: none; }
details.ff-horizon summary:hover { background: var(--ff-surface); }
details.ff-horizon summary::after {
    content: ""; margin-left: auto; flex: none; width: 7px; height: 7px;
    border-right: 2px solid var(--ff-text-muted); border-bottom: 2px solid var(--ff-text-muted);
    transform: rotate(45deg); transition: transform 0.15s ease;
}
details.ff-horizon[open] summary::after { transform: rotate(-135deg); }
details.ff-horizon[open] summary { background: var(--ff-surface); border-bottom: 1px solid var(--ff-border); }
.ff-h-period {
    flex: none; background: var(--ff-surface); border: 1px solid var(--ff-border);
    border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.75rem; font-weight: 700;
    color: var(--ff-text) !important;
}
details.ff-horizon[open] .ff-h-period { background: var(--ff-accent) !important; color: #FFFFFF !important; border-color: var(--ff-accent); }
.ff-h-return-wrap { display: flex; align-items: baseline; gap: 0.4rem; }
.ff-h-return-label { color: var(--ff-text-muted) !important; font-size: 0.78rem; }
.ff-h-return { font-weight: 700; color: var(--ff-text) !important; }
.ff-h-return.pos { color: #0F8A4B !important; }
.ff-h-return.neg { color: #D14343 !important; }
details.ff-horizon ul { margin: 0; padding: 0.7rem 1rem 0.85rem 2rem; }
details.ff-horizon li { color: var(--ff-text) !important; padding: 0.15rem 0; }

/* ---- AI Verdict card ---- */
.ff-verdict-anchor + div { margin-top: 0.3rem; }
a { color: var(--ff-accent) !important; }

/* Status pill row */
.ff-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.76rem; font-weight: 600; border-radius: 999px;
    padding: 0.2rem 0.65rem; margin: 0 0.4rem 0.4rem 0;
    border: 1px solid var(--ff-border);
}
.ff-pill.on { color: var(--ff-accent) !important; border-color: var(--ff-accent); }
.ff-pill.off { color: var(--ff-text-muted) !important; }
.ff-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; margin-right: 0.2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fund dataset...")
def load_bot() -> FinanceBot:
    return FinanceBot()


try:
    bot = load_bot()
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load dataset: {e}")
    st.stop()

llm_fallback = get_llm_fallback()
fund_summarizer = get_fund_risk_summarizer()

# ---------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_subcat" not in st.session_state:
    st.session_state.selected_subcat = None
if "selected_fund" not in st.session_state:
    st.session_state.selected_fund = None
if "search_text" not in st.session_state:
    st.session_state.search_text = ""


def go_home():
    st.session_state.view = "home"
    st.session_state.search_text = ""
    st.rerun()


def open_category(raw_subcat: str):
    st.session_state.view = "category"
    st.session_state.selected_subcat = raw_subcat
    st.rerun()


def open_fund(scheme_name: str):
    st.session_state.view = "fund"
    st.session_state.selected_fund = scheme_name
    st.rerun()


# A fund-name link clicked inside a top-funds card list lands back here
# as "?fund=<name>" (see finance_bot._fund_link) -- route it into the
# fund view exactly like clicking a search suggestion would.
if st.query_params.get("fund"):
    fund_from_link = st.query_params["fund"]
    st.query_params.clear()
    open_fund(fund_from_link)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    '<div class="ff-brand"><div class="ff-mark">FF</div>'
    '<div class="ff-title">FundFinder</div></div>'
    '<div class="ff-sub">Search any mutual fund or category to see its returns, risk profile, and AI verdict.</div>',
    unsafe_allow_html=True,
)

status_bits = []
status_bits.append(
    '<span class="ff-pill on"><span class="dot"></span>{:,} funds</span>'.format(bot.fund_count())
)
if fund_summarizer:
    status_bits.append('<span class="ff-pill on"><span class="dot"></span>AI Verdict enabled</span>')
else:
    status_bits.append('<span class="ff-pill off"><span class="dot"></span>AI Verdict needs GROQ_API_KEY</span>')
if llm_fallback:
    status_bits.append('<span class="ff-pill on"><span class="dot"></span>General Q&A enabled</span>')
st.markdown("".join(status_bits), unsafe_allow_html=True)

if st.session_state.view != "home":
    st.markdown('<div class="ff-back">', unsafe_allow_html=True)
    if st.button("← Back to search", key="back_link"):
        go_home()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Search box — always visible, drives fund/category navigation
# ---------------------------------------------------------------------
query = st.text_input(
    "Search",
    value=st.session_state.search_text,
    placeholder="Search a fund or category, e.g. HDFC Flexi Cap Fund, Large Cap Fund, ELSS",
    label_visibility="collapsed",
    key="ff_search_input",
)
st.session_state.search_text = query

if query.strip():
    suggestions = bot.search_suggestions(query, limit=8)
    if suggestions:
        st.markdown('<div class="ff-suggestions"><div class="ff-suggestions-box">', unsafe_allow_html=True)
        for s in suggestions:
            icon = "📁" if s["type"] == "category" else "📈"
            btn_label = f'{icon}  {s["label"]}' + ("  ·  category" if s["type"] == "category" else "")
            if st.button(btn_label, key=f'sugg_{s["type"]}_{s["value"]}', use_container_width=True):
                if s["type"] == "fund":
                    open_fund(s["value"])
                else:
                    open_category(s["value"])
        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.caption("No matching fund or category found.")
        if llm_fallback:
            with st.spinner("Asking the AI assistant..."):
                answer = llm_fallback(query)
            st.markdown(f'<div class="ff-suggestions-box" style="padding:1rem;">{answer}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Home view — category directory (kept exactly as-is, just shown as a
# browsable directory instead of a sidebar/chat flow)
# ---------------------------------------------------------------------
if st.session_state.view == "home" and not query.strip():
    st.markdown('<div class="ff-section-label">Browse by category</div>', unsafe_allow_html=True)
    for atype in bot.asset_types:
        subs = bot.asset_type_to_subcats.get(atype, [])
        with st.expander(f"{atype}  ·  {len(subs)} categories"):
            cols = st.columns(2)
            for i, sc in enumerate(subs):
                with cols[i % 2]:
                    if st.button(subcat_browse_label(sc), key=f"cat_{sc}", use_container_width=True):
                        open_category(sc)

    st.markdown('<div class="ff-section-label">Disclaimer</div>', unsafe_allow_html=True)
    st.caption(
        "Fund rankings, scores, and AI verdicts on this site are generated from our own "
        "research using approximately the last 3 years of historical NAV data and other "
        "publicly available information. They are for informational and educational "
        "purposes only and do not constitute financial, investment, tax, or legal advice. "
        "Past performance does not guarantee future returns — please do your own research "
        "and consult a qualified financial advisor before investing. We are not "
        "responsible for any financial losses or investment decisions made based on this "
        "analysis."
    )

# ---------------------------------------------------------------------
# Category results view
# ---------------------------------------------------------------------
elif st.session_state.view == "category":
    subcat = st.session_state.selected_subcat
    st.markdown(bot.format_top_funds(subcat, n=10), unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Fund profile view — AI Verdict is always requested/shown here
# ---------------------------------------------------------------------
elif st.session_state.view == "fund":
    fund_name = st.session_state.selected_fund
    match = bot.df[bot.df["Scheme Name"] == fund_name]
    if match.empty:
        st.error(f"Couldn't find '{fund_name}' in the dataset.")
    else:
        spinner_text = "Fetching AI verdict..." if fund_summarizer else "Loading fund profile..."
        with st.spinner(spinner_text):
            st.markdown(
                bot.format_fund_profile(match.iloc[0], fund_summarizer=fund_summarizer),
                unsafe_allow_html=True,
            )
