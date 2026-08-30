"""
llm_fallback.py
----------------
Optional LLM fallback for free-form finance questions that aren't a
direct "top N funds" or "tell me about <fund>" request (e.g. "what does
Sharpe ratio mean?"), plus the per-fund AI Verdict summarizer used on
every fund's profile page.

Uses the Groq API (fast inference, generous free tier) when a
GROQ_API_KEY is configured -- either as an environment variable or as
a Streamlit secret. If no key is configured, get_llm_fallback() and
get_fund_risk_summarizer() both return None; the fund profile page
still always shows an "AI Verdict" section, just with a note that it's
unavailable until a key is configured (see finance_bot.py).
"""

from __future__ import annotations

import os
import re

SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in FundFinder, a mutual fund "
    "analytics website. Answer general finance / mutual fund questions "
    "concisely and in plain language. You do NOT have access to the "
    "live fund dataset in this fallback -- for fund-specific rankings "
    "or metrics, tell the user to search for something like 'large cap "
    "funds' or a fund name instead, which are handled directly by the "
    "site. Never give personalized investment advice; include a brief "
    "reminder to consult a qualified financial advisor before making "
    "investment decisions."
)

MODEL = "openai/gpt-oss-20b"


def _get_api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def get_llm_fallback():
    """Returns a callable `fallback(query: str) -> str`, or None if no
    GROQ_API_KEY is configured (or the `groq` package isn't installed)."""
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=api_key)

    def fallback(query: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.4,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            return f"Sorry, I couldn't reach the general Q&A model right now ({e})."

    return fallback


# ----------------------------------------------------------------------
# Fund risk summary -- a separate, narrowly-scoped LLM call (different
# system prompt from the general Q&A fallback above) that takes a single
# fund's own return/risk metrics and produces a short plain-language
# summary plus an explicit lean ("worth considering" / "exercise
# caution" / "probably not"), so the reader gets an at-a-glance takeaway
# on the fund's profile page instead of having to read every metric
# themselves. Reuses the same GROQ_API_KEY config as get_llm_fallback().
# Called for EVERY fund profile shown on the site (see
# finance_bot.format_fund_profile) -- not opt-in per fund.
# ----------------------------------------------------------------------
FUND_SUMMARY_SYSTEM_PROMPT = (
    "You are a cautious mutual fund risk analyst embedded in FundFinder. "
    "You will be given one fund's own return and risk metrics (returns "
    "across horizons, volatility, max drawdown, Sharpe/Sortino/Calmar "
    "ratios, Value at Risk, downside deviation, and its percentile rank "
    "vs. category peers). Using ONLY the numbers given:\n"
    "1. Your FIRST line must be exactly 'VERDICT: X' where X is one of "
    "INVEST, AVOID, or NEUTRAL -- your verdict on whether this fund is "
    "worth investing in, judged purely from the numbers. Use NEUTRAL "
    "only when the metrics are genuinely mixed with no clear lean either "
    "way; prefer INVEST or AVOID whenever the numbers support one.\n"
    "2. Then, after a blank line, write 3-5 short sentences in plain "
    "language explaining WHY -- reference the specific return "
    "consistency, volatility, drawdown severity, and peer standing that "
    "drove your verdict. Reference only the few numbers that matter "
    "most; do not repeat every metric back.\n"
    "3. End with exactly one more line: a short reminder that this is an "
    "automated read of historical numbers only, is NOT personalized "
    "financial advice, and the reader should consult a qualified "
    "financial advisor and consider their own goals/horizon before "
    "investing.\n"
    "Do not include any other headings, labels, or text before the "
    "VERDICT line."
)

# Parses the "VERDICT: X" line the prompt above requires back out of the
# model's response, so the UI can render it as its own prominent badge
# instead of leaving the invest/avoid call buried inside a paragraph.
# Falls back to (None, original_text) untouched if the model ever
# deviates from the requested format, so a malformed response still
# displays (just without the badge) instead of losing the summary.
_VERDICT_RE = re.compile(r"^\s*VERDICT:\s*(INVEST|AVOID|NEUTRAL)\s*$", re.IGNORECASE | re.MULTILINE)

_VERDICT_BADGES = {
    "INVEST": "🟢 **INVEST**",
    "AVOID": "🔴 **AVOID**",
    "NEUTRAL": "🟡 **NEUTRAL**",
}


def parse_fund_verdict(summary_text: str) -> tuple[str | None, str]:
    """Splits a fund-summary response into (verdict_badge_markdown, rest).
    verdict_badge_markdown is None if no VERDICT line was found (e.g. the
    model deviated from the format) -- callers should just render `rest`
    (equal to the original text in that case) with no badge."""
    if not summary_text:
        return None, summary_text
    m = _VERDICT_RE.search(summary_text)
    if not m:
        return None, summary_text
    verdict = m.group(1).upper()
    rest = (summary_text[:m.start()] + summary_text[m.end():]).strip()
    return _VERDICT_BADGES[verdict], rest


def get_fund_risk_summarizer():
    """Returns a callable `summarize(metrics_text: str) -> str` that
    turns a plain-text dump of one fund's metrics into a short AI
    summary + investment lean, or None if no GROQ_API_KEY is configured
    (or the `groq` package isn't installed) -- same fallback-to-None
    behavior as get_llm_fallback()."""
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=api_key)

    def summarize(metrics_text: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": FUND_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": metrics_text},
                ],
                temperature=0.3,
                max_tokens=350,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            return f"_AI summary unavailable right now ({e})._"

    return summarize
