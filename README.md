## Features

- **Search-box typeahead** — one search box resolves to either a
  fund's full profile or a category's top-10 list, ranked by fuzzy
  match against fund names and Sub Categories.
- **Category directory** — browse funds by Asset Type → Sub Category
  (Equity, Debt, Hybrid, Index/ETF, Solution Oriented, etc.) without
  needing to know the exact fund or category name.
- **Fund profile pages** — Scheme details, Composite Score / Peer
  Rank, and a tap-to-expand breakdown of returns and risk metrics
  (CAGR, Volatility, Max Drawdown, Sharpe, Sortino, Calmar, VaR,
  Downside Deviation, and 3Y peer-percentile ranks) across every
  horizon from 1D to Since Inception.
- **AI Verdict on every fund** — a short, plain-language INVEST /
  AVOID / NEUTRAL read generated from the fund's own 3-year metrics,
  shown on every fund's profile page (requires a `GROQ_API_KEY`; see
  below).
- **Optional general Q&A** — free-form finance questions ("what is a
  Sharpe ratio?") get answered by the same LLM when a search doesn't
  match a specific fund or category.
- **Local, offline query understanding** — AMC/fund-house detection,
  category matching, and typo tolerance all run on-device via spaCy +
  rapidfuzz. No network call is needed for search/browse; only the AI
  Verdict and general Q&A features call out to Groq.

## Project structure

```
FundFinder/
├── app.py             # Streamlit UI: search box, category directory, fund/category pages
├── finance_bot.py      # Data loading, matching, ranking, and formatting logic
├── nlp_utils.py         # Offline NLP helpers (AMC matching, fuzzy scoring)
├── llm_fallback.py      # Optional Groq-backed AI Verdict + general Q&A
├── MF_Risk_Metrics.xlsx # Your fund dataset (not included — see below)
└── README.md
```

## Requirements

- Python 3.10+
- The dataset file `MF_Risk_Metrics.xlsx` (sheet name: `Risk Metrics`)
  placed in the same folder as `finance_bot.py`. This is the fixed,
  single source of data — there is no upload path, and the app will
  not start without it.

Install dependencies:

```bash
pip install streamlit pandas rapidfuzz spacy openpyxl groq
```

`spacy` only needs its base package — no model download is required
(`nlp_utils.py` uses a blank tokenizer-only pipeline).

## Dataset format

`MF_Risk_Metrics.xlsx`, sheet `Risk Metrics`, must include at least:

- `Scheme Name`, `Scheme Code`, `AMC (Fund House)`, `Sub Category`,
  `Asset Class`, `ELSS`, `Latest NAV`, `Latest NAV Date`
- Per-horizon metric columns named `<Horizon>_<Metric>`, e.g.
  `3Y_CAGR`, `1Y_AbsoluteReturn`, `5Y_MaxDrawdown`, for horizons
  `1D, 1W, 1M, 6M, 1Y, 3Y, 5Y, 10Y, SI` and metrics
  `AbsoluteReturn, CAGR, Volatility, MaxDrawdown, Sharpe, Sortino,
  DownsideDev, VaR95, Calmar, RollMean, RollMin, RollMax`
- `Composite_Score`, `Peer_Rank`
- Optional 3Y peer-percentile columns, e.g. `3Y_CAGR_PeerPctile`

Missing columns degrade gracefully (a horizon or metric with no data
is simply not shown) rather than causing an error.

## Enabling the AI Verdict

The AI Verdict and general Q&A features use the [Groq API](https://groq.com/).
Without a key, every fund profile still shows an "AI Verdict" section
— it just displays a note that the feature isn't configured, instead
of a generated verdict.

Set your key as an environment variable:

```bash
export GROQ_API_KEY="your-key-here"
```

or add it to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-key-here"
```

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Notes on the data pipeline

- **Mojibake repair**: a handful of cells in the source spreadsheet
  can have UTF‑8 text saved through Latin‑1/cp1252 (curly apostrophes,
  en‑dashes) — this is auto-corrected at load time.
- **Duplicate category merging**: Sub Categories that are really the
  same SEBI category under two spellings (e.g. `ELSS` vs. `ELSS Tax
  Saver Fund`) are merged into one before building the category
  directory.
- **Plan/option de-duplication**: multiple rows for the same
  underlying fund under different payout options (Growth / IDCW /
  Dividend / …) are collapsed to a single row, preferring Growth,
  before ranking "top funds."
- **One fund per AMC** in a top-10 list, unless the search was already
  filtered to a specific AMC (e.g. "HDFC small cap funds").

## Disclaimer

Fund rankings, scores, and AI verdicts shown by this app are generated
from the maintainer's own research using approximately the last 3
years of historical NAV data and other publicly available information.
They are for informational and educational purposes only and do **not**
constitute financial, investment, tax, or legal advice. Past
performance does not guarantee future returns. Please do your own
research and consult a qualified financial advisor before making any
investment decisions. The maintainers are not responsible for any
financial losses or investment decisions made based on this analysis.
