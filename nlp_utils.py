"""
nlp_utils.py
------------
Local, fully offline NLP helpers for FundFinder's query understanding.
No network calls anywhere in this module -- everything runs on-device.

Two problems this solves that plain regex + difflib couldn't:

  1. AMC / fund-house extraction: "HDFC Small cap funds" was being
     fuzzy-matched as a whole string against Sub Category labels. It
     happened to *find* "Small Cap Fund" (via the substring boost in
     best_sub_category_match), but "HDFC" was silently dropped instead
     of being used to filter results -- so the bot showed top-10 across
     ALL AMCs, not just HDFC's fund. AMCMatcher below recognizes an AMC
     name/brand-word anywhere in the query (built from the dataset's own
     "AMC (Fund House)" column -- never a hardcoded list) and strips it
     out, returning both pieces separately.

  2. Fuzzy matching quality: difflib.SequenceMatcher.ratio() operates on
     raw character sequences, so word reordering or a few extra/missing
     words (exactly what AMC-prefixed queries introduce) drag the score
     down even when the *meaning* is a clean match. rapidfuzz's
     token_sort_ratio / token_set_ratio compare bags of words instead,
     so "hdfc small cap fund" vs "small cap fund" scores high, and it's
     a compiled C library so it's also substantially faster than difflib
     at matching against a large candidate list.

Both spaCy and rapidfuzz are required as declared dependencies (see
requirements.txt). If the spaCy model isn't installed, AMCMatcher
degrades gracefully to plain first-word matching rather than crashing.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz, process

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    _SPACY_IMPORT_OK = True
except ImportError:  # pragma: no cover - exercised only if spaCy isn't installed
    _SPACY_IMPORT_OK = False


# ----------------------------------------------------------------------
# Spelled-out numbers ("top five funds") -- digit counts ("top 5 funds")
# are already handled by the existing regex in finance_bot.py; this is
# purely a fallback for when the count is written as a word.
# ----------------------------------------------------------------------
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50,
}


def extract_number_word(text: str) -> int | None:
    """Find a spelled-out count word in `text`. Checks longer words
    first so e.g. 'fifteen' isn't mistakenly short-circuited."""
    tl = text.lower()
    for word in sorted(_WORD_NUMBERS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", tl):
            return _WORD_NUMBERS[word]
    return None


@lru_cache(maxsize=1)
def _get_nlp():
    """A blank spaCy English pipeline -- tokenizer only, no trained
    components and no downloaded model. We build our own AMC vocabulary
    via PhraseMatcher below (the stock English NER model was never
    trained to recognize Indian AMC names anyway, so a downloaded model
    would buy us nothing here), so all we actually need from spaCy is
    its tokenizer. spacy.blank("en") ships with the base `spacy` pip
    package -- no `python -m spacy download ...` step, no separate model
    wheel to install, which also sidesteps deploy environments (e.g.
    Streamlit Community Cloud) whose installers can reject a direct-URL
    model dependency in requirements.txt.
    Returns None only if spaCy itself isn't installed, so callers can
    fall back to a pure string-matching path instead of crashing."""
    if not _SPACY_IMPORT_OK:
        return None
    return spacy.blank("en")


class AMCMatcher:
    """Recognizes an AMC / fund-house name or brand word inside a
    free-text query, built entirely from the dataset's own
    "AMC (Fund House)" column values -- so it stays correct automatically
    if the sheet's fund-house roster changes, with no hardcoded list to
    maintain.
    """

    def __init__(self, amc_names: list[str]):
        # Longest names first so PhraseMatcher's span-length tie-break
        # (see extract()) naturally prefers "HDFC Mutual Fund" over a
        # coincidental shorter overlap, if both were ever added.
        self.amc_names = sorted({a.strip() for a in amc_names if a and a.strip()},
                                 key=len, reverse=True)
        self._brand_words = {
            a.split(" ", 1)[0].lower()
            for a in self.amc_names
            if len(a.split(" ", 1)[0]) >= 3  # skip short tokens, too noisy
        }

        self._nlp = _get_nlp()
        self._matcher = None
        if self._nlp is not None and self.amc_names:
            self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
            patterns = [self._nlp.make_doc(name) for name in self.amc_names]
            patterns += [self._nlp.make_doc(w) for w in sorted(self._brand_words)]
            self._matcher.add("AMC", patterns)

    def extract(self, query: str) -> tuple[str | None, str]:
        """Returns (matched_amc_text, rest_of_query_with_match_removed).
        Returns (None, original_query) if no AMC is recognized."""
        q = (query or "").strip()
        if not q:
            return None, q

        if self._matcher is not None:
            doc = self._nlp(q)
            matches = self._matcher(doc)
            if matches:
                # Prefer the longest span (full AMC name beats a bare brand word).
                _match_id, start, end = max(matches, key=lambda m: m[2] - m[1])
                span = doc[start:end]
                rest = (q[:span.start_char] + " " + q[span.end_char:]).strip()
                rest = re.sub(r"\s+", " ", rest)
                return span.text, rest

        # Fallback with no spaCy model available: check the first token
        # against known AMC brand words only.
        tokens = q.split()
        if tokens and tokens[0].lower() in self._brand_words:
            return tokens[0], " ".join(tokens[1:]).strip()
        return None, q


def fuzzy_ratio(a: str, b: str) -> float:
    """0-1 similarity using rapidfuzz's token_sort_ratio -- word-order
    and extra-word tolerant, unlike difflib.SequenceMatcher.ratio()."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def fuzzy_partial_ratio(a: str, b: str) -> float:
    """0-1 similarity using rapidfuzz's token_set_ratio -- good for
    'is one string's word-set essentially contained in the other's',
    e.g. a short query against a longer scheme name."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def best_fuzzy_match(
    query: str, candidates: list[str], scorer=fuzz.token_sort_ratio
) -> tuple[str | None, float]:
    """rapidfuzz.process.extractOne wrapper. Returns (best_candidate, score
    0-1), or (None, 0.0) if candidates is empty or nothing scores above 0."""
    if not query or not candidates:
        return None, 0.0
    result = process.extractOne(query, candidates, scorer=scorer)
    if result is None:
        return None, 0.0
    match, score, _idx = result
    return match, score / 100.0


def ranked_fuzzy_matches(
    query: str, candidates: list[str], limit: int = 6,
    scorer=fuzz.token_set_ratio, score_cutoff: float = 0.0,
) -> list[tuple[str, float]]:
    """rapidfuzz.process.extract wrapper. Returns up to `limit`
    (candidate, score_0_to_1) pairs, best first."""
    if not query or not candidates:
        return []
    results = process.extract(
        query, candidates, scorer=scorer, limit=limit,
        score_cutoff=score_cutoff * 100,
    )
    return [(match, score / 100.0) for match, score, _idx in results]
