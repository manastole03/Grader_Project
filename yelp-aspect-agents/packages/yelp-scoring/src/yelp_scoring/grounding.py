"""Check that a model-quoted snippet really occurs in the review.

Models paraphrase, change quote marks, re-case words and elide with "...". A quote is
split on ellipses (the quote needs at least two words in total) and every fragment must be found, first by exact match on a
normalised copy of the text, then by a near-exact match (longest common block
covering >= 85% of the fragment). Offsets are mapped back to the original text.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.+|…)\s*")
MIN_FRAGMENT_CHARS = 4
MIN_QUOTE_WORDS = 2  # a lone word ("amazing") can match anywhere, so it is not evidence
FUZZY_COVERAGE = 0.85


def _normalise(text: str) -> tuple[str, list[int]]:
    """Lower-case, unify quotes/dashes, collapse whitespace; keep a map to original offsets."""
    out: list[str] = []
    index: list[int] = []
    prev_space = False
    for i, ch in enumerate(text.translate(_TRANS).lower()):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def _fragments(quote: str) -> list[str]:
    parts = _ELLIPSIS.split(quote.strip())
    cleaned = [p.strip().strip("\"'").strip() for p in parts]
    return [p for p in cleaned if len(p) >= MIN_FRAGMENT_CHARS]


def _find(norm_text: str, frag: str) -> tuple[int, int] | None:
    i = norm_text.find(frag)
    if i >= 0:
        return i, i + len(frag)
    # tolerate a trailing period / comma the model added or dropped
    trimmed = frag.rstrip(".,!?;:")
    if len(trimmed) >= MIN_FRAGMENT_CHARS and (i := norm_text.find(trimmed)) >= 0:
        return i, i + len(trimmed)
    m = SequenceMatcher(None, norm_text, frag, autojunk=False).find_longest_match(0, len(norm_text), 0, len(frag))
    if m.size >= FUZZY_COVERAGE * len(frag):
        start = max(0, m.a - m.b)
        return start, min(len(norm_text), start + len(frag))
    return None


def ground_quote(text: str, quote: str) -> list[list[int]] | None:
    """Return [start, end) spans in *text* for every fragment of *quote*, or None if any is missing."""
    frags = _fragments(quote)
    if not frags or sum(len(f.split()) for f in frags) < MIN_QUOTE_WORDS:
        return None
    norm_text, index = _normalise(text)
    spans: list[list[int]] = []
    for frag in frags:
        norm_frag, _ = _normalise(frag)
        hit = _find(norm_text, norm_frag)
        if hit is None:
            return None
        a, b = hit
        spans.append([index[a], index[b - 1] + 1])
    return spans
