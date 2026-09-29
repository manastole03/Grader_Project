"""Deterministic scoring formulas.

    mention value     v = sign(polarity) * intensity / 3        in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}
    aspect score      mean of v over grounded, de-duplicated mentions (0 when none)
    aspect label      not_mentioned if no grounded mention, else sign of the score
                      (neutral when mentions are all neutral or exactly balance out)
    overall score     sign(lead polarity) * lead intensity / 3
    rule-based stars  3 + 2 * overall score                      in [1, 5]
"""
from __future__ import annotations

from typing import Any

from yelp_core.schemas import POLARITIES, POLARITY_SIGN, AspectResult, Mention, OverallResult

from yelp_scoring.grounding import ground_quote

MAX_INTENSITY = 3


def _intensity(x: Any) -> int:
    try:
        return max(1, min(MAX_INTENSITY, int(round(float(x)))))
    except (TypeError, ValueError):
        return 1


def _polarity(x: Any) -> str | None:
    p = str(x or "").strip().lower()
    return p if p in POLARITIES else None


def mention_value(m: Mention) -> float:
    return POLARITY_SIGN[m.polarity] * m.intensity / MAX_INTENSITY


def aspect_score(mentions: list[Mention]) -> float:
    if not mentions:
        return 0.0
    return round(sum(mention_value(m) for m in mentions) / len(mentions), 4)


def aspect_label(mentions: list[Mention], score: float) -> str:
    if not mentions:
        return "not_mentioned"
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def _overlaps(a: list[list[int]], b: list[list[int]]) -> bool:
    return any(x0 < y1 and y0 < x1 for x0, x1 in a for y0, y1 in b)


def build_aspect_result(aspect: str, review_text: str, raw_mentions: list[dict[str, Any]]) -> AspectResult:
    """Validate, ground and de-duplicate raw LLM mentions, then score them."""
    kept: list[Mention] = []
    dropped: list[Mention] = []
    for raw in raw_mentions or []:
        if not isinstance(raw, dict):
            continue
        polarity = _polarity(raw.get("polarity"))
        quote = str(raw.get("quote", "")).strip()
        if polarity is None or not quote:
            continue
        m = Mention(quote=quote[:400], polarity=polarity, intensity=_intensity(raw.get("intensity")))
        spans = ground_quote(review_text, quote)
        if spans is None:
            m.note = "quote not found in review"
            dropped.append(m)
            continue
        m.grounded, m.spans = True, spans
        if any(_overlaps(spans, k.spans) for k in kept):  # same evidence counted once
            continue
        kept.append(m)
    score = aspect_score(kept)
    return AspectResult(aspect=aspect, mentions=kept, dropped=dropped, score=score,
                        sentiment=aspect_label(kept, score))


def rescore(result: AspectResult) -> AspectResult:
    """Recompute score and label after mentions were removed (e.g. by the arbiter)."""
    result.score = aspect_score(result.mentions)
    result.sentiment = aspect_label(result.mentions, result.score)
    return result


def overall_score(polarity: str, intensity: int) -> float:
    return round(POLARITY_SIGN[polarity] * intensity / MAX_INTENSITY, 4)


def stars_rule(score: float) -> float:
    return round(3 + 2 * max(-1.0, min(1.0, score)), 3)


def stars_to_label(stars: float) -> str:
    """Map a continuous star prediction onto the ground-truth classes (midpoints 2.5 / 3.5)."""
    if stars < 2.5:
        return "negative"
    if stars > 3.5:
        return "positive"
    return "neutral"


def build_overall(raw: dict[str, Any], rationale_limit: int = 500) -> OverallResult | None:
    polarity = _polarity(raw.get("polarity"))
    if polarity is None:
        return None
    intensity = 1 if polarity == "neutral" else _intensity(raw.get("intensity"))
    score = overall_score(polarity, intensity)
    factors = raw.get("other_factors") or []
    return OverallResult(
        polarity=polarity, intensity=intensity, score=score, sentiment=polarity,
        stars_rule=stars_rule(score),
        other_factors=[str(f)[:80] for f in factors if f][:6] if isinstance(factors, list) else [],
        rationale=str(raw.get("rationale", ""))[:rationale_limit],
    )
