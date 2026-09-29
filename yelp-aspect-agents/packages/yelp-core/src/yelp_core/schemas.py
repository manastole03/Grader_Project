"""Data contracts passed between packages and agents.

Agents never emit numbers. They emit *mentions*: a verbatim quote plus a categorical
polarity and a 1-3 intensity. Every numeric score is computed afterwards in Python
(see the yelp-scoring package), so scores are deterministic and auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The three dimensions every review is projected onto.
ASPECTS: tuple[str, ...] = ("food", "service", "ambience")

POLARITIES = ("positive", "negative", "neutral")
POLARITY_SIGN = {"positive": 1, "negative": -1, "neutral": 0}
ASPECT_LABELS = ("positive", "negative", "neutral", "not_mentioned")
SENTIMENT_LABELS = ("negative", "neutral", "positive")  # overall / ground-truth classes


def stars_to_sentiment(stars: float) -> str:
    """Ground-truth label derived from Yelp stars (1-2 neg, 3 neutral, 4-5 pos)."""
    if stars <= 2:
        return "negative"
    if stars >= 4:
        return "positive"
    return "neutral"


@dataclass
class Review:
    review_id: str
    business_id: str
    stars: float
    text: str
    business_name: str = ""
    city: str = ""
    categories: str = ""
    date: str = ""

    @property
    def label(self) -> str:
        return stars_to_sentiment(self.stars)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Review":
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


@dataclass
class Mention:
    quote: str
    polarity: str            # positive | negative | neutral
    intensity: int           # 1 mild, 2 clear, 3 strong
    grounded: bool = False   # every fragment of the quote was found in the review text
    spans: list[list[int]] = field(default_factory=list)  # [start, end) offsets of each fragment
    note: str = ""           # why a mention was dropped, if it was


@dataclass
class AspectResult:
    aspect: str
    mentions: list[Mention] = field(default_factory=list)  # grounded mentions only
    dropped: list[Mention] = field(default_factory=list)   # ungrounded, or reassigned by the arbiter
    score: float = 0.0          # computed by yelp_scoring.aspect_score, in [-1, 1]
    sentiment: str = "not_mentioned"
    error: str | None = None

    @property
    def mentioned(self) -> bool:
        return bool(self.mentions)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AspectResult":
        d = dict(d)
        d["mentions"] = [Mention(**m) for m in d.get("mentions", [])]
        d["dropped"] = [Mention(**m) for m in d.get("dropped", [])]
        return cls(**d)


@dataclass
class OverallResult:
    polarity: str = "neutral"   # lead agent's categorical judgement
    intensity: int = 1
    score: float = 0.0          # computed: sign(polarity) * intensity / 3
    sentiment: str = "neutral"
    stars_rule: float = 3.0     # computed: 3 + 2 * score (uncalibrated)
    other_factors: list[str] = field(default_factory=list)
    rationale: str = ""
    error: str | None = None


@dataclass
class ReviewAnalysis:
    review: Review
    aspects: dict[str, AspectResult]
    overall: OverallResult
    provider: str = ""
    model: str = ""
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def vector(self) -> list[float]:
        """The review's point in 3-D aspect space: [food, service, ambience]."""
        return [self.aspects[a].score for a in ASPECTS]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["vector"] = self.vector
        d["label"] = self.review.label
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReviewAnalysis":
        return cls(
            review=Review.from_dict(d["review"]),
            aspects={k: AspectResult.from_dict(v) for k, v in d["aspects"].items()},
            overall=OverallResult(**d["overall"]),
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            latency_s=d.get("latency_s", 0.0),
            meta=d.get("meta", {}),
        )
