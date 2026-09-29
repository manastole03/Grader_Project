"""All numeric scores in the project are computed here, in plain Python.

LLM agents only supply categorical judgements (quotes, polarity, intensity 1-3).
This package turns them into numbers:

* grounding    - verifies each quoted mention actually occurs in the review
* scoring      - aspect score, overall score, rule-based stars
* star_model   - cross-validated ridge regression from the 3-D vector to stars
"""
from yelp_scoring.grounding import ground_quote
from yelp_scoring.scoring import (
    aspect_label, aspect_score, build_aspect_result, build_overall, mention_value, overall_score,
    rescore, stars_rule, stars_to_label,
)
from yelp_scoring.star_model import FEATURES, StarModel, feature_row

__all__ = [
    "FEATURES", "StarModel", "aspect_label", "aspect_score", "build_aspect_result", "build_overall",
    "feature_row", "ground_quote", "mention_value", "overall_score", "rescore", "stars_rule", "stars_to_label",
]
