"""The numbers must be right: grounding, formulas, calibration and metrics."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from yelp_core.schemas import AspectResult, Mention, OverallResult, Review, ReviewAnalysis
from yelp_eval.metrics import classification_block, evaluate, wilson_interval
from yelp_scoring import (
    StarModel, aspect_score, build_aspect_result, build_overall, feature_row, ground_quote, stars_to_label,
)

TEXT = "The pasta was “amazing”.   Our waiter was rude and slow. It's loud inside, but cozy."


def span_text(spans):
    return [TEXT[a:b] for a, b in spans]


# ---------------------------------------------------------------- grounding
def test_exact_and_normalised_matches_map_back_to_original_offsets():
    assert span_text(ground_quote(TEXT, "Our waiter was rude and slow.")) == ["Our waiter was rude and slow."]
    # straight vs curly quotes, case, collapsed whitespace
    assert span_text(ground_quote(TEXT, 'the pasta was "AMAZING". our')) == ["The pasta was “amazing”.   Our"]


def test_ellipsis_fragments_are_each_grounded():
    assert span_text(ground_quote(TEXT, "The pasta was ... rude and slow")) == ["The pasta was", "rude and slow"]
    assert ground_quote(TEXT, "The pasta was ... fantastic dessert menu") is None


def test_near_exact_tolerated_but_paraphrase_rejected():
    assert ground_quote(TEXT, "Our waiter was rude and slow!") is not None      # punctuation drift
    assert ground_quote(TEXT, "The staff were impolite and sluggish") is None     # paraphrase


def test_single_word_is_not_evidence():
    assert ground_quote(TEXT, "loud") is None
    assert ground_quote(TEXT, "loud inside") is not None


# ---------------------------------------------------------------- aspect scoring
def test_aspect_score_formula_and_labels():
    m = lambda p, i: Mention("q", p, i)
    assert aspect_score([]) == 0.0
    assert aspect_score([m("positive", 2), m("negative", 1)]) == pytest.approx((2 - 1) / 6, abs=1e-4)
    assert aspect_score([m("positive", 3)]) == 1.0
    assert aspect_score([m("negative", 3), m("negative", 3)]) == -1.0
    assert aspect_score([m("neutral", 3)]) == 0.0


def test_build_aspect_result_grounds_dedupes_and_labels():
    raw = [
        {"quote": "Our waiter was rude and slow.", "polarity": "negative", "intensity": 2},
        {"quote": "waiter was rude", "polarity": "negative", "intensity": 3},        # overlaps -> counted once
        {"quote": "The service was lightning fast", "polarity": "positive", "intensity": 3},  # not in text
        {"quote": "It's loud inside", "polarity": "NEGATIVE", "intensity": "7"},     # normalised + clamped
        {"quote": "", "polarity": "positive", "intensity": 1},                       # ignored
        {"quote": "cozy", "polarity": "sideways", "intensity": 1},                   # bad polarity ignored
    ]
    r = build_aspect_result("service", TEXT, raw)
    assert [m.quote for m in r.mentions] == ["Our waiter was rude and slow.", "It's loud inside"]
    assert [m.intensity for m in r.mentions] == [2, 3]
    assert [m.note for m in r.dropped] == ["quote not found in review"]
    assert r.score == pytest.approx(-(2 + 3) / 6, abs=1e-4) and r.sentiment == "negative"

    balanced = build_aspect_result("x", TEXT, [
        {"quote": "The pasta was", "polarity": "positive", "intensity": 2},
        {"quote": "rude and slow", "polarity": "negative", "intensity": 2}])
    assert balanced.score == 0 and balanced.sentiment == "neutral"
    assert build_aspect_result("x", TEXT, []).sentiment == "not_mentioned"


def test_overall_and_star_mapping():
    o = build_overall({"polarity": "negative", "intensity": 3})
    assert (o.score, o.stars_rule, o.sentiment) == (-1.0, 1.0, "negative")
    n = build_overall({"polarity": "neutral", "intensity": 3})
    assert (n.intensity, n.score, n.stars_rule) == (1, 0.0, 3.0)
    assert build_overall({"polarity": "great"}) is None
    assert [stars_to_label(s) for s in (1, 2.49, 2.5, 3.5, 3.51, 5)] == \
        ["negative", "negative", "neutral", "neutral", "positive", "positive"]


# ---------------------------------------------------------------- calibration
def _analysis(i: int, stars: float, food: float, mentioned=True, overall=0.0) -> ReviewAnalysis:
    asp = {a: AspectResult(a) for a in ("food", "service", "ambience")}
    if mentioned:
        asp["food"] = AspectResult("food", mentions=[Mention("q", "positive", 1)], score=food, sentiment="positive")
    return ReviewAnalysis(Review(f"r{i}", f"b{i % 7}", stars, "t"), asp, OverallResult(score=overall))


def test_cross_validation_is_out_of_fold_and_matches_manual_kfold():
    rng = np.random.default_rng(1)
    rs = [_analysis(i, float(s), (s - 3) / 2 + rng.normal(0, 0.1)) for i, s in enumerate(rng.integers(1, 6, 60))]
    oof = StarModel().cross_validated(rs, folds=5, seed=0)
    X = np.array([feature_row(r) for r in rs])
    y = np.array([r.review.stars for r in rs])
    manual = np.empty(len(rs))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        manual[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    assert np.allclose(oof, np.clip(manual, 1, 5))
    assert np.mean(np.abs(oof - y)) < 0.5            # recovers the planted relationship
    assert StarModel().fit(rs).coefficients()["food"] > 1.0


def test_star_model_requires_enough_data():
    assert not StarModel.usable([_analysis(i, 4, 0.5) for i in range(20)])  # one star level only
    assert not StarModel.usable([_analysis(i, 1 + i % 5, 0.5) for i in range(9)])


# ---------------------------------------------------------------- metrics
def test_wilson_interval_known_value():
    lo, hi = wilson_interval(28, 30)
    assert lo == pytest.approx(0.7868, abs=1e-3) and hi == pytest.approx(0.9815, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_classification_block_uses_true_then_pred():
    y_true = ["negative"] * 4 + ["positive"] * 4
    y_pred = ["negative"] * 4 + ["positive", "positive", "negative", "negative"]
    b = classification_block(y_true, y_pred)
    assert b["accuracy"] == 6 / 8
    assert b["report"]["positive"]["recall"] == 0.5          # 2 of 4 actual positives found
    assert b["report"]["negative"]["precision"] == 4 / 6     # 4 of 6 predicted negatives correct
    assert b["confusion"] == [[4, 0, 0], [0, 0, 0], [2, 0, 2]]


def test_evaluate_end_to_end_numbers():
    rs = [_analysis(i, s, (s - 3) / 2, overall=(s - 3) / 2) for i, s in enumerate([1, 2, 3, 4, 5] * 4)]
    for r in rs:
        r.overall.sentiment = r.review.label
        r.overall.stars_rule = 3 + 2 * r.overall.score
    m = evaluate(rs)
    assert m["lead_agent"]["accuracy"] == 1.0
    assert m["stars_rule"]["mae"] == pytest.approx(0.0)
    assert m["stars_rule"]["spearman"] == pytest.approx(1.0)
    assert m["calibrated"]["stars"]["mae"] < 0.5
    assert m["aspects"]["food"]["mention_rate"] == 1.0
    assert m["aspects"]["service"]["mention_rate"] == 0.0
