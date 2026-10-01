"""Restaurant sampling and the within-restaurant comparison of reviews."""
import io
import json
import tarfile

import pytest
from scipy import stats

from yelp_core.schemas import AspectResult, Mention, OverallResult, Review, ReviewAnalysis
from yelp_data import RestaurantSampleConfig, sample_restaurants_from_tar
from yelp_eval.compare import compare_restaurants, format_comparison, format_restaurant, one_way
from yelp_scoring import aspect_label, aspect_score


# ---------------------------------------------------------------- one-way ANOVA / ICC(1)
def test_one_way_matches_scipy_and_hand_computed_icc():
    groups = [[1, 2, 3], [4, 5, 6]]
    out = one_way(groups)
    ref = stats.f_oneway(*groups)
    assert out["F"] == pytest.approx(ref.statistic) and out["p"] == pytest.approx(ref.pvalue)
    # MSB = 13.5, MSW = 1, n0 = 3 -> (13.5 - 1) / (13.5 + 2 * 1)
    assert out["icc1"] == pytest.approx(12.5 / 15.5)


def test_one_way_unequal_groups_and_degenerate_cases():
    groups = [[0.2, 0.4, 1.0], [-0.5, 0.1], [0.9, 1.0, 0.7, 0.8]]
    out = one_way(groups + [[0.3]])                  # singleton groups are ignored
    assert out["groups"] == 3 and out["n"] == 9
    assert out["F"] == pytest.approx(stats.f_oneway(*groups).statistic)
    assert one_way([[1, 2]])["icc1"] is None          # a single group
    assert one_way([[1, 1], [1, 1]])["F"] is None     # no variance at all


# ---------------------------------------------------------------- comparison
def _analysis(rid, bid, stars, food=(), service=(), date="2020-01-01"):
    aspects = {}
    for a, spec in (("food", food), ("service", service), ("ambience", ())):
        ms = [Mention(f"{a} quote {k} in {rid}", p, i, True, [[k, k + 1]]) for k, (p, i) in enumerate(spec)]
        score = aspect_score(ms)
        aspects[a] = AspectResult(a, mentions=ms, score=score, sentiment=aspect_label(ms, score))
    review = Review(rid, bid, stars, "text", f"Biz {bid}", "Tampa", date=date,
                    business_stars={"a": 4.5, "b": 2.0}.get(bid), business_review_count=99)
    return ReviewAnalysis(review, aspects, OverallResult(stars_rule=float(stars)))


RESULTS = [
    _analysis("a1", "a", 5, food=[("positive", 3)], service=[("negative", 2)], date="2019-05-01"),
    _analysis("a2", "a", 4, food=[("positive", 2)], date="2018-01-01"),
    _analysis("a3", "a", 5, food=[("positive", 2)]),
    _analysis("a4", "a", 1, food=[("negative", 3)]),
    _analysis("b1", "b", 4, food=[("positive", 1)]),
    _analysis("b2", "b", 2, food=[("negative", 1)]),
    _analysis("c1", "c", 3, food=[("positive", 1)]),   # only one review: not compared
]


def test_per_restaurant_agreement_consensus_and_quotes():
    c = compare_restaurants(RESULTS)
    by = {x["business_id"]: x for x in c["restaurants"]}
    assert set(by) == {"a", "b"}
    a = by["a"]
    assert a["review_ids"] == ["a2", "a1", "a3", "a4"]                  # oldest first
    assert a["stars"]["mean"] == pytest.approx(3.75) and a["stars"]["counts"] == {1: 1, 2: 0, 3: 0, 4: 1, 5: 2}
    assert a["yelp_stars"] == 4.5 and a["yelp_review_count"] == 99
    food = a["aspects"]["food"]
    assert food["mentioned"] == 4 and food["labels"] == {"positive": 3, "neutral": 0, "negative": 1, "not_mentioned": 0}
    assert food["agreement"] == 0.75 and food["consensus"] == "positive" and food["contested"]
    assert food["mean"] == pytest.approx((1 + 2 / 3 + 2 / 3 - 1) / 4, abs=1e-4)
    assert food["most_positive"]["review_id"] == "a1" and food["most_positive"]["value"] == 1.0
    assert food["most_negative"]["review_id"] == "a4" and food["most_negative"]["value"] == -1.0
    service = a["aspects"]["service"]
    assert service["mentioned"] == 1 and service["consensus"] == "too_few" and service["agreement"] is None
    assert service["most_positive"] is None and service["most_negative"]["review_id"] == "a1"
    assert a["aspects"]["ambience"]["mean"] is None and a["aspects"]["ambience"]["labels"]["not_mentioned"] == 4
    b_food = by["b"]["aspects"]["food"]
    assert b_food["agreement"] == 0.5 and b_food["consensus"] == "mixed"
    assert a["contested"] == ["food"] and by["b"]["contested"] == ["food"]


def test_summary_across_restaurants():
    s = compare_restaurants(RESULTS)["summary"]
    assert s["restaurants"] == 2 and s["reviews"] == 6 and s["restaurants_contested"] == 2
    food = s["aspects"]["food"]
    assert food["restaurants_compared"] == 2 and food["contested"] == 2
    assert food["consensus"] == {"positive": 1, "neutral": 0, "negative": 0, "mixed": 1}
    ref = stats.f_oneway([1, 2 / 3, 2 / 3, -1], [1 / 3, -1 / 3])
    assert food["anova"]["F"] == pytest.approx(ref.statistic, rel=1e-3)
    assert s["aspects"]["service"]["restaurants_compared"] == 0 and s["aspects"]["service"]["anova"]["F"] is None
    # stars_rule == actual stars here, so the prediction error is zero at both levels
    assert s["stars"]["review_mae"] == 0 and s["stars"]["restaurant_mae"] == 0
    assert s["sample_vs_yelp"]["mae"] == pytest.approx((abs(3.75 - 4.5) + abs(3.0 - 2.0)) / 2)


def test_text_reports():
    c = compare_restaurants(RESULTS)
    report = format_comparison(c)
    assert "ICC(1)" in report and "Biz a" in report and "Biz b" in report
    detail = format_restaurant(c["restaurants"][0], RESULTS)
    assert "food quote 0 in a1" in detail and "food quote 0 in a4" in detail
    assert "No restaurant" in format_comparison(compare_restaurants(RESULTS[-1:]))


# ---------------------------------------------------------------- sampling out of the tar
def _tar(path, businesses, reviews):
    with tarfile.open(path, "w") as tar:
        for name, rows in (("yelp_academic_dataset_business.json", businesses),
                           ("yelp_academic_dataset_review.json", reviews)):
            data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _biz(bid, cats="Restaurants", count=5, stars=4.0):
    return {"business_id": bid, "name": f"Biz {bid}", "city": "Tampa", "categories": cats,
            "stars": stars, "review_count": count}


def _rev(bid, i, n_chars=80):
    return {"review_id": f"{bid}{i}", "business_id": bid, "stars": 1 + i % 5,
            "text": ("x" * n_chars), "date": f"2020-01-{10 - i:02d} 12:00:00"}


def test_restaurant_sampler(tmp_path):
    path = tmp_path / "yelp.tar"
    businesses = [_biz("A", stars=4.5), _biz("B", stars=2.0), _biz("C"),
                  _biz("D", cats="Doctors", count=50), _biz("E", count=2)]
    reviews = ([_rev("A", i) for i in range(5)] + [_rev("B", i) for i in range(4)] + [_rev("B", 9, n_chars=10)]
               + [_rev("C", i) for i in range(2)] + [_rev("C", i, n_chars=10) for i in range(2, 5)]
               + [_rev("D", i) for i in range(5)] + [_rev("E", i) for i in range(2)])
    _tar(path, businesses, reviews)
    logs = []
    cfg = RestaurantSampleConfig(businesses=5, per_business=3, seed=1)
    out = sample_restaurants_from_tar(path, cfg, log=logs.append)
    # C has only 2 reviews in the length window, D is not a restaurant, E lists too few reviews
    assert sorted({r.business_id for r in out}) == ["A", "B"] and len(out) == 6
    assert any("warning" in m for m in logs)
    for bid in ("A", "B"):                       # grouped, oldest first, carrying the Yelp rating
        rs = [r for r in out if r.business_id == bid]
        assert [r.date for r in rs] == sorted(r.date for r in rs)
        assert rs[0].business_stars == {"A": 4.5, "B": 2.0}[bid] and rs[0].business_review_count == 5
        assert all(len(r.text) >= cfg.min_chars for r in rs)
    ids = [r.business_id for r in out]
    assert ids == sorted(ids, key=ids.index)     # each restaurant's reviews are contiguous
    assert [r.review_id for r in sample_restaurants_from_tar(path, cfg, log=lambda m: None)] == [r.review_id for r in out]


def test_review_from_dict_defaults_for_older_files():
    r = Review.from_dict({"review_id": "x", "business_id": "b", "stars": 4, "text": "t"})
    assert r.business_stars is None and r.business_review_count is None and r.city == ""
