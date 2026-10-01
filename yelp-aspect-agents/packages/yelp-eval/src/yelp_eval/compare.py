"""Compare the reviews of one restaurant with each other.

A restaurant sample (`yelp-agents sample-restaurants`) holds several reviews per restaurant, each
scored by the agent team on food, service and ambience. Per restaurant and aspect this module asks:

  * what do its reviewers say on balance   mean aspect score over the reviews that mention it
  * do they agree                          agreement = share of those reviews on the majority
                                           polarity, and the standard deviation of their scores
  * where exactly do they disagree         the most positive and the most negative quote

and, across restaurants, whether reviews of the same restaurant agree more than reviews of different
restaurants: a one-way ANOVA with restaurants as groups, and ICC(1), the share of score variance that
lies between restaurants. The reviewers' own star ratings get the same treatment as a baseline.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy import stats

from yelp_core.schemas import ASPECTS, Mention, ReviewAnalysis
from yelp_scoring import mention_value

from yelp_eval.metrics import _corr

CONSENSUS = 0.75   # agreement at or above this is a verdict; below it the aspect is "mixed"
MIN_COMPARE = 2    # reviews that must mention an aspect before a restaurant's reviewers are compared on it
POLAR = ("positive", "neutral", "negative")


def one_way(groups: list[list[float]]) -> dict[str, Any]:
    """One-way ANOVA (F, p) and ICC(1) over the groups with >= 2 values; None where undefined.

    ICC(1) = (MSB - MSW) / (MSB + (n0 - 1) MSW), with n0 the size-adjusted mean group size
    for unequal groups. 0 means reviews of the same restaurant agree no more than random reviews.
    """
    groups = [list(map(float, g)) for g in groups if len(g) >= 2]
    k, n = len(groups), sum(map(len, groups))
    out: dict[str, Any] = {"groups": k, "n": n, "F": None, "p": None, "icc1": None}
    if k < 2 or n <= k:
        return out
    grand = np.mean([x for g in groups for x in g])
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((np.asarray(g) - np.mean(g)) ** 2).sum() for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    if msw > 0:
        f = msb / msw
        out.update(F=float(f), p=float(stats.f.sf(f, k - 1, n - k)))
    if msb + (n0 - 1) * msw > 0:
        out["icc1"] = float((msb - msw) / (msb + (n0 - 1) * msw))
    return out


def _sd(xs: list[float]) -> float | None:
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else None


def _quote(r: ReviewAnalysis, m: Mention) -> dict[str, Any]:
    return {"review_id": r.review.review_id, "stars": r.review.stars, "date": r.review.date[:10],
            "quote": m.quote, "value": round(mention_value(m), 4)}


def _aspect_block(rs: list[ReviewAnalysis], aspect: str) -> dict[str, Any]:
    ok = [r for r in rs if not r.aspects[aspect].error]
    said = [r for r in ok if r.aspects[aspect].mentioned]
    scores = [r.aspects[aspect].score for r in said]
    labels = Counter(r.aspects[aspect].sentiment for r in ok)
    polar = {p: labels.get(p, 0) for p in POLAR}
    block: dict[str, Any] = {
        "reviews": len(ok), "mentioned": len(said),
        "mean": float(np.mean(scores)) if scores else None, "sd": _sd(scores),
        "min": min(scores, default=None), "max": max(scores, default=None),
        "labels": {**polar, "not_mentioned": labels.get("not_mentioned", 0)},
        "agreement": None, "consensus": "too_few",
        "contested": polar["positive"] > 0 and polar["negative"] > 0,
        "most_positive": None, "most_negative": None,
    }
    if len(said) >= MIN_COMPARE:
        top, count = max(polar.items(), key=lambda kv: kv[1])
        block["agreement"] = count / len(said)
        block["consensus"] = top if block["agreement"] >= CONSENSUS else "mixed"
    mentions = [(r, m) for r in said for m in r.aspects[aspect].mentions]
    if mentions:
        best = max(mentions, key=lambda x: mention_value(x[1]))
        worst = min(mentions, key=lambda x: mention_value(x[1]))
        block["most_positive"] = _quote(*best) if mention_value(best[1]) > 0 else None
        block["most_negative"] = _quote(*worst) if mention_value(worst[1]) < 0 else None
    return block


def compare_restaurants(results: list[ReviewAnalysis], min_reviews: int = 2) -> dict[str, Any]:
    """Per-restaurant comparison of reviews plus a cross-restaurant summary. Restaurants are
    ordered with the most contested (aspects with both positive and negative reviewers) first."""
    groups: dict[str, list[ReviewAnalysis]] = defaultdict(list)
    for r in results:
        groups[r.review.business_id].append(r)
    groups = {b: sorted(rs, key=lambda r: (r.review.date, r.review.review_id))
              for b, rs in groups.items() if len(rs) >= min_reviews}

    restaurants = []
    for bid, rs in groups.items():
        rv, stars = rs[0].review, [r.review.stars for r in rs]
        aspects = {a: _aspect_block(rs, a) for a in ASPECTS}
        restaurants.append({
            "business_id": bid, "business": rv.business_name, "city": rv.city, "categories": rv.categories,
            "yelp_stars": rv.business_stars, "yelp_review_count": rv.business_review_count,
            "reviews": len(rs), "review_ids": [r.review.review_id for r in rs],
            "stars": {"mean": float(np.mean(stars)), "sd": _sd(stars),
                      "counts": {s: sum(int(x) == s for x in stars) for s in range(1, 6)}},
            "predicted_stars": float(np.mean([r.overall.stars_rule for r in rs])),
            "overall_labels": {p: sum(r.overall.sentiment == p for r in rs) for p in POLAR},
            "aspects": aspects,
            "contested": [a for a in ASPECTS if aspects[a]["contested"]],
        })
    restaurants.sort(key=lambda x: (-len(x["contested"]), x["business"]))
    return {"summary": _summary(groups, restaurants), "restaurants": restaurants}


def _summary(groups: dict[str, list[ReviewAnalysis]], restaurants: list[dict[str, Any]]) -> dict[str, Any]:
    n_reviews = sum(map(len, groups.values()))
    out: dict[str, Any] = {"restaurants": len(restaurants), "reviews": n_reviews,
                           "reviews_per_restaurant": n_reviews / len(restaurants) if restaurants else None,
                           "restaurants_contested": sum(bool(x["contested"]) for x in restaurants),
                           "consensus_threshold": CONSENSUS, "min_compare": MIN_COMPARE}
    aspects: dict[str, Any] = {}
    for a in ASPECTS:
        blocks = [x["aspects"][a] for x in restaurants]
        compared = [b for b in blocks if b["mentioned"] >= MIN_COMPARE]
        means = [(b["mean"], x["yelp_stars"]) for x, b in zip(restaurants, blocks) if b["mean"] is not None]
        with_yelp = [(m, y) for m, y in means if y is not None]
        aspects[a] = {
            "restaurants_mentioning": len(means),
            "restaurants_compared": len(compared),
            "within_sd": float(np.mean([b["sd"] for b in compared])) if compared else None,
            "between_sd": _sd([m for m, _ in means]),
            "anova": one_way([[r.aspects[a].score for r in rs if r.aspects[a].mentioned and not r.aspects[a].error]
                              for rs in groups.values()]),
            "contested": sum(b["contested"] for b in blocks),
            "consensus": {c: sum(b["consensus"] == c for b in compared) for c in (*POLAR, "mixed")},
            "vs_yelp_stars": _corr([m for m, _ in with_yelp], [y for _, y in with_yelp]),
        }
    out["aspects"] = aspects
    out["stars_anova"] = one_way([[r.review.stars for r in rs] for rs in groups.values()])

    # Averaging several reviews should predict a restaurant better than one review predicts itself.
    sampled = [x["stars"]["mean"] for x in restaurants]
    predicted = [x["predicted_stars"] for x in restaurants]
    all_rs = [r for rs in groups.values() for r in rs]
    out["stars"] = {
        "review_mae": float(np.mean([abs(r.overall.stars_rule - r.review.stars) for r in all_rs])) if all_rs else None,
        "restaurant_mae": float(np.mean(np.abs(np.subtract(predicted, sampled)))) if restaurants else None,
        "restaurant_corr": _corr(predicted, sampled),
    }
    yelp = [(s, x["yelp_stars"]) for s, x in zip(sampled, restaurants) if x["yelp_stars"] is not None]
    out["sample_vs_yelp"] = {
        "mae": float(np.mean([abs(s - y) for s, y in yelp])) if yelp else None,
        **_corr([s for s, _ in yelp], [y for _, y in yelp]),
    }
    return out


# ---------------------------------------------------------------- text report (CLI)
def _f(x: float | None, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def _cell(b: dict[str, Any]) -> str:
    if b["mean"] is None:
        return "—"
    agree = "" if b["agreement"] is None else f" {b['agreement']:.0%}"
    return f"{b['mean']:+.2f}{agree} n{b['mentioned']}" + ("*" if b["contested"] else "")


def format_comparison(c: dict[str, Any], top: int | None = None) -> str:
    s, rows = c["summary"], c["restaurants"]
    if not rows:
        return "No restaurant has >= 2 analysed reviews. Sample with `yelp-agents sample-restaurants`."
    lines = [f"Restaurants: {s['restaurants']}   reviews: {s['reviews']}   "
             f"({s['reviews_per_restaurant']:.1f} per restaurant)   "
             f"with a contested aspect: {s['restaurants_contested']}", "",
             "Do reviewers of the same restaurant agree?  (reviews that mention the aspect; restaurants as groups)",
             f"  {'':10}{'compared':>10}{'within sd':>11}{'between sd':>12}{'ICC(1)':>8}{'ANOVA p':>10}"
             f"{'contested':>11}{'ρ vs Yelp★':>12}"]
    for a, v in s["aspects"].items():
        lines.append(f"  {a:10}{v['restaurants_compared']:>7}/{s['restaurants']:<2}{_f(v['within_sd'], '.2f'):>11}"
                     f"{_f(v['between_sd'], '.2f'):>12}{_f(v['anova']['icc1'], '.2f'):>8}"
                     f"{_f(v['anova']['p'], '.3g'):>10}{v['contested']:>8}/{s['restaurants']:<2}"
                     f"{_f(v['vs_yelp_stars']['spearman'], '.2f'):>12}")
    sa = s["stars_anova"]
    lines += [f"  {'stars':10}{'':33}{_f(sa['icc1'], '.2f'):>8}{_f(sa['p'], '.3g'):>10}"
              "   <- the reviewers' own 1-5 ratings, as a baseline",
              f"  compared = restaurants with >= {s['min_compare']} reviews mentioning the aspect; "
              "contested = at least one positive and one negative reviewer", ""]
    st, sy = s["stars"], s["sample_vs_yelp"]
    lines += [f"Predicted stars   MAE per review {_f(st['review_mae'], '.2f')}  ->  per restaurant (mean of its reviews) "
              f"{_f(st['restaurant_mae'], '.2f')}, Pearson r {_f(st['restaurant_corr']['pearson'], '.2f')}",
              f"Sample vs Yelp    mean sampled stars vs the restaurant's Yelp rating: MAE {_f(sy['mae'], '.2f')}, "
              f"Pearson r {_f(sy['pearson'], '.2f')}", "",
              "Per restaurant: mean aspect score, agreement, reviews mentioning it (* = contested)",
              f"  {'restaurant':30}{'city':14}{'Yelp★':>6}{'sample★':>8}{'pred★':>6}  "
              f"{'food':18}{'service':18}{'ambience':18}"]
    for x in rows[:top]:
        lines.append(f"  {x['business'][:29]:30}{x['city'][:13]:14}{_f(x['yelp_stars'], '.1f'):>6}"
                     f"{x['stars']['mean']:>8.1f}{x['predicted_stars']:>6.1f}  "
                     + "".join(f"{_cell(x['aspects'][a]):18}" for a in ASPECTS))
    return "\n".join(lines)


def format_restaurant(x: dict[str, Any], results: list[ReviewAnalysis]) -> str:
    """One restaurant's reviews side by side, with the quotes that pull furthest apart."""
    by_id = {r.review.review_id: r for r in results}
    lines = [f"{x['business']} · {x['city']} · Yelp {_f(x['yelp_stars'], '.1f')}★ over "
             f"{x['yelp_review_count'] or 'n/a'} reviews",
             f"  {x['reviews']} sampled reviews: mean {x['stars']['mean']:.1f}★ "
             f"(sd {_f(x['stars']['sd'], '.2f')}), lead agent predicts {x['predicted_stars']:.1f}★", ""]
    for a in ASPECTS:
        b = x["aspects"][a]
        lab = b["labels"]
        lines.append(f"  {a:9} {b['consensus']:8} mean {_f(b['mean'], '+.2f')} sd {_f(b['sd'], '.2f')}  "
                     f"{lab['positive']} positive · {lab['neutral']} neutral · {lab['negative']} negative · "
                     f"{lab['not_mentioned']} silent")
        for key in ("most_positive", "most_negative"):
            q = b[key]
            if q:
                lines.append(f"            {q['value']:+.2f}  “{q['quote'][:90]}”  ({q['stars']:.0f}★, {q['date']})")
    lines += ["", f"  {'date':11}{'★':>3}{'pred':>6}  {'food':>7}{'service':>9}{'ambience':>10}  review"]
    for rid in x["review_ids"]:
        r = by_id[rid]
        v = "".join(f"{(f'{r.aspects[a].score:+.2f}' if r.aspects[a].mentioned else '—'):>{w}}"
                    for a, w in zip(ASPECTS, (7, 9, 10)))
        lines.append(f"  {r.review.date[:10]:11}{r.review.stars:>3.0f}{r.overall.stars_rule:>6.1f}  {v}  "
                     f"{r.review.text[:70].replace(chr(10), ' ')}")
    return "\n".join(lines)
