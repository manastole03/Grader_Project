"""Accuracy of the agent scores against Yelp star ratings.

Yelp has no aspect-level labels, so the star rating is the only ground truth:
1-2 stars = negative, 3 = neutral, 4-5 = positive. Everything here uses
scikit-learn / scipy implementations, and every headline number carries a 95%
interval because samples are small (tens to hundreds of reviews).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error,
)

from yelp_core.schemas import ASPECTS, SENTIMENT_LABELS, ReviewAnalysis
from yelp_scoring import StarModel, stars_to_label

LABELS = list(SENTIMENT_LABELS)
BOOTSTRAP_ROUNDS = 2000


def load_results(path: str | Path) -> list[ReviewAnalysis]:
    """Latest analysis per review. A half-written final line (a run still appending) is skipped."""
    seen: dict[str, ReviewAnalysis] = {}
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        a = ReviewAnalysis.from_dict(d)
        seen[a.review.review_id] = a
    return list(seen.values())


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (better than the normal approx at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _bootstrap_macro_f1(y_true: list[str], y_pred: list[str], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    t, p = np.array(y_true), np.array(y_pred)
    n = len(t)
    scores = [f1_score(t[idx], p[idx], labels=LABELS, average="macro", zero_division=0)
              for idx in (rng.integers(0, n, n) for _ in range(BOOTSTRAP_ROUNDS))]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def _corr(x: list[float], y: list[float]) -> dict[str, Any]:
    """Pearson and Spearman with p-values; None when undefined (n < 3 or a constant input)."""
    out: dict[str, Any] = {"n": len(x), "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return out
    r = stats.pearsonr(x, y)
    s = stats.spearmanr(x, y)
    out.update(pearson=float(r.statistic), pearson_p=float(r.pvalue),
               spearman=float(s.statistic), spearman_p=float(s.pvalue))
    return out


def classification_block(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Note the argument order: sklearn expects (y_true, y_pred)."""
    k = sum(a == b for a, b in zip(y_true, y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci": wilson_interval(k, len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_ci": _bootstrap_macro_f1(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),  # rows = actual
    }


def star_block(actual: list[float], pred: list[float]) -> dict[str, Any]:
    return {"mae": float(mean_absolute_error(actual, pred)),
            "rmse": float(math.sqrt(mean_squared_error(actual, pred))),
            **_corr(list(pred), list(actual))}


def evaluate(results: list[ReviewAnalysis], folds: int = 5, seed: int = 0) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    y_true = [r.review.label for r in results]
    stars = [r.review.stars for r in results]

    out: dict[str, Any] = {
        "n": len(results),
        "providers": dict(Counter(f"{r.provider}:{r.model}" for r in results)),
        "reviews_with_errors": sum(any(a.error for a in r.aspects.values()) or bool(r.overall.error) for r in results),
        "latency_s": {"mean": float(np.mean([r.latency_s for r in results])),
                      "median": float(np.median([r.latency_s for r in results]))},
        "lead_agent": classification_block(y_true, [r.overall.sentiment for r in results]),
        "stars_rule": star_block(stars, [r.overall.stars_rule for r in results]),
        "calibrated": None,
    }

    if StarModel.usable(results):
        model = StarModel()
        oof = model.cross_validated(results, folds=folds, seed=seed)
        model.fit(results)
        out["calibrated"] = {
            "folds": max(2, min(folds, len(results))),
            "stars": star_block(stars, oof.tolist()),
            "sentiment": classification_block(y_true, [stars_to_label(p) for p in oof]),
            "coefficients": model.coefficients(),
            "oof_predictions": {r.review.review_id: float(p) for r, p in zip(results, oof)},
        }

    aspects: dict[str, Any] = {}
    for a in ASPECTS:
        ok = [r for r in results if not r.aspects[a].error]
        mentioned = [r for r in ok if r.aspects[a].mentioned]
        kept = sum(len(r.aspects[a].mentions) for r in ok)
        dropped = sum(len(r.aspects[a].dropped) for r in ok)
        aspects[a] = {
            "n": len(ok),
            "mention_rate": len(mentioned) / len(ok) if ok else 0.0,
            "mention_rate_ci": wilson_interval(len(mentioned), len(ok)),
            "mean_score": float(np.mean([r.aspects[a].score for r in mentioned])) if mentioned else None,
            "vs_stars": _corr([r.aspects[a].score for r in mentioned], [r.review.stars for r in mentioned]),
            "labels": dict(Counter(r.aspects[a].sentiment for r in ok)),
            "mentions_kept": kept,
            "mentions_dropped": dropped,
            "grounding_rate": kept / (kept + dropped) if kept + dropped else None,
            "errors": len(results) - len(ok),
        }
    out["aspects"] = aspects
    kept = sum(v["mentions_kept"] for v in aspects.values())
    dropped = sum(v["mentions_dropped"] for v in aspects.values())
    out["grounding"] = {"kept": kept, "dropped": dropped, "rate": kept / (kept + dropped) if kept + dropped else None}
    return out


def results_frame(results: list[ReviewAnalysis], oof: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per review, for tables and charts."""
    rows = []
    for r in results:
        row = {"review_id": r.review.review_id, "business": r.review.business_name, "city": r.review.city,
               "date": r.review.date[:10], "stars": r.review.stars, "label": r.review.label,
               "text": r.review.text, "overall": r.overall.sentiment, "overall_score": r.overall.score,
               "stars_rule": r.overall.stars_rule,
               "stars_calibrated": (oof or {}).get(r.review.review_id), "latency_s": r.latency_s}
        for a in ASPECTS:
            row[a] = r.aspects[a].score if r.aspects[a].mentioned else None
            row[f"{a}_label"] = r.aspects[a].sentiment
            row[f"{a}_n"] = len(r.aspects[a].mentions)
        rows.append(row)
    return pd.DataFrame(rows)


def business_rollup(results: list[ReviewAnalysis], min_reviews: int = 1) -> pd.DataFrame:
    """Mean aspect score per business, over the reviews that mention each aspect."""
    groups: dict[str, list[ReviewAnalysis]] = defaultdict(list)
    for r in results:
        groups[r.review.business_id].append(r)
    rows = []
    for bid, rs in groups.items():
        if len(rs) < min_reviews:
            continue
        row: dict[str, Any] = {"business_id": bid, "business": rs[0].review.business_name,
                               "city": rs[0].review.city, "reviews": len(rs),
                               "mean_stars": float(np.mean([r.review.stars for r in rs]))}
        for a in ASPECTS:
            m = [r.aspects[a].score for r in rs if r.aspects[a].mentioned]
            row[a] = float(np.mean(m)) if m else None
            row[f"{a}_reviews"] = len(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["reviews", "business"], ascending=[False, True]).reset_index(drop=True) if rows else df


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _class_lines(block: dict[str, Any], title: str) -> list[str]:
    rep = block["report"]
    lines = [title,
             f"  accuracy {_f(block['accuracy'])} 95% CI {_ci(block['accuracy_ci'])}   "
             f"macro-F1 {_f(block['macro_f1'])} 95% CI {_ci(block['macro_f1_ci'])}",
             f"  {'':10}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}"]
    for lab in LABELS:
        v = rep[lab]
        lines.append(f"  {lab:10}{v['precision']:>10.3f}{v['recall']:>8.3f}{v['f1-score']:>8.3f}{int(v['support']):>9}")
    lines.append("  confusion (rows = actual, cols = predicted: neg / neu / pos)")
    for lab, row in zip(LABELS, block["confusion"]):
        lines.append(f"  {lab:10}" + "".join(f"{v:>6}" for v in row))
    return lines


def format_report(m: dict[str, Any]) -> str:
    if not m.get("n"):
        return "No results."
    lines = [f"Reviews evaluated: {m['n']}   with agent errors: {m['reviews_with_errors']}   "
             f"backends: {m['providers']}   latency/review: mean {m['latency_s']['mean']:.1f}s", ""]
    lines += _class_lines(m["lead_agent"], "Overall sentiment (lead agent) vs star label [1-2 neg | 3 neu | 4-5 pos]")
    sr = m["stars_rule"]
    lines += ["", "Predicted stars",
              f"  {'method':28}{'MAE':>7}{'RMSE':>7}{'Pearson':>9}{'Spearman':>10}",
              f"  {'rule: 3 + 2*overall':28}{sr['mae']:>7.3f}{sr['rmse']:>7.3f}{_f(sr['pearson']):>9}{_f(sr['spearman']):>10}"]
    cal = m.get("calibrated")
    if cal:
        cs = cal["stars"]
        lines.append(f"  {'ridge, ' + str(cal['folds']) + '-fold out-of-fold':28}{cs['mae']:>7.3f}{cs['rmse']:>7.3f}"
                     f"{_f(cs['pearson']):>9}{_f(cs['spearman']):>10}")
        lines += ["", *_class_lines(cal["sentiment"], "Sentiment from calibrated stars (out-of-fold)")]
        coef = cal["coefficients"]
        lines += ["", "  ridge coefficients (stars per unit feature): " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in coef.items())]
    else:
        lines += ["  (calibrated model needs >= 10 results with >= 2 distinct star ratings)"]
    lines += ["", "Per-dimension (the 3 axes; correlations over reviews that mention the aspect)",
              f"  {'aspect':10}{'mentioned':>10}{'mean':>8}{'Spearman':>10}{'p':>8}{'grounded':>10}"]
    for a, v in m["aspects"].items():
        vs = v["vs_stars"]
        g = "n/a" if v["grounding_rate"] is None else f"{v['grounding_rate']:.0%}"
        lines.append(f"  {a:10}{v['mention_rate']:>10.0%}{_f(v['mean_score']):>8}{_f(vs['spearman']):>10}"
                     f"{_f(vs['spearman_p']):>8}{g:>10}")
    return "\n".join(lines)
