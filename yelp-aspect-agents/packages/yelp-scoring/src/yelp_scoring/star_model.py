"""Calibrated star prediction: ridge regression on the 3-D aspect vector.

Features per review: the three aspect scores, three "was it mentioned" indicators
(so 'not mentioned' is not confused with 'neutral'), and the lead agent's overall score.
Evaluation uses k-fold cross_val_predict, so every reported prediction comes from a
model that never saw that review.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from yelp_core.schemas import ASPECTS, ReviewAnalysis

FEATURES: tuple[str, ...] = (*ASPECTS, *(f"{a}_mentioned" for a in ASPECTS), "overall")
MIN_TRAIN = 10


def feature_row(a: ReviewAnalysis) -> list[float]:
    return ([a.aspects[x].score for x in ASPECTS]
            + [1.0 if a.aspects[x].mentioned else 0.0 for x in ASPECTS]
            + [a.overall.score])


def _xy(results: list[ReviewAnalysis]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([feature_row(r) for r in results], dtype=float)
    y = np.array([r.review.stars for r in results], dtype=float)
    return X, y


class StarModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None

    @staticmethod
    def usable(results: list[ReviewAnalysis]) -> bool:
        return len(results) >= MIN_TRAIN and len({r.review.stars for r in results}) >= 2

    def fit(self, results: list[ReviewAnalysis]) -> "StarModel":
        X, y = _xy(results)
        self.model = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, analyses: list[ReviewAnalysis]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("StarModel.fit() first")
        X = np.array([feature_row(a) for a in analyses], dtype=float)
        return np.clip(self.model.predict(X), 1.0, 5.0)

    def cross_validated(self, results: list[ReviewAnalysis], folds: int = 5, seed: int = 0) -> np.ndarray:
        """Out-of-fold predictions, one per review, in input order."""
        X, y = _xy(results)
        k = max(2, min(folds, len(results)))
        pred = cross_val_predict(Ridge(alpha=self.alpha), X, y, cv=KFold(k, shuffle=True, random_state=seed))
        return np.clip(pred, 1.0, 5.0)

    def coefficients(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("StarModel.fit() first")
        out = {f: float(c) for f, c in zip(FEATURES, self.model.coef_)}
        out["intercept"] = float(self.model.intercept_)
        return out
