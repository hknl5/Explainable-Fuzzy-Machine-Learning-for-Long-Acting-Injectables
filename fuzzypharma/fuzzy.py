"""Fuzzification of numeric features into linguistic membership degrees.

A crisp interval assigns a value to exactly one bin, so two nearly identical
measurements straddling a cutoff receive completely different representations.
Fuzzy sets remove that discontinuity: a value carries a graded membership in
several overlapping sets at once.

``FuzzyFeaturizer`` is a scikit-learn transformer, so the knots that define the
sets are learned in ``fit`` and merely applied in ``transform``. Fitting it
inside a cross-validation fold is therefore what keeps membership boundaries
from absorbing held-out information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#: Names used when a feature is split into three sets.
THREE_SET_LABELS = ("Low", "Medium", "High")


def default_labels(n_sets: int) -> tuple[str, ...]:
    """Linguistic labels for ``n_sets`` fuzzy sets."""
    if n_sets == 3:
        return THREE_SET_LABELS
    if n_sets == 5:
        return ("VeryLow", "Low", "Medium", "High", "VeryHigh")
    return tuple(f"Set{i}" for i in range(n_sets))


@dataclass
class FeatureKnots:
    """Learned membership parameters for one feature."""

    feature: str
    knots: np.ndarray | None
    #: How the knots were obtained: ``quantile``, ``uniform``,
    #: ``quantile-unique``, ``uniform-unique``, or ``skipped``.
    source: str
    fuzzifiable: bool

    @property
    def note(self) -> str:
        if not self.fuzzifiable:
            return "not fuzzifiable (no strictly increasing knots); crisp value only"
        if self.source.endswith("-unique"):
            return "knots from distinct values (raw quantiles were tied)"
        return "knots from raw quantiles"


def _strictly_increasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) > 0))


def _quantile_levels(n_sets: int) -> np.ndarray:
    """Quantile levels for the set centers.

    For ``n_sets=3`` this is (0.25, 0.50, 0.75), matching the exploratory
    preview in ``archive/starting.ipynb``.
    """
    return np.linspace(0.25, 0.75, n_sets)


def _candidate_knots(values: np.ndarray, n_sets: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        return np.quantile(values, _quantile_levels(n_sets))
    if strategy == "uniform":
        return np.linspace(values.min(), values.max(), n_sets)
    raise ValueError(f"Unknown knot strategy: {strategy!r}")


def fit_knots(values: np.ndarray, n_sets: int = 3, strategy: str = "quantile") -> FeatureKnots:
    """Learn knots for one feature, degrading gracefully on discrete data.

    Two features in this dataset defeat plain quantile knots: ``LA/GA`` is 71%
    a single value and ``Solubility Enhancer Concentration`` is 64% zero, so
    their 25th and 50th percentiles coincide and the membership functions would
    divide by zero. The fallback recomputes the knots over the *distinct*
    observed values, which spreads them across the real support instead of
    collapsing onto the dominant mass.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return FeatureKnots("", None, "skipped", False)

    knots = _candidate_knots(values, n_sets, strategy)
    if _strictly_increasing(knots):
        return FeatureKnots("", knots, strategy, True)

    unique = np.unique(values)
    if unique.size >= n_sets:
        knots = _candidate_knots(unique, n_sets, strategy)
        if _strictly_increasing(knots):
            return FeatureKnots("", knots, f"{strategy}-unique", True)

    # Constant or near-constant within this training fold: membership would be
    # degenerate, so the feature contributes its crisp value only.
    return FeatureKnots("", None, "skipped", False)


def triangular_memberships(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Shoulder-triangle memberships; rows sum to 1 (a Ruspini partition).

    The first and last sets are shoulders saturating at 1 beyond the outer
    knots; interior sets are triangles peaked at their knot.
    """
    x = np.asarray(x, dtype=float)[:, None]
    knots = np.asarray(knots, dtype=float)
    n_sets = len(knots)
    out = np.zeros((x.shape[0], n_sets), dtype=float)

    if n_sets == 1:
        return np.ones_like(out)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Left shoulder: 1 below knot 0, ramping down to 0 at knot 1.
        out[:, 0] = np.clip((knots[1] - x[:, 0]) / (knots[1] - knots[0]), 0.0, 1.0)
        # Right shoulder: 0 at the second-to-last knot, 1 beyond the last.
        out[:, -1] = np.clip(
            (x[:, 0] - knots[-2]) / (knots[-1] - knots[-2]), 0.0, 1.0
        )
        for i in range(1, n_sets - 1):
            rising = (x[:, 0] - knots[i - 1]) / (knots[i] - knots[i - 1])
            falling = (knots[i + 1] - x[:, 0]) / (knots[i + 1] - knots[i])
            out[:, i] = np.clip(np.minimum(rising, falling), 0.0, 1.0)

    return out


def gaussian_memberships(
    x: np.ndarray, knots: np.ndarray, width: float = 1.0, normalize: bool = True
) -> np.ndarray:
    """Gaussian memberships centered on the knots.

    Sigma is derived from the local knot spacing scaled by ``width``. Unlike the
    triangular form these do not sum to 1, so ``normalize`` rescales each row
    to keep the two designs on a comparable scale.
    """
    x = np.asarray(x, dtype=float)[:, None]
    knots = np.asarray(knots, dtype=float)

    spacing = np.diff(knots)
    sigma = np.empty_like(knots)
    sigma[0] = spacing[0]
    sigma[-1] = spacing[-1]
    for i in range(1, len(knots) - 1):
        sigma[i] = 0.5 * (spacing[i - 1] + spacing[i])
    sigma = np.maximum(sigma * width, np.finfo(float).eps)

    out = np.exp(-0.5 * ((x - knots) / sigma) ** 2)
    if normalize:
        total = out.sum(axis=1, keepdims=True)
        out = np.divide(out, total, out=np.zeros_like(out), where=total > 0)
    return out


class FuzzyFeaturizer(BaseEstimator, TransformerMixin):
    """Turn numeric columns into fuzzy membership columns.

    Parameters
    ----------
    n_sets:
        Number of linguistic sets per feature (3 -> Low/Medium/High).
    membership:
        ``"triangular"`` or ``"gaussian"``.
    knot_strategy:
        ``"quantile"`` (default) or ``"uniform"`` equal-width over the training
        range.
    width:
        Gaussian width multiplier; ignored for triangular memberships.
    features:
        Columns to fuzzify. ``None`` fuzzifies every column it is fitted on.

    Notes
    -----
    ``fit`` must only ever see training-fold rows. Which features turn out to be
    fuzzifiable is itself decided from training data, so a feature may be
    fuzzified in one fold and passed over in another; each fold trains its own
    model, so this is consistent, and ``report()`` records what happened.
    """

    def __init__(
        self,
        n_sets: int = 3,
        membership: str = "triangular",
        knot_strategy: str = "quantile",
        width: float = 1.0,
        features: list[str] | None = None,
    ):
        self.n_sets = n_sets
        self.membership = membership
        self.knot_strategy = knot_strategy
        self.width = width
        self.features = features

    # -- sklearn API ------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        if self.n_sets < 2:
            raise ValueError("n_sets must be at least 2.")
        if self.membership not in {"triangular", "gaussian"}:
            raise ValueError(f"Unknown membership: {self.membership!r}")

        X = self._as_frame(X)
        self.feature_names_in_ = list(X.columns)
        self.features_ = list(self.features) if self.features is not None else list(X.columns)

        missing = sorted(set(self.features_) - set(X.columns))
        if missing:
            raise ValueError(f"Columns to fuzzify are absent: {missing}")

        self.labels_ = default_labels(self.n_sets)
        self.knots_: dict[str, FeatureKnots] = {}
        for column in self.features_:
            knots = fit_knots(X[column].to_numpy(), self.n_sets, self.knot_strategy)
            knots.feature = column
            self.knots_[column] = knots

        self.fuzzified_ = [c for c in self.features_ if self.knots_[c].fuzzifiable]
        self.skipped_ = [c for c in self.features_ if not self.knots_[c].fuzzifiable]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "knots_"):
            raise RuntimeError("FuzzyFeaturizer must be fitted before transform.")

        X = self._as_frame(X)
        missing = sorted(set(self.fuzzified_) - set(X.columns))
        if missing:
            raise ValueError(f"Columns missing at transform time: {missing}")

        blocks = {}
        for column in self.fuzzified_:
            knots = self.knots_[column].knots
            values = X[column].to_numpy(dtype=float)
            if self.membership == "triangular":
                memberships = triangular_memberships(values, knots)
            else:
                memberships = gaussian_memberships(values, knots, self.width)
            for label, series in zip(self.labels_, memberships.T):
                blocks[f"{column} [{label}]"] = series

        return pd.DataFrame(blocks, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(
            [f"{c} [{label}]" for c in self.fuzzified_ for label in self.labels_]
        )

    # -- introspection ----------------------------------------------------

    def report(self) -> pd.DataFrame:
        """One row per feature describing how its knots were obtained."""
        rows = []
        for column in self.features_:
            knots = self.knots_[column]
            rows.append(
                {
                    "feature": column,
                    "fuzzifiable": knots.fuzzifiable,
                    "knot source": knots.source,
                    "knots": (
                        np.round(knots.knots, 4).tolist() if knots.knots is not None else None
                    ),
                    "note": knots.note,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
