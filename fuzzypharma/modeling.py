"""Leakage-safe feature representations and evaluation helpers for all three tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import METHOD_COL, STATIC_FEATURES
from .features import Representation, encode_formulation_method
from .fuzzy import FuzzyFeaturizer


ENGINEERED_FEATURES = [
    "Log Drug MW",
    "Log Polymer MW",
    "Log Particle Size",
    "Log Initial DPR",
    "Log Enhancer Concentration",
    "Drug Polarity Density",
    "Drug-to-Polymer MW Ratio",
    "Effective Encapsulated Fraction",
    "Loading Realization",
    "Hydrophobicity × Lactide Fraction",
    "Polymer MW per Particle Size",
    "Enhancer Present",
]


def engineer_numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, domain-motivated features without learning from data."""
    x = frame[STATIC_FEATURES].astype(float).copy()
    eps = np.finfo(float).eps
    out = x.copy()
    out["Log Drug MW"] = np.log1p(x["Drug MW"].clip(lower=0))
    out["Log Polymer MW"] = np.log1p(x["Polymer MW"].clip(lower=0))
    out["Log Particle Size"] = np.log1p(x["Particle Size"].clip(lower=0))
    out["Log Initial DPR"] = np.log1p(x["Initial Drug-to-Polymer Ratio"].clip(lower=0))
    out["Log Enhancer Concentration"] = np.log1p(x["Solubility Enhancer Concentration"].clip(lower=0))
    out["Drug Polarity Density"] = x["Drug TPSA"] / x["Drug MW"].clip(lower=eps)
    out["Drug-to-Polymer MW Ratio"] = x["Drug MW"] / x["Polymer MW"].clip(lower=eps)
    out["Effective Encapsulated Fraction"] = (
        x["Initial Drug-to-Polymer Ratio"] * x["Drug Encapsulation Efficiency"] / 100.0
    )
    out["Loading Realization"] = x["Drug Loading Capacity"] / x[
        "Initial Drug-to-Polymer Ratio"
    ].clip(lower=eps)
    out["Hydrophobicity × Lactide Fraction"] = x["Drug LogP"] * x["LA/GA"] / 100.0
    out["Polymer MW per Particle Size"] = x["Polymer MW"] / x["Particle Size"].clip(lower=eps)
    out["Enhancer Present"] = (x["Solubility Enhancer Concentration"] > 0).astype(float)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Engineered features contain non-finite values.")
    return out


class ModelRepresentation(BaseEstimator, TransformerMixin):
    """Crisp/fuzzy/hybrid representations with an optional engineered crisp block."""

    def __init__(
        self,
        kind: str = "crisp",
        membership: str = "triangular",
        n_sets: int = 3,
        knot_strategy: str = "quantile",
    ):
        self.kind = kind
        self.membership = membership
        self.n_sets = n_sets
        self.knot_strategy = knot_strategy

    def fit(self, X: pd.DataFrame, y=None):
        if self.kind in {"crisp", "fuzzy", "hybrid"}:
            self.base_ = Representation(
                self.kind,
                membership=self.membership,
                n_sets=self.n_sets,
                knot_strategy=self.knot_strategy,
            ).fit(X)
            self.scaler_ = self.fuzzifier_ = None
        elif self.kind in {"engineered", "hybrid_engineered"}:
            numeric = engineer_numeric_features(X)
            self.scaler_ = StandardScaler().fit(numeric)
            self.fuzzifier_ = None
            if self.kind == "hybrid_engineered":
                self.fuzzifier_ = FuzzyFeaturizer(
                    membership=self.membership,
                    n_sets=self.n_sets,
                    knot_strategy=self.knot_strategy,
                ).fit(X[STATIC_FEATURES])
            self.base_ = None
        else:
            raise ValueError(f"Unknown representation: {self.kind}")
        self.feature_names_out_ = list(self.transform(X.head(1)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.base_ is not None:
            return self.base_.transform(X)
        numeric = engineer_numeric_features(X)
        crisp = pd.DataFrame(
            self.scaler_.transform(numeric), columns=numeric.columns, index=X.index
        )
        parts = [crisp]
        if self.fuzzifier_ is not None:
            parts.append(self.fuzzifier_.transform(X[STATIC_FEATURES]))
        parts.append(encode_formulation_method(X[METHOD_COL]))
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self):
        return list(self.feature_names_out_)


def regression_rows(y_true, y_pred, fold, representation, model, targets):
    rows = []
    for j, target in enumerate(targets):
        truth, pred = y_true[:, j], y_pred[:, j]
        r = pearsonr(truth, pred).statistic if np.std(pred) > 0 else np.nan
        rows.append(
            {
                "fold": fold,
                "representation": representation,
                "model": model,
                "target": target,
                "RMSE": mean_squared_error(truth, pred) ** 0.5,
                "MAE": mean_absolute_error(truth, pred),
                "R2": r2_score(truth, pred),
                "Pearson_r": r,
            }
        )
    return rows


def classification_row(y_true, y_pred, y_prob, fold, representation, model):
    return {
        "fold": fold,
        "representation": representation,
        "model": model,
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def curve_row(y_true, y_pred, fold, representation, model, grid):
    trap = getattr(np, "trapezoid", None) or np.trapz
    true_auc = trap(y_true, grid, axis=1)
    pred_auc = trap(y_pred, grid, axis=1)
    return {
        "fold": fold,
        "representation": representation,
        "model": model,
        "curve_RMSE": mean_squared_error(y_true.ravel(), y_pred.ravel()) ** 0.5,
        "curve_MAE": mean_absolute_error(y_true.ravel(), y_pred.ravel()),
        "mean_profile_RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)).mean(),
        "AUC_MAE": mean_absolute_error(true_auc, pred_auc),
        "endpoint_MAE": mean_absolute_error(y_true[:, -1], y_pred[:, -1]),
    }
