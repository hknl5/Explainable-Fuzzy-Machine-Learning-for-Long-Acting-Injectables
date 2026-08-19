"""The controlled crisp / fuzzy / hybrid comparison under drug-grouped CV.

Every experiment here obeys the same three rules:

1. Folds come from :func:`fuzzypharma.grouped_cv.assign_smiles_folds` and are
   frozen once, so all arms see identical drug partitions.
2. Anything learned from the data distribution -- the crisp scaler, the fuzzy
   knots, the classification undersampler -- is fitted on the training fold and
   only applied to the held-out fold.
3. The model, its hyperparameters and its seed are identical across arms. No
   arm gets a tuning budget the others do not, so a difference in score is a
   difference in representation.

Fuzzy knots and scaler statistics are fitted on the training fold's
*formulation* rows even when the model is trained on measurement rows. Fitting
them on measurement rows would weight each formulation by how many time points
it happens to contain, so a 48-point curve would pull the quantiles almost ten
times as hard as a 5-point curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from .config import ID_COL, RELEASE_COL, SEED, STATIC_FEATURES
from .features import Representation
from .folds import FOLD_COL, undersample_majority
from .grouped_cv import LOG_TIME_COL, iter_positions
from .targets import auc_class_binary

#: One fixed configuration used by every arm. Deliberately untuned: a search
#: run per arm would confound representation with search luck, and a search run
#: across folds would leak held-out data into model selection.
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1,
    random_state=SEED,
    n_jobs=4,
    tree_method="hist",
)

#: The arms of the comparison. ``knot_strategy`` is irrelevant to ``crisp`` and
#: is recorded as ``"-"`` there.
ARMS = [
    ("crisp", "-"),
    ("fuzzy", "quantile"),
    ("fuzzy", "uniform"),
    ("hybrid", "quantile"),
    ("hybrid", "uniform"),
]


def arm_label(kind: str, knot_strategy: str) -> str:
    return kind if knot_strategy == "-" else f"{kind} ({knot_strategy})"


def model_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename membership columns into names XGBoost will accept.

    ``FuzzyFeaturizer`` emits ``Particle Size [Low]``, but XGBoost rejects any
    feature name containing ``[``, ``]`` or ``<``. Rewriting it as
    ``Particle Size is Low`` satisfies that and happens to be exactly how the
    condition should read in an extracted IF-THEN rule, so the same names carry
    through to the rule and SHAP output unchanged.
    """
    renamed = {
        column: column.replace(" [", " is ").replace("]", "")
        for column in frame.columns
        if column.endswith("]")
    }
    return frame.rename(columns=renamed) if renamed else frame


# --- Task C diagnostics --------------------------------------------------


def knot_report(
    model_table: pd.DataFrame, knot_strategy: str = "quantile"
) -> pd.DataFrame:
    """Per-fold knot values and saturation counts for every static feature.

    Saturation is the share of *validation* rows whose ``High`` membership is
    exactly 1, i.e. values at or beyond the upper knot. A large share means the
    top of the range has been flattened into one linguistic label and the model
    can no longer tell a 46 kDa polymer from a 215 kDa one through that feature.
    """
    rows = []
    for fold, train, valid in iter_positions(model_table):
        representation = Representation(
            "fuzzy", knot_strategy=knot_strategy
        ).fit(model_table.iloc[train])
        memberships = representation.fuzzifier_.transform(
            model_table.iloc[valid][STATIC_FEATURES]
        )

        for feature in STATIC_FEATURES:
            learned = representation.fuzzifier_.knots_[feature]
            high = f"{feature} [High]"
            low = f"{feature} [Low]"
            rows.append(
                {
                    "fold": fold,
                    "knot strategy": knot_strategy,
                    "feature": feature,
                    "fuzzifiable": learned.fuzzifiable,
                    "knot source": learned.source,
                    "knots": (
                        np.round(learned.knots, 4).tolist()
                        if learned.knots is not None
                        else None
                    ),
                    "valid High=1 share": (
                        round(float((memberships[high] >= 1.0).mean()), 4)
                        if high in memberships
                        else np.nan
                    ),
                    "valid Low=1 share": (
                        round(float((memberships[low] >= 1.0).mean()), 4)
                        if low in memberships
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


# --- Metrics -------------------------------------------------------------


def _pearson(truth: np.ndarray, pred: np.ndarray) -> float:
    """Pearson r, or NaN when a constant prediction makes it undefined."""
    if np.std(pred) == 0 or np.std(truth) == 0:
        return float("nan")
    return float(pearsonr(truth, pred).statistic)


def regression_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "RMSE": float(mean_squared_error(truth, pred) ** 0.5),
        "MAE": float(mean_absolute_error(truth, pred)),
        "Pearson_r": _pearson(truth, pred),
        "R2": float(r2_score(truth, pred)),
    }


# --- Experiment runners --------------------------------------------------


@dataclass
class FoldFit:
    """What one fold of one arm produced, kept for reuse by Tasks E and F."""

    fold: int
    model: object
    representation: Representation
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    valid_positions: np.ndarray
    predictions: np.ndarray = field(default=None)


def _fit_representation(
    fit_frame: pd.DataFrame, kind: str, knot_strategy: str
) -> Representation:
    """Fit one representation on training-fold formulation rows."""
    kwargs = {} if kind == "crisp" else {"knot_strategy": knot_strategy}
    return Representation(kind, **kwargs).fit(fit_frame)


def run_static_regression(
    model_table: pd.DataFrame,
    target: str,
    kind: str,
    knot_strategy: str,
) -> tuple[pd.DataFrame, list[FoldFit]]:
    """Time setting (b): predict one early-release target from static features.

    Time is not an input. The target is the release interpolated at a fixed
    absolute time (day 1, 2 or 3), so the model must place the whole curve from
    formulation characteristics alone.
    """
    usable = model_table[model_table[target].notna()].reset_index(drop=True)
    dropped = len(model_table) - len(usable)

    rows, fits = [], []
    for fold, train, valid in iter_positions(usable):
        train_frame, valid_frame = usable.iloc[train], usable.iloc[valid]
        representation = _fit_representation(train_frame, kind, knot_strategy)

        X_train = model_ready(representation.transform(train_frame))
        X_valid = model_ready(representation.transform(valid_frame))
        y_train, y_valid = train_frame[target], valid_frame[target]

        model = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)
        pred = model.predict(X_valid)

        rows.append(
            {
                "time setting": "(b) time-independent",
                "target": target,
                "representation": arm_label(kind, knot_strategy),
                "fold": fold,
                "n valid": len(valid),
                "n dropped (no interpolation support)": dropped,
                **regression_metrics(y_valid.to_numpy(), pred),
            }
        )
        fits.append(
            FoldFit(
                fold, model, representation, X_train, X_valid, y_train, y_valid,
                valid, pred,
            )
        )
    return pd.DataFrame(rows), fits


def run_pointwise_regression(
    measurements: pd.DataFrame,
    model_table: pd.DataFrame,
    kind: str,
    knot_strategy: str,
) -> tuple[pd.DataFrame, list[FoldFit]]:
    """Time setting (a): predict release at every measured time point.

    ``log1p(Time)`` enters as a crisp input in **all** arms, including
    fuzzy-only. Fuzzification is a representation for static formulation
    characteristics; time is the within-curve axis, not a characteristic of the
    formulation, so fuzzifying it would change what the arms are comparing.
    This is a stated exception, not an oversight.
    """
    rows, fits = [], []
    for fold, train, valid in iter_positions(measurements):
        # Knots and scaler come from the training fold's formulations, one row
        # each, so long curves do not dominate the quantiles.
        train_formulations = model_table[model_table[FOLD_COL] != fold]
        representation = _fit_representation(
            train_formulations, kind, knot_strategy
        )

        train_frame, valid_frame = measurements.iloc[train], measurements.iloc[valid]
        X_train = model_ready(representation.transform(train_frame))
        X_valid = model_ready(representation.transform(valid_frame))
        X_train[LOG_TIME_COL] = train_frame[LOG_TIME_COL].to_numpy()
        X_valid[LOG_TIME_COL] = valid_frame[LOG_TIME_COL].to_numpy()

        y_train, y_valid = train_frame[RELEASE_COL], valid_frame[RELEASE_COL]

        model = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)
        pred = model.predict(X_valid)

        rows.append(
            {
                "time setting": "(a) time-dependent",
                "target": "Release (pointwise)",
                "representation": arm_label(kind, knot_strategy),
                "fold": fold,
                "n valid": len(valid),
                "n dropped (no interpolation support)": 0,
                **regression_metrics(y_valid.to_numpy(), pred),
            }
        )
        fits.append(
            FoldFit(
                fold, model, representation, X_train, X_valid, y_train, y_valid,
                valid, pred,
            )
        )
    return pd.DataFrame(rows), fits


def run_auc_classification(
    model_table: pd.DataFrame, kind: str, knot_strategy: str, seed: int = SEED
) -> tuple[pd.DataFrame, list[dict]]:
    """AUC <= 0.5 vs > 0.5 from static features, with training-fold undersampling.

    The undersampler is applied to the training fold only. Resampling a
    validation fold would change the class prevalence the metrics are computed
    against and make accuracy and precision uninterpretable.

    The AUC class is a summary of the normalized release curve. It is not a
    clinical or regulatory burst-release label.
    """
    y_all = auc_class_binary(model_table["AUC Class"])

    rows, folds_detail = [], []
    for fold, train, valid in iter_positions(model_table):
        train_frame, valid_frame = model_table.iloc[train], model_table.iloc[valid]
        representation = _fit_representation(train_frame, kind, knot_strategy)

        X_train = model_ready(representation.transform(train_frame))
        X_valid = model_ready(representation.transform(valid_frame))
        y_train = y_all.iloc[train].reset_index(drop=True)
        y_valid = y_all.iloc[valid].to_numpy()

        keep = undersample_majority(y_train, seed=seed)
        X_train_bal = X_train.iloc[keep]
        y_train_bal = y_train.iloc[keep]

        model = XGBClassifier(**XGB_PARAMS, eval_metric="logloss").fit(
            X_train_bal, y_train_bal
        )
        prob = model.predict_proba(X_valid)[:, 1]
        pred = (prob >= 0.5).astype(int)

        matrix = confusion_matrix(y_valid, pred, labels=[0, 1])
        rows.append(
            {
                "representation": arm_label(kind, knot_strategy),
                "fold": fold,
                "n valid": len(valid),
                "valid positive rate": round(float(y_valid.mean()), 4),
                "train n before balancing": len(y_train),
                "train n after balancing": len(y_train_bal),
                "accuracy": accuracy_score(y_valid, pred),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "F1": f1_score(y_valid, pred, zero_division=0),
                "AUROC": (
                    roc_auc_score(y_valid, prob) if len(np.unique(y_valid)) > 1 else np.nan
                ),
            }
        )
        folds_detail.append(
            {
                "representation": arm_label(kind, knot_strategy),
                "fold": fold,
                "confusion_matrix": matrix,
            }
        )
    return pd.DataFrame(rows), folds_detail


def run_slow_release_classification(
    model_table: pd.DataFrame, kind: str, knot_strategy: str, seed: int = SEED
) -> pd.DataFrame:
    """The published <=20%-by-day-3 criterion, run as a separate experiment.

    Deliberately a near-copy of :func:`run_auc_classification`: same folds, same
    model, same training-fold-only undersampling. Only the target differs, so
    the two tasks' numbers are produced identically and any gap between them is
    the target, not the protocol.

    Formulations whose curve stops before day 3 are dropped rather than
    imputed -- the criterion is simply not observable for them.
    """
    from .targets import slow_release_class

    labels = slow_release_class(model_table)
    usable = model_table[labels.notna()].reset_index(drop=True)
    y_all = labels[labels.notna()].astype(int).reset_index(drop=True)
    dropped = len(model_table) - len(usable)

    rows = []
    for fold, train, valid in iter_positions(usable):
        train_frame, valid_frame = usable.iloc[train], usable.iloc[valid]
        representation = _fit_representation(train_frame, kind, knot_strategy)

        X_train = model_ready(representation.transform(train_frame))
        X_valid = model_ready(representation.transform(valid_frame))
        y_train = y_all.iloc[train].reset_index(drop=True)
        y_valid = y_all.iloc[valid].to_numpy()

        keep = undersample_majority(y_train, seed=seed)
        model = XGBClassifier(**XGB_PARAMS, eval_metric="logloss").fit(
            X_train.iloc[keep], y_train.iloc[keep]
        )
        prob = model.predict_proba(X_valid)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append(
            {
                "target": "Slow release (<=20% by day 3)",
                "representation": arm_label(kind, knot_strategy),
                "fold": fold,
                "n valid": len(valid),
                "n dropped (curve ends before day 3)": dropped,
                "valid positive rate": round(float(y_valid.mean()), 4),
                "accuracy": accuracy_score(y_valid, pred),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "F1": f1_score(y_valid, pred, zero_division=0),
                "AUROC": (
                    roc_auc_score(y_valid, prob)
                    if len(np.unique(y_valid)) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# --- Aggregation ---------------------------------------------------------

REGRESSION_METRICS = ["RMSE", "MAE", "Pearson_r", "R2"]
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "F1", "AUROC"]


def aggregate(per_fold: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and standard deviation across folds, one row per arm.

    Folds are averaged unweighted. Fold sizes here are close enough that
    weighting changes little, and the unweighted mean is what the per-fold
    standard deviation describes.
    """
    grouped = per_fold.groupby(by, sort=False)[metrics]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def paired_vs_crisp(
    per_fold: pd.DataFrame, metric: str = "RMSE", baseline: str = "crisp"
) -> pd.DataFrame:
    """Fold-by-fold difference from the crisp baseline on identical splits.

    Comparing arm means against their own across-fold standard deviations is
    the wrong test here. That sd is dominated by how hard each fold's held-out
    drugs are -- fold 0 is hard for *every* arm -- and it swamps differences
    between arms many times smaller. Because all arms are evaluated on exactly
    the same folds, the fold effect cancels in a paired difference.

    Reports the mean paired improvement (positive = the arm beat crisp) and how
    many of the five folds it won. With five paired observations, a consistent
    sign across all folds is the strongest claim available; a formal p-value on
    n=5 would be more precision than the design supports.
    """
    rows = []
    for (setting, target), part in per_fold.groupby(["time setting", "target"], sort=False):
        wide = part.pivot(index="fold", columns="representation", values=metric)
        if baseline not in wide:
            continue
        for arm in wide.columns:
            if arm == baseline:
                continue
            # RMSE and MAE are losses: crisp minus arm is the improvement.
            difference = wide[baseline] - wide[arm]
            rows.append(
                {
                    "time setting": setting,
                    "target": target,
                    "representation": arm,
                    f"mean Δ{metric} vs crisp": round(float(difference.mean()), 5),
                    "folds better than crisp": f"{int((difference > 0).sum())}/{len(difference)}",
                    "consistent across folds": bool(
                        (difference > 0).all() or (difference < 0).all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def format_comparison(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Collapse ``mean``/``std`` pairs into readable ``mean ± sd`` strings."""
    out = summary.copy()
    for metric in metrics:
        mean, std = out.pop(f"{metric} mean"), out.pop(f"{metric} std")
        out[metric] = [
            "nan" if pd.isna(m) else f"{m:.4f} ± {0.0 if pd.isna(s) else s:.4f}"
            for m, s in zip(mean, std)
        ]
    return out
