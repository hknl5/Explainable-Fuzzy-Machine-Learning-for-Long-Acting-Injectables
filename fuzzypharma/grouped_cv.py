"""Drug-grouped folds via ``GroupKFold`` and the measurement-level design matrix.

Two things live here that the earlier preprocessing phase did not need.

**Folds from exact SMILES.** :mod:`fuzzypharma.folds` assigns folds with a
greedy balanced partitioner. This module instead uses scikit-learn's
``GroupKFold`` keyed on the exact ``Drug SMILES`` string, which is what the
study protocol for these experiments specifies. Both keep every formulation of
a drug on one side of every split; they differ only in how evenly they balance
fold size and AUC-class rate. ``GroupKFold`` is not seeded and not shuffled, so
the split is a deterministic function of the group sizes.

**The measurement level.** Time setting (a) predicts release at each measured
time point, so its design matrix has one row per *measurement* rather than one
row per formulation. Fold labels are assigned once at the formulation level and
then propagated down, so both time settings are evaluated on identical drug
partitions and the comparison between them is not confounded by the split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ID_COL, METHOD_COL, N_SPLITS, RELEASE_COL, SMILES_COL, TIME_COL
from .config import GROUP_COL, STATIC_FEATURES
from .folds import FOLD_COL

#: Name of the transformed time column used by time setting (a).
LOG_TIME_COL = "Log1p Time"

#: Flag marking formulations whose curve contained a repeated time point.
DUPLICATE_TIME_FLAG = "Had Duplicate Time"

#: Per-curve min-max time, kept alongside (never replacing) raw ``Time``.
NORM_TIME_COL = "Normalized Time"


def assign_smiles_folds(
    model_table: pd.DataFrame, n_splits: int = N_SPLITS
) -> pd.Series:
    """Return the held-out fold index for each formulation, grouped by SMILES.

    ``GroupKFold`` sorts groups by size and fills the currently smallest fold,
    so the assignment depends only on the group sizes -- no random state is
    involved and reruns cannot drift.
    """
    if SMILES_COL not in model_table.columns:
        raise ValueError(
            f"{SMILES_COL!r} is required to split by drug; the source workbook "
            "must be present."
        )

    groups = model_table[SMILES_COL].to_numpy()
    folds = np.empty(len(model_table), dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold_index, (_, valid) in enumerate(
        splitter.split(model_table, groups=groups)
    ):
        folds[valid] = fold_index

    return pd.Series(folds, index=model_table.index, name=FOLD_COL)


def verify_no_smiles_leakage(model_table: pd.DataFrame, fold_col: str = FOLD_COL) -> None:
    """Raise if any SMILES string appears in more than one fold."""
    per_smiles = model_table.groupby(SMILES_COL)[fold_col].nunique()
    offenders = per_smiles[per_smiles > 1]
    if len(offenders):
        raise AssertionError(
            f"{len(offenders)} drug(s) span multiple folds: "
            f"{offenders.index.tolist()[:5]}"
        )


def summarize_smiles_folds(
    model_table: pd.DataFrame, fold_col: str = FOLD_COL
) -> pd.DataFrame:
    """Per-fold formulation counts, drug counts, and AUC-class rate."""
    from .targets import auc_class_binary

    positive = auc_class_binary(model_table["AUC Class"])
    rows = []
    for fold_index, part in model_table.groupby(fold_col):
        rows.append(
            {
                "fold": int(fold_index),
                "valid formulations": len(part),
                "valid drugs": part[SMILES_COL].nunique(),
                "train formulations": len(model_table) - len(part),
                "train drugs": model_table[SMILES_COL].nunique()
                - part[SMILES_COL].nunique(),
                "valid AUC>0.5 rate": round(positive.loc[part.index].mean(), 4),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def flag_duplicate_times(raw: pd.DataFrame) -> pd.Series:
    """Formulations whose raw curve recorded the same time more than once.

    The duplicate-time policy is to **average** the repeated release values when
    building the curve table (:func:`fuzzypharma.data.deduplicate_profile_points`)
    and to carry this flag alongside, so any result can be re-checked with the
    four affected formulations excluded. Formulations 52 and 305 record two
    *different* release values at one time and 136 and 148 record the same value
    twice; averaging is exact for the latter pair and a stated choice for the
    former.
    """
    duplicated = raw.groupby(ID_COL)[TIME_COL].apply(lambda s: s.duplicated().any())
    return duplicated.rename(DUPLICATE_TIME_FLAG)


def build_measurement_table(
    points: pd.DataFrame, model_table: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    """One row per measurement: static features, time, fold, and target.

    ``points`` is the deduplicated curve table (4,909 rows). Static features,
    formulation method, drug identity and the frozen fold come from
    ``model_table``, so a measurement inherits exactly the fold its formulation
    was assigned.

    Adds ``Log1p Time`` for time setting (a) and ``Normalized Time`` for
    per-curve shape work. ``Time`` itself is never overwritten.
    """
    carried = [
        ID_COL,
        GROUP_COL,
        SMILES_COL,
        METHOD_COL,
        *STATIC_FEATURES,
        "Profile Duration",
        FOLD_COL,
    ]

    table = points[[ID_COL, TIME_COL, RELEASE_COL]].merge(
        model_table[carried], on=ID_COL, how="left", validate="many_to_one"
    )
    if table[FOLD_COL].isna().any():
        raise ValueError("Some measurements did not inherit a fold.")

    table[LOG_TIME_COL] = np.log1p(table[TIME_COL])

    # Per-curve min-max time. Duration is already known per formulation, so this
    # only needs each curve's own start time.
    start = table.groupby(ID_COL)[TIME_COL].transform("min")
    span = table.groupby(ID_COL)[TIME_COL].transform("max") - start
    if (span <= 0).any():
        raise ValueError("A profile has zero duration and cannot be normalized.")
    table[NORM_TIME_COL] = (table[TIME_COL] - start) / span

    flags = flag_duplicate_times(raw)
    table[DUPLICATE_TIME_FLAG] = table[ID_COL].map(flags).fillna(False).astype(bool)

    table[FOLD_COL] = table[FOLD_COL].astype(int)
    return table


def iter_positions(table: pd.DataFrame, fold_col: str = FOLD_COL):
    """Yield ``(fold, train_positions, valid_positions)`` for any fold-labelled table."""
    folds = table[fold_col].to_numpy()
    for fold_index in sorted(np.unique(folds)):
        yield (
            int(fold_index),
            np.flatnonzero(folds != fold_index),
            np.flatnonzero(folds == fold_index),
        )
