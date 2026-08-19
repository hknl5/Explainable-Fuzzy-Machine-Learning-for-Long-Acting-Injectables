"""Drug-group-aware cross-validation folds.

Every formulation of one drug stays in a single fold. A random row-level split
would place formulations of the same drug on both sides, leaking that drug's
physicochemical signature into the held-out set and inflating results.

Folds are assigned once and reused by every experiment so that the crisp,
fuzzy-only, and hybrid representations are compared on identical partitions.

Why not ``StratifiedGroupKFold``
-------------------------------
It was used first and produced folds that **differ between scikit-learn
versions for the same seed**. On this dataset, sklearn 1.2.2 yields validation
folds of 38-101 formulations while 1.8.0 yields 63-65 -- so the notebook kernel
(Anaconda, 1.8.0) and the shell interpreter (1.2.2) silently disagreed about
which drugs were held out.

That is fatal for a study whose whole claim rests on comparing representations
across identical splits. The greedy assignment below is deterministic, depends
on nothing but NumPy, and balances fold size and class rate by construction, so
the split is a fixed property of the dataset rather than of the environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AUC_CLASS_COL, GROUP_COL, ID_COL, N_SPLITS, SEED

FOLD_COL = "Fold"


#: Smallest number of minority-class formulations a validation fold may hold.
#: Below roughly this, per-fold F1 and recall stop carrying information.
MIN_MINORITY_PER_FOLD = 10


def choose_class_weight(
    model_table: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    candidates=None,
    floor: int = MIN_MINORITY_PER_FOLD,
) -> tuple[float, pd.DataFrame]:
    """Pick the class weight by a stated rule instead of a hardcoded constant.

    The rule: keep every validation fold at or above ``floor`` minority-class
    formulations, then among the weights that manage it take the one with the
    most equal fold sizes. If none reaches the floor, take whichever gets the
    minority count highest.

    Greedy assignment is discrete, so this landscape is jumpy rather than smooth
    and a single tuned constant would be brittle. Searching a stated criterion
    keeps the choice reproducible and lets it adapt if the data or ``n_splits``
    changes. Only fold composition is examined -- no model is fitted.

    Returns the chosen weight and the full scan for auditing.
    """
    from .targets import auc_class_binary

    if candidates is None:
        candidates = np.round(np.arange(0.0, 6.01, 0.25), 4)

    minority = 1 - auc_class_binary(model_table[AUC_CLASS_COL])

    rows = []
    for weight in candidates:
        folds = assign_folds(
            model_table, n_splits=n_splits, seed=seed, class_weight=float(weight)
        )
        sizes = folds.value_counts()
        counts = minority.groupby(folds).sum()
        rows.append(
            {
                "class_weight": float(weight),
                "min minority": int(counts.min()),
                "size spread": int(sizes.max() - sizes.min()),
            }
        )

    scan = pd.DataFrame(rows)
    eligible = scan[scan["min minority"] >= floor]
    pool = eligible if len(eligible) else scan
    best = pool.sort_values(
        ["size spread", "min minority"], ascending=[True, False]
    ).iloc[0]
    return float(best["class_weight"]), scan


def assign_folds(
    model_table: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    size_weight: float = 1.0,
    class_weight: float = 0.0,
) -> pd.Series:
    """Return the held-out fold index (0-based) for each formulation.

    Greedy balanced group partitioning. Drug groups are placed largest-first --
    the standard ordering for bin packing, because the big groups are the ones
    that constrain the result, and one drug here holds 49 of 321 formulations.
    Each group goes to whichever fold minimises a cost combining how far that
    fold would drift from the target size and from the overall AUC-class rate.

    Both objectives matter: size keeps per-fold metrics comparably precise, and
    class balance keeps the 1:2.9 imbalance from concentrating in one fold.

    ``class_weight`` sets their relative pull. Prefer
    :func:`choose_class_weight`, which selects it by a stated rule; the default
    of 0.0 is what that rule picks on this dataset, where the size objective
    alone already yields folds of 64/64/64/64/65 holding 10-21 minority-class
    formulations each.

    Only fold composition is consulted. No model is fitted and no predictive
    score is involved, so this cannot tune the split toward a result.

    ``seed`` only shuffles the order of equal-sized groups, which breaks ties
    reproducibly and lets repeated cross-validation draw alternative splits. The
    result does not otherwise depend on it.
    """
    from .targets import auc_class_binary

    positives = auc_class_binary(model_table[AUC_CLASS_COL]).to_numpy()
    groups = model_table[GROUP_COL].to_numpy()

    unique_groups = np.unique(groups)
    sizes = np.array([(groups == g).sum() for g in unique_groups])
    group_positives = np.array([positives[groups == g].sum() for g in unique_groups])

    target_size = len(model_table) / n_splits
    target_positives = max(positives.sum() / n_splits, 1e-9)

    # Largest first; a seeded shuffle only reorders groups of identical size.
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(len(unique_groups)), -sizes))

    fold_sizes = np.zeros(n_splits)
    fold_positives = np.zeros(n_splits)
    assignment = {}

    for index in order:
        size, positive = sizes[index], group_positives[index]

        # Least-loaded-first: send the group to whichever fold would end up
        # carrying the lightest combined load. Scoring the *resulting load*
        # rather than its distance from the target matters -- distance-to-target
        # would keep topping up a nearly-full fold instead of opening an empty
        # one, since a fold holding 1 group looks further from target than one
        # holding 60.
        cost = (
            size_weight * (fold_sizes + size) / target_size
            + class_weight * (fold_positives + positive) / target_positives
        )
        chosen = int(np.argmin(cost))

        assignment[unique_groups[index]] = chosen
        fold_sizes[chosen] += size
        fold_positives[chosen] += positive

    folds = pd.Series(groups, index=model_table.index).map(assignment)
    if folds.isna().any():
        raise ValueError("Some formulations were never assigned to a fold.")
    return folds.astype(int).rename(FOLD_COL)


def iter_folds(model_table: pd.DataFrame, fold_col: str = FOLD_COL):
    """Yield ``(fold_index, train_positions, valid_positions)`` as integer arrays."""
    folds = model_table[fold_col].to_numpy()
    for fold_index in sorted(np.unique(folds)):
        valid = np.flatnonzero(folds == fold_index)
        train = np.flatnonzero(folds != fold_index)
        yield int(fold_index), train, valid


def verify_no_group_leakage(model_table: pd.DataFrame, fold_col: str = FOLD_COL) -> None:
    """Raise if any drug group appears in both sides of any split."""
    for fold_index, train, valid in iter_folds(model_table, fold_col):
        train_groups = set(model_table.iloc[train][GROUP_COL])
        valid_groups = set(model_table.iloc[valid][GROUP_COL])
        shared = train_groups & valid_groups
        if shared:
            raise AssertionError(
                f"Fold {fold_index} leaks {len(shared)} drug group(s): {sorted(shared)[:5]}"
            )


def summarize_folds(model_table: pd.DataFrame, fold_col: str = FOLD_COL) -> pd.DataFrame:
    """Per-fold sizes, group counts, and class balance."""
    from .targets import auc_class_binary

    rows = []
    for fold_index, train, valid in iter_folds(model_table, fold_col):
        train_part = model_table.iloc[train]
        valid_part = model_table.iloc[valid]
        rows.append(
            {
                "fold": fold_index,
                "train formulations": len(train),
                "valid formulations": len(valid),
                "train drug groups": train_part[GROUP_COL].nunique(),
                "valid drug groups": valid_part[GROUP_COL].nunique(),
                "train AUC>0.5": round(auc_class_binary(train_part[AUC_CLASS_COL]).mean(), 4),
                "valid AUC>0.5": round(auc_class_binary(valid_part[AUC_CLASS_COL]).mean(), 4),
                "group overlap": len(
                    set(train_part[GROUP_COL]) & set(valid_part[GROUP_COL])
                ),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def undersample_majority(
    y: pd.Series, seed: int = SEED, indices: np.ndarray | None = None
) -> np.ndarray:
    """Random majority-class undersampling, for training folds only.

    The benchmark balances the AUC classes this way. Never apply this to a
    validation or test fold: it would change the evaluation distribution.

    Returns the retained positions, indexing into ``y``.
    """
    y = pd.Series(np.asarray(y))
    counts = y.value_counts()
    if len(counts) < 2:
        return np.arange(len(y))

    minority_size = int(counts.min())
    rng = np.random.default_rng(seed)

    keep = []
    for label in counts.index:
        positions = np.flatnonzero(y.to_numpy() == label)
        if len(positions) > minority_size:
            positions = rng.choice(positions, size=minority_size, replace=False)
        keep.append(positions)

    kept = np.sort(np.concatenate(keep))
    return indices[kept] if indices is not None else kept
