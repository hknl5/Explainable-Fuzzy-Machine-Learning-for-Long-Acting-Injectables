"""Construction of the three prediction targets described in the proposal.

Task 1  early release at 24/48/72 h  -> regression on absolute time (days)
Task 2  normalized-AUC class          -> binary classification
Task 3  complete release curve        -> fixed-length curve on normalized time
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AUC_CLASS_COL,
    AUC_CLASS_HIGH,
    AUC_CLASS_LOW,
    AUC_COL,
    AUC_THRESHOLD,
    CURVE_GRID_POINTS,
    EARLY_TARGETS,
    GROUP_COL,
    ID_COL,
    RELEASE_COL,
    TIME_COL,
)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """``np.trapezoid`` on NumPy >= 2, ``np.trapz`` before it."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def normalized_time(time: np.ndarray) -> np.ndarray:
    """Min-max scale a profile's time axis to [0, 1] (benchmark Eq. 1)."""
    t_min, t_max = time.min(), time.max()
    span = t_max - t_min
    if span <= 0:
        raise ValueError("Profile has zero duration and cannot be normalized.")
    return (time - t_min) / span


def interpolate_at(time: np.ndarray, release: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linear interpolation (benchmark Eq. 2), refusing to extrapolate.

    Targets outside the profile's observed range return NaN rather than being
    clamped to the nearest endpoint, which would invent measurements.
    """
    out = np.interp(targets, time, release)
    outside = (targets < time.min()) | (targets > time.max())
    out[outside] = np.nan
    return out


def curve_grid(n_points: int = CURVE_GRID_POINTS) -> np.ndarray:
    """The common normalized-time grid shared by every resampled profile."""
    return np.linspace(0.0, 1.0, n_points)


def build_profile_targets(points: pd.DataFrame, n_points: int = CURVE_GRID_POINTS):
    """Derive per-profile targets and the fixed-length curve matrix.

    ``points`` must already be deduplicated on (formulation, time).

    Returns
    -------
    targets : DataFrame, one row per formulation
    curves  : DataFrame, one row per formulation, ``n_points`` resampled columns
    """
    grid = curve_grid(n_points)
    early_names = list(EARLY_TARGETS)
    early_days = np.array([EARLY_TARGETS[n] for n in early_names], dtype=float)

    target_rows, curve_rows, index = [], [], []

    for formulation_id, group in points.groupby(ID_COL, sort=True):
        group = group.sort_values(TIME_COL)
        time = group[TIME_COL].to_numpy(dtype=float)
        release = group[RELEASE_COL].to_numpy(dtype=float)

        if len(np.unique(time)) != len(time):
            raise ValueError(
                f"Formulation {formulation_id} still has duplicate times; "
                "run deduplicate_profile_points first."
            )

        t_norm = normalized_time(time)
        auc = _trapezoid(release, t_norm)
        early = interpolate_at(time, release, early_days)

        row = {
            ID_COL: int(formulation_id),
            GROUP_COL: int(group[GROUP_COL].iloc[0]),
            AUC_COL: auc,
            AUC_CLASS_COL: AUC_CLASS_HIGH if auc > AUC_THRESHOLD else AUC_CLASS_LOW,
            "Profile Duration": float(time.max() - time.min()),
            "N Time Points": int(len(time)),
            "Max Observed Release": float(release.max()),
        }
        row.update(dict(zip(early_names, early)))
        target_rows.append(row)

        # Resampling happens on normalized time, so the grid is always covered.
        curve_rows.append(np.interp(grid, t_norm, release))
        index.append(int(formulation_id))

    targets = pd.DataFrame(target_rows)
    curves = pd.DataFrame(
        curve_rows,
        index=pd.Index(index, name=ID_COL),
        columns=[f"t{value:.4f}" for value in grid],
    )
    return targets, curves


def build_model_table(formulations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join static inputs to derived targets -> the formulation-level table."""
    table = formulations.merge(targets.drop(columns=[GROUP_COL]), on=ID_COL, how="inner")
    if len(table) != len(formulations):
        raise ValueError("Join dropped formulations; target coverage is incomplete.")
    return table


def auc_class_binary(labels: pd.Series) -> pd.Series:
    """Map the AUC class strings to 1 (``> 0.5``) / 0 (``<= 0.5``)."""
    return (labels == AUC_CLASS_HIGH).astype(int)
