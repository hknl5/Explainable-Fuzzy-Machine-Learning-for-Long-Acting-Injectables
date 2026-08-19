"""End-to-end assembly of the prepared dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CURVE_GRID_POINTS, ID_COL, N_SPLITS, PREPARED_DIR, SEED
from .data import QualityReport, assign_drug_groups, attach_metadata, audit
from .data import build_formulation_table, deduplicate_profile_points, load_metadata
from .data import load_raw, verify_group_equivalence
from .folds import FOLD_COL, assign_folds, summarize_folds, verify_no_group_leakage
from .targets import build_model_table, build_profile_targets


@dataclass
class Dataset:
    """Every table produced by the preprocessing phase."""

    raw: pd.DataFrame
    """4,913 measurement rows, exactly as read."""

    points: pd.DataFrame
    """Measurement rows after averaging same-time duplicates."""

    formulations: pd.DataFrame
    """321 rows: static features and drug group."""

    model_table: pd.DataFrame
    """321 rows: static features, all targets, drug group, frozen fold."""

    curves: pd.DataFrame
    """321 x grid: release resampled onto a shared normalized-time axis."""

    quality: QualityReport
    fold_summary: pd.DataFrame

    def export(self, directory: Path | str = PREPARED_DIR) -> list[Path]:
        """Write the prepared tables to CSV and return the paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        written = []
        artifacts = {
            "formulation_model_table.csv": self.model_table,
            "release_curves_normalized.csv": self.curves.reset_index(),
            "release_profiles_deduplicated.csv": self.points,
            "fold_summary.csv": self.fold_summary.reset_index(),
            "quality_report.csv": self.quality.to_frame(),
        }
        for name, frame in artifacts.items():
            path = directory / name
            frame.to_csv(path, index=False)
            written.append(path)
        return written


def compare_release_policies() -> pd.DataFrame:
    """Quantify what clipping release to [0, 1] would change.

    108 measurements exceed 1.0. Clipping them is defensible as a *documented
    sensitivity analysis*, never as a silent repair, so this reports the effect
    on every derived target rather than leaving the question open.

    Monotonic correction is deliberately not offered: two thirds of profiles
    contain at least one measured decrease, so enforcing monotonicity would
    rewrite a large share of real measurements rather than fix a few outliers.
    """
    preserved = build_dataset(clip_release=False)
    clipped = build_dataset(clip_release=True)

    targets = ["Release_24h", "Release_48h", "Release_72h", "AUC"]
    rows = []
    for column in targets:
        left = preserved.model_table[column]
        right = clipped.model_table[column]
        rows.append(
            {
                "target": column,
                "preserved mean": left.mean(),
                "clipped mean": right.mean(),
                "max abs change": (left - right).abs().max(),
                "formulations changed": int((left - right).abs().gt(1e-12).sum()),
            }
        )

    flipped = (
        preserved.model_table["AUC Class"] != clipped.model_table["AUC Class"]
    ).sum()
    rows.append(
        {
            "target": "AUC Class",
            "preserved mean": float("nan"),
            "clipped mean": float("nan"),
            "max abs change": float("nan"),
            "formulations changed": int(flipped),
        }
    )
    return pd.DataFrame(rows).set_index("target")


def build_dataset(
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    curve_points: int = CURVE_GRID_POINTS,
    clip_release: bool | None = None,
) -> Dataset:
    """Run the full preprocessing phase and validate it.

    Loads and validates the workbook, separates the measurement and formulation
    levels, derives the three targets, and assigns drug-group folds. Raises if
    any fold leaks a drug group.
    """
    raw = load_raw()

    metadata = load_metadata()
    if metadata is not None:
        raw = attach_metadata(raw, metadata)

    df = assign_drug_groups(raw)
    formulations = build_formulation_table(df)
    points = deduplicate_profile_points(df, clip_release=clip_release)
    quality = audit(df, points)

    if metadata is not None:
        agreement = verify_group_equivalence(df)
        if agreement != 1.0:
            raise ValueError(
                f"SMILES grouping disagrees with the descriptor proxy (ARI {agreement:.4f})."
            )
        quality.notes.append(
            "Drug identity from canonical SMILES; identical to the descriptor proxy (ARI 1.0)."
        )
        quality.notes.append(
            "Formulation method recovered from the source dataset and one-hot encoded."
        )
    else:
        quality.notes.append(
            "Source dataset absent: drug identity falls back to the descriptor proxy, "
            "and formulation method is unavailable."
        )

    targets, curves = build_profile_targets(points, n_points=curve_points)
    model_table = build_model_table(formulations, targets)

    model_table[FOLD_COL] = assign_folds(model_table, n_splits=n_splits, seed=seed).values
    verify_no_group_leakage(model_table)

    curves = curves.loc[model_table[ID_COL].to_numpy()]

    return Dataset(
        raw=df,
        points=points,
        formulations=formulations,
        model_table=model_table,
        curves=curves,
        quality=quality,
        fold_summary=summarize_folds(model_table),
    )
