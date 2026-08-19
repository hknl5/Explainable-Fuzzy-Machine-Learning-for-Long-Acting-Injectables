"""Loading, validation, and reshaping of the raw workbook.

The workbook stores one row per *measurement*, not per formulation. These
helpers separate the two observational levels and audit the quirks that matter
downstream, without silently repairing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CLIP_RELEASE,
    DATA_PATH,
    DRUG_COL,
    DRUG_DESCRIPTOR_COLS,
    EXPECTED_COLUMNS,
    GROUP_COL,
    ID_COL,
    METADATA_COLS,
    METADATA_PATH,
    METADATA_SHEET,
    METHOD_COL,
    RELEASE_BOUNDS,
    RELEASE_COL,
    SHEET_NAME,
    SMILES_COL,
    STATIC_FEATURES,
    TIME_COL,
)


@dataclass
class QualityReport:
    """Audit flags. These are observations for review, not automatic errors."""

    n_rows: int
    n_profiles: int
    n_drug_groups: int
    missing_cells: int
    non_finite_cells: int
    exact_duplicate_rows: int
    profiles_with_duplicate_times: int
    profiles_unordered_in_workbook: int
    profiles_starting_after_zero: int
    profiles_not_covering_72h: int
    rows_outside_unit_interval: int
    profiles_outside_unit_interval: int
    profiles_with_release_decrease: int
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            (k, v)
            for k, v in self.__dict__.items()
            if k != "notes"
        ]
        return pd.DataFrame(rows, columns=["check", "value"])


def load_raw(path: Path | str = DATA_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Read the workbook and validate its schema and numeric parsability."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_block = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    unparsable = numeric_block.isna() & df[EXPECTED_COLUMNS].notna()
    if unparsable.any().any():
        bad = unparsable.any()[lambda s: s].index.tolist()
        raise ValueError(f"Non-numeric values found in numeric columns: {bad}")
    df[EXPECTED_COLUMNS] = numeric_block

    return df


def load_metadata(path: Path | str = METADATA_PATH) -> pd.DataFrame | None:
    """Read drug identity and formulation method from the source dataset.

    Returns ``None`` if the file is absent, so the pipeline still runs on the
    processed workbook alone.
    """
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_excel(path, sheet_name=METADATA_SHEET)
    missing = sorted(set([ID_COL, *METADATA_COLS]) - set(frame.columns))
    if missing:
        raise ValueError(f"Source dataset is missing columns: {missing}")
    return frame


def attach_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join drug name, SMILES, formulation method and DOI onto the workbook.

    The processed workbook is a numeric projection of the source file, so the
    join is positional and must be verified rather than assumed. This checks
    that both frames agree row for row on ``Formulation Index`` and on every
    shared numeric column before attaching anything.
    """
    if len(metadata) != len(df):
        raise ValueError(
            f"Row count mismatch: workbook {len(df)}, source {len(metadata)}"
        )

    if not (metadata[ID_COL].to_numpy() == df[ID_COL].to_numpy()).all():
        raise ValueError("Formulation Index does not align row for row.")

    # The source spells this column "Encapuslation"; compare on values, not names.
    renamed = metadata.rename(
        columns={"Drug Encapuslation Efficiency": "Drug Encapsulation Efficiency"}
    )
    shared = [c for c in renamed.columns if c in df.columns and c != ID_COL]
    for column in shared:
        left = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(renamed[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True):
            raise ValueError(f"Source and workbook disagree on {column!r}.")

    df = df.copy()
    for column in METADATA_COLS:
        df[column] = metadata[column].to_numpy()
    return df


def assign_drug_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a drug-group id, preferring canonical SMILES over descriptors.

    When the source dataset is present, drug identity comes from ``Drug SMILES``.
    Otherwise the exact ``(Drug MW, Drug TPSA, Drug LogP)`` triple stands in.

    Both give the same 88 groups on this dataset -- adjusted Rand index 1.0, with
    no distinct drugs merged and no single drug split -- so the descriptor proxy
    turned out to be exactly right. It is kept as the fallback and checked
    against SMILES by :func:`verify_group_equivalence`.

    Note that 89 drug *names* map to 88 SMILES: ``dexamethasone`` and
    ``β-methasone`` are recorded with an identical SMILES string, which carries
    no stereochemistry at the position that would distinguish them. Grouping on
    SMILES therefore yields the benchmark's 88.
    """
    df = df.copy()
    if SMILES_COL in df.columns:
        keys = df[SMILES_COL]
    else:
        keys = df[DRUG_DESCRIPTOR_COLS].apply(tuple, axis=1)

    # Rank by key so ids are stable across runs regardless of row order.
    ordered = {key: i for i, key in enumerate(sorted(set(keys)))}
    df[GROUP_COL] = keys.map(ordered).astype(int)
    return df


def verify_group_equivalence(df: pd.DataFrame) -> float:
    """Agreement between SMILES grouping and the descriptor proxy.

    Returns the adjusted Rand index; 1.0 means the two partitions are identical.
    """
    if SMILES_COL not in df.columns:
        return float("nan")

    formulations = df.drop_duplicates(ID_COL)
    pairs = pd.DataFrame(
        {
            "proxy": formulations[DRUG_DESCRIPTOR_COLS].apply(tuple, axis=1).astype(str),
            "smiles": formulations[SMILES_COL].to_numpy(),
        }
    )

    # Identical partitions: every proxy group maps to exactly one SMILES, and
    # every SMILES to exactly one proxy group.
    merged = pairs.groupby("proxy")["smiles"].nunique()
    split = pairs.groupby("smiles")["proxy"].nunique()
    if (merged == 1).all() and (split == 1).all():
        return 1.0

    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(pairs["proxy"], pairs["smiles"]))


def build_formulation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per formulation, carrying the static features.

    Raises if a static feature varies within a formulation, which would mean the
    two observational levels cannot be separated cleanly.
    """
    varying = df.groupby(ID_COL)[STATIC_FEATURES].nunique(dropna=False)
    offenders = varying.gt(1).sum()
    if offenders.any():
        detail = offenders[offenders > 0].to_dict()
        raise ValueError(f"Static features vary within a formulation: {detail}")

    present = [c for c in METADATA_COLS if c in df.columns]
    cols = [ID_COL, GROUP_COL, *present, *STATIC_FEATURES]
    return (
        df.sort_values([ID_COL, TIME_COL])
        .drop_duplicates(subset=ID_COL, keep="first")[cols]
        .reset_index(drop=True)
    )


def deduplicate_profile_points(
    df: pd.DataFrame, clip_release: bool | None = None
) -> pd.DataFrame:
    """Average release values sharing a formulation and time.

    Four profiles record the same time twice. Interpolation and trapezoidal AUC
    both require a strictly increasing time axis, so same-time measurements are
    averaged here. The raw frame is left untouched.

    ``clip_release`` overrides the ``CLIP_RELEASE`` default. Clipping to [0, 1]
    is a sensitivity analysis, not the primary analysis -- see
    :func:`fuzzypharma.pipeline.compare_release_policies`.
    """
    if clip_release is None:
        clip_release = CLIP_RELEASE

    points = (
        df.groupby([ID_COL, TIME_COL], as_index=False, sort=True)
        .agg({RELEASE_COL: "mean", GROUP_COL: "first"})
        .sort_values([ID_COL, TIME_COL])
        .reset_index(drop=True)
    )
    if clip_release:
        points[RELEASE_COL] = points[RELEASE_COL].clip(*RELEASE_BOUNDS)
    return points


def audit(df: pd.DataFrame, points: pd.DataFrame) -> QualityReport:
    """Compute the data-quality flags described in ``PREPROCESSING.md``."""
    lo, hi = RELEASE_BOUNDS
    by_id = df.groupby(ID_COL)

    max_time = by_id[TIME_COL].max()
    min_time = by_id[TIME_COL].min()
    out_of_range = df[RELEASE_COL].lt(lo) | df[RELEASE_COL].gt(hi)

    sorted_df = df.sort_values([ID_COL, TIME_COL])
    decreases = (
        sorted_df.groupby(ID_COL)[RELEASE_COL].apply(lambda s: s.diff().lt(0).any()).sum()
    )
    unordered = (
        df.groupby(ID_COL, sort=False)[TIME_COL].apply(lambda s: s.diff().lt(0).any()).sum()
    )

    numeric = df[EXPECTED_COLUMNS]
    report = QualityReport(
        n_rows=len(df),
        n_profiles=df[ID_COL].nunique(),
        n_drug_groups=df[GROUP_COL].nunique(),
        missing_cells=int(df.isna().sum().sum()),
        non_finite_cells=int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()),
        exact_duplicate_rows=int(df.duplicated().sum()),
        profiles_with_duplicate_times=int(
            by_id[TIME_COL].apply(lambda s: s.duplicated().any()).sum()
        ),
        profiles_unordered_in_workbook=int(unordered),
        profiles_starting_after_zero=int(min_time.gt(0).sum()),
        profiles_not_covering_72h=int(max_time.lt(3.0).sum()),
        rows_outside_unit_interval=int(out_of_range.sum()),
        profiles_outside_unit_interval=int(df.loc[out_of_range, ID_COL].nunique()),
        profiles_with_release_decrease=int(decreases),
    )
    report.notes = [
        "Release values above 1.0 and measured decreases are preserved, not repaired.",
        "Same-time duplicates are averaged only in the profile-points table.",
        f"Deduplication removed {len(df) - len(points)} rows for target construction.",
    ]
    return report


def load_prepared() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, QualityReport]:
    """Convenience entry point returning every intermediate table at once."""
    raw = load_raw()
    df = assign_drug_groups(raw)
    formulations = build_formulation_table(df)
    points = deduplicate_profile_points(df)
    return df, formulations, points, audit(df, points)
