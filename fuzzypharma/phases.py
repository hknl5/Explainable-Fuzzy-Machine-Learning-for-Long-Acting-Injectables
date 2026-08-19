"""Early / Middle / Late release phases, and the reason rules need them.

Cumulative release only goes up. A rule learner given raw time will therefore
spend its splits on time and return statements like *IF Time is large THEN
release is high* -- true, high-coverage, and of no use to a formulator, because
it describes the shape of a release curve rather than anything about the
formulation.

The fix used here is to hold time roughly fixed and ask the formulation question
inside that window: within measurements taken in the same phase, does a
formulation characteristic still separate the release values? Time is then not
an input to the rule learner at all (see
:func:`fuzzypharma.rules.phase_tree_rules`), so it cannot win a split, and any
rule that comes out is about formulation by construction rather than by a
post-hoc correction.

Where the boundaries come from
------------------------------
``EARLY_END = 3`` days
    Has external support: the early-release criterion used in the PLGA
    literature this project benchmarks against is "<= 20% released within 3
    days" (Pharmaceutics 19(5):767, 2026), so 3 days is a boundary that
    pharmaceutical work on this dataset already treats as meaningful.

``MIDDLE_END = 14`` days
    **Has no such support and is not a pharmaceutical claim.** It was chosen
    because it splits this corpus into three comparably sized parts (27% / 38% /
    34% of measurements; it sits at the 66th percentile of measured times) and
    leaves every phase with enough formulations to fit and test a tree. That is
    a statement about the dataset, not about release biology.

Both are module constants so they can be overridden, and every phase-based
result records which boundaries produced it. **Selecting the final boundaries is
a supervisor / pharmaceutical-expert decision**, and
:func:`phase_boundary_alternatives` exists so the sensitivity of any conclusion
to that decision can be measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TIME_COL

#: Upper bound of the Early phase, in days (inclusive).
EARLY_END = 3.0

#: Upper bound of the Middle phase, in days (inclusive).
MIDDLE_END = 14.0

PHASE_COL = "Release Phase"
EARLY, MIDDLE, LATE = "Early", "Middle", "Late"
PHASES = (EARLY, MIDDLE, LATE)

PHASE_PROVENANCE = {
    EARLY: (
        "0-3 d. Boundary supported by the published <=20%-within-3-days early "
        "release criterion for PLGA microspheres (Pharmaceutics 19(5):767, 2026)."
    ),
    MIDDLE: (
        "3-14 d. Upper boundary is DATA-DERIVED (66th percentile of measured "
        "times; gives three comparably sized phases). NOT a pharmaceutical "
        "threshold -- REQUIRES SUPERVISOR / EXPERT SELECTION."
    ),
    LATE: (
        ">14 d. Defined as the remainder. Inherits the same unvalidated 14-day "
        "boundary -- REQUIRES SUPERVISOR / EXPERT SELECTION."
    ),
}

#: Alternative boundary pairs for the sensitivity check. None is endorsed; they
#: span the range a reviewer might reasonably propose.
ALTERNATIVE_BOUNDS = {
    "default (3 / 14 d)": (3.0, 14.0),
    "tighter early (1 / 7 d)": (1.0, 7.0),
    "wider (7 / 28 d)": (7.0, 28.0),
    "corpus tertiles": None,  # filled in from the data by phase_boundary_alternatives
}


def assign_phase(
    time, early_end: float = EARLY_END, middle_end: float = MIDDLE_END
) -> np.ndarray:
    """Label each time point Early / Middle / Late.

    Boundaries are inclusive on the left phase: a measurement at exactly 3.0
    days is Early. Phases are assigned from the *absolute* recorded time, not
    from per-curve normalized time, because "3 days after injection" is the
    quantity a formulator reasons about; a curve-relative fraction would mean a
    different number of days for every formulation.
    """
    values = np.asarray(time, dtype=float)
    out = np.full(values.shape, LATE, dtype=object)
    out[values <= middle_end] = MIDDLE
    out[values <= early_end] = EARLY
    return out


def add_phase(
    measurements: pd.DataFrame,
    early_end: float = EARLY_END,
    middle_end: float = MIDDLE_END,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Return ``measurements`` with a :data:`PHASE_COL` column added."""
    out = measurements.copy()
    out[PHASE_COL] = assign_phase(out[time_col], early_end, middle_end)
    return out


def phase_summary(measurements: pd.DataFrame, release_col: str = "Release") -> pd.DataFrame:
    """Rows, formulations, drugs and release level in each phase.

    Reported before any rule is extracted: a phase holding few formulations or
    few drugs cannot support a grouped-CV rule test, and the reader needs to see
    that rather than infer it from a rule table that silently came out empty.
    """
    from .config import ID_COL, SMILES_COL

    rows = []
    for phase in PHASES:
        part = measurements[measurements[PHASE_COL] == phase]
        if part.empty:
            rows.append({"phase": phase, "n measurements": 0})
            continue
        rows.append(
            {
                "phase": phase,
                "n measurements": len(part),
                "share of measurements": round(len(part) / len(measurements), 4),
                "n formulations": int(part[ID_COL].nunique()),
                "n drugs": int(part[SMILES_COL].nunique())
                if SMILES_COL in part
                else np.nan,
                "mean release": round(float(part[release_col].mean()), 4),
                "sd release": round(float(part[release_col].std()), 4),
                "time range (d)": (
                    f"{part[TIME_COL].min():.2f}-{part[TIME_COL].max():.2f}"
                ),
                "provenance": PHASE_PROVENANCE[phase],
            }
        )
    return pd.DataFrame(rows)


def phase_boundary_alternatives(measurements: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Boundary pairs for the sensitivity check, including corpus tertiles."""
    alternatives = {k: v for k, v in ALTERNATIVE_BOUNDS.items() if v is not None}
    tertiles = np.quantile(measurements[TIME_COL].to_numpy(dtype=float), [1 / 3, 2 / 3])
    alternatives["corpus tertiles"] = (float(tertiles[0]), float(tertiles[1]))
    return alternatives
