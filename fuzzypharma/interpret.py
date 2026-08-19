"""Fuzzy interpretation of the *predicted release value* (the output side).

This is the layer the project's motivating example is about. A crisp rule that
calls 69% release "Medium" and 70% "High" makes a one-point difference look
categorical. Fuzzy output memberships replace that step with a ramp: 69% is
Medium 0.55 / High 0.45, 70% is Medium 0.48 / High 0.52, and a formulator sees a
borderline profile described as borderline.

Distinct from :mod:`fuzzypharma.fuzzy`, which fuzzifies model *inputs*. Nothing
here touches the predictor: a release value goes in -- measured or predicted --
and a membership vector comes out. It can therefore be layered on top of any
regressor without retraining.

Provenance of the boundaries
----------------------------
Where the Low/Medium/High cut points come from is a scientific claim, and the
three possible origins are not interchangeable. :class:`ReleaseInterpreter`
records which one produced its knots in ``provenance``:

``design``
    Declared numbers (the default 1/3 and 2/3 of fractional release). They are a
    readable, symmetric partition of [0, 1] and nothing more. **They are not a
    pharmaceutical fact and no result here should be read as if they were.**
``data``
    Tertiles of the *training fold's* release distribution. Honest about the
    data, but "Medium" then means "middle third of what this corpus measured",
    which is a statement about the corpus and not about therapy.
``expert``
    Supplied by a pharmacist or supervisor for a stated drug, route and
    indication. No such definition was available when this module was written.

Until an ``expert`` definition exists, everything downstream is a *parameterised
framework* whose numbers move when the boundaries move. That dependence is
measured, not assumed away, by :func:`boundary_sensitivity_to_knots`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fuzzy import THREE_SET_LABELS, triangular_memberships

#: Declared fractional-release cut points for the ``design`` provenance.
#:
#: An even three-way split of the [0, 1] release axis. Chosen because it is
#: symmetric and inspectable, NOT because any source identifies 1/3 and 2/3 as
#: pharmaceutically meaningful. Flagged for supervisor/expert replacement.
DESIGN_KNOTS = (1.0 / 3.0, 0.5, 2.0 / 3.0)

#: Half-width of the perturbation used by the boundary-robustness test, in
#: fractional release. 0.01 is one percentage point -- the size of the step in
#: the project's own 69% / 70% example.
DEFAULT_DELTA = 0.01

PROVENANCE_NOTES = {
    "design": (
        "DECLARED by this study (even split of the [0,1] release axis). "
        "REQUIRES SUPERVISOR / PHARMACEUTICAL-EXPERT REVIEW before any "
        "clinical reading."
    ),
    "data": (
        "DERIVED from the training fold's release distribution (tertiles). "
        "Describes this corpus, not therapeutic adequacy."
    ),
    "expert": "SUPPLIED by a domain expert for a stated clinical context.",
}


@dataclass
class ReleaseInterpreter:
    """Map a release value to graded Low / Medium / High memberships.

    Parameters
    ----------
    knots:
        Three ascending cut points on the release axis. Membership is the same
        shoulder-triangle (Ruspini) family used for the inputs, so the three
        degrees always sum to 1 and read directly as shares.
    provenance:
        ``"design"``, ``"data"`` or ``"expert"`` -- see the module docstring.
        Carried through to every table this object produces so a reader never
        sees a membership without seeing where its boundaries came from.

    Notes
    -----
    ``fit_from_training`` is the only constructor that looks at data, and it
    must be handed training-fold rows only. The class is deliberately not a
    scikit-learn transformer: it is applied to a target/prediction vector, not
    to a design matrix, and keeping it out of the feature pipeline makes it hard
    to accidentally fit it on held-out release values.
    """

    knots: tuple[float, float, float] = DESIGN_KNOTS
    provenance: str = "design"
    labels: tuple[str, ...] = THREE_SET_LABELS

    def __post_init__(self) -> None:
        knots = np.asarray(self.knots, dtype=float)
        if knots.size != 3:
            raise ValueError("ReleaseInterpreter expects exactly three knots.")
        if not np.all(np.diff(knots) > 0):
            raise ValueError(f"Knots must be strictly increasing; got {self.knots}.")
        if self.provenance not in PROVENANCE_NOTES:
            raise ValueError(
                f"provenance must be one of {sorted(PROVENANCE_NOTES)}, "
                f"got {self.provenance!r}"
            )
        self.knots = tuple(float(k) for k in knots)

    # -- construction -----------------------------------------------------

    @classmethod
    def fit_from_training(
        cls, release: np.ndarray, quantiles: tuple[float, float, float] = (1 / 3, 0.5, 2 / 3)
    ) -> "ReleaseInterpreter":
        """Tertile knots from training-fold release values.

        Falls back to :data:`DESIGN_KNOTS` if the quantiles are not strictly
        increasing, which happens when a fold's release values pile up on one
        value. The fallback is reported rather than silent: the returned object
        still says ``provenance="design"``.
        """
        values = np.asarray(release, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return cls(DESIGN_KNOTS, "design")

        knots = np.quantile(values, quantiles)
        if not np.all(np.diff(knots) > 0):
            return cls(DESIGN_KNOTS, "design")
        return cls(tuple(knots), "data")

    @property
    def note(self) -> str:
        return PROVENANCE_NOTES[self.provenance]

    # -- application ------------------------------------------------------

    def memberships(self, release) -> pd.DataFrame:
        """Membership degrees for each release value; rows sum to 1."""
        values = np.asarray(release, dtype=float)
        degrees = triangular_memberships(values, np.asarray(self.knots))
        index = release.index if isinstance(release, pd.Series) else None
        return pd.DataFrame(
            degrees, columns=[f"{label}" for label in self.labels], index=index
        )

    def crisp_label(self, release) -> np.ndarray:
        """The hard label a crisp three-bin scheme would assign.

        Cut at the outer knots, which is the natural crisp counterpart: it is
        exactly where the fuzzy winner changes. This is the baseline the
        boundary-robustness comparison is against.
        """
        values = np.asarray(release, dtype=float)
        low, _, high = self.knots
        out = np.full(values.shape, self.labels[1], dtype=object)
        out[values < low] = self.labels[0]
        out[values >= high] = self.labels[2]
        return out

    def fuzzy_label(self, release) -> np.ndarray:
        """The highest-membership label (for agreement checks only).

        Reporting only this would throw away the entire point of the layer. It
        exists so the fuzzy and crisp schemes can be compared on equal terms.
        """
        degrees = self.memberships(release).to_numpy()
        return np.asarray(self.labels, dtype=object)[degrees.argmax(axis=1)]

    def describe(self, release_value: float, digits: int = 2) -> str:
        """One predicted value rendered the way a pharmacist would read it."""
        degrees = self.memberships(np.array([release_value])).iloc[0]
        parts = ", ".join(
            f"{label} {degrees[label] * 100:.0f}%"
            for label in self.labels
            if degrees[label] > 0
        )
        return f"Predicted release {release_value * 100:.{digits}f}% -> {parts}"

    def report(self) -> pd.DataFrame:
        """One row stating the knots and, explicitly, where they came from."""
        return pd.DataFrame(
            [
                {
                    "provenance": self.provenance,
                    "Low/Medium knot": round(self.knots[0], 4),
                    "Medium peak": round(self.knots[1], 4),
                    "Medium/High knot": round(self.knots[2], 4),
                    "status": self.note,
                }
            ]
        )


# --- Boundary robustness -------------------------------------------------


def boundary_robustness(
    release,
    interpreter: ReleaseInterpreter,
    delta: float = DEFAULT_DELTA,
) -> dict:
    """Quantify how violently each scheme reacts to a tiny change in release.

    The claim under test is the project's motivating one: a crisp scheme flips a
    formulation between categories over a difference too small to be
    pharmaceutically real, and a fuzzy scheme does not.

    Both schemes are given the same nudge -- every release value moved by
    ``±delta`` -- and asked how much their description of the formulation
    changed.

    ``crisp label flip rate``
        Share of values whose hard label is different at ``x - delta`` and
        ``x + delta``. A flip is total: the formulation is reassigned.
    ``mean fuzzy membership shift``
        Mean total-variation distance between the membership vectors at those
        same two points, i.e. ``0.5 * sum |m(x+d) - m(x-d)|``. On the same 0-1
        scale as the flip rate, so the two numbers are directly comparable: a
        flip costs 1.0, and this is what the graded scheme costs instead.
    ``max fuzzy membership shift``
        The worst case over all values. If even this stays well below 1, no
        formulation anywhere is being reassigned by the nudge.

    The crisp rate is not guaranteed to be large -- it depends on how many
    formulations sit within ``delta`` of a cut point. That is the honest form of
    the question: the fuzzy layer only helps the formulations that are actually
    near a boundary, and this reports how many those are.
    """
    values = np.asarray(release, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    low, high = values - delta, values + delta

    crisp_flip = interpreter.crisp_label(low) != interpreter.crisp_label(high)
    fuzzy_low = interpreter.memberships(low).to_numpy()
    fuzzy_high = interpreter.memberships(high).to_numpy()
    shift = 0.5 * np.abs(fuzzy_high - fuzzy_low).sum(axis=1)

    # A value is "near a boundary" if the perturbation band straddles a knot --
    # exactly the population the fuzzy layer is meant to serve.
    knots = np.asarray(interpreter.knots)[[0, 2]]
    near = np.any((low[:, None] <= knots) & (high[:, None] >= knots), axis=1)

    degrees = interpreter.memberships(values).to_numpy()
    graded = (degrees > 0.001).sum(axis=1) > 1

    return {
        "n values": int(values.size),
        "delta (fractional release)": delta,
        "n within delta of a boundary": int(near.sum()),
        "share within delta of a boundary": round(float(near.mean()), 4),
        "crisp label flip rate": round(float(crisp_flip.mean()), 4),
        "crisp flip rate | near boundary": (
            round(float(crisp_flip[near].mean()), 4) if near.any() else np.nan
        ),
        "mean fuzzy membership shift": round(float(shift.mean()), 4),
        "max fuzzy membership shift": round(float(shift.max()), 4),
        "fuzzy shift | near boundary": (
            round(float(shift[near].mean()), 4) if near.any() else np.nan
        ),
        "share with graded (multi-set) description": round(float(graded.mean()), 4),
        "knot provenance": interpreter.provenance,
    }


def boundary_sensitivity_to_knots(
    release,
    knot_sets: dict[str, tuple[float, float, float]],
    delta: float = DEFAULT_DELTA,
) -> pd.DataFrame:
    """Re-run :func:`boundary_robustness` under several boundary definitions.

    The Low/Medium/High cut points are not established pharmaceutical facts (see
    the module docstring), so any single-row robustness result is conditional on
    a choice this study is not entitled to make. Running the same measurement
    across plausible alternatives shows which conclusions survive the choice and
    which are artefacts of it.
    """
    rows = []
    for name, knots in knot_sets.items():
        interpreter = ReleaseInterpreter(knots, provenance="design")
        rows.append({"knot set": name, "knots": tuple(round(k, 4) for k in knots),
                     **boundary_robustness(release, interpreter, delta)})
    return pd.DataFrame(rows).drop(columns=["knot provenance"])


def interpretation_table(
    frame: pd.DataFrame,
    predicted_col: str,
    interpreter: ReleaseInterpreter,
    id_cols: list[str] | None = None,
    actual_col: str | None = None,
) -> pd.DataFrame:
    """Per-row predicted release alongside its fuzzy interpretation.

    This is the pharmacist-facing artifact: one line per formulation giving the
    number the model produced and the graded description of it, rather than a
    single hard category.
    """
    id_cols = id_cols or []
    out = frame[id_cols].copy() if id_cols else pd.DataFrame(index=frame.index)
    out["predicted release"] = frame[predicted_col].to_numpy()
    if actual_col is not None and actual_col in frame:
        out["measured release"] = frame[actual_col].to_numpy()

    degrees = interpreter.memberships(frame[predicted_col])
    for label in interpreter.labels:
        out[f"predicted is {label}"] = degrees[label].to_numpy().round(4)

    out["crisp label"] = interpreter.crisp_label(frame[predicted_col])
    out["graded"] = (degrees.to_numpy() > 0.001).sum(axis=1) > 1
    return out
