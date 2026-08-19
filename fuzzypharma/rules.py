"""Layer 3: IF-THEN rules extracted from a trained predictor, and their validation.

Adding fuzzy features to XGBoost does not by itself produce rules. This module
is the explicit rule-generation step the proposal requires, in two forms:

* :func:`rulefit_rules` -- ``imodels.RuleFitRegressor`` fitted as a *surrogate*:
  its target is the trained XGBoost model's own predictions, so the rules
  describe the predictor rather than re-fitting the data independently.
* :func:`surrogate_tree_rules` -- a depth-limited ``DecisionTreeRegressor``
  fitted to the same target, giving a small set of mutually exclusive paths.

Both are fitted on the **training fold only**. A rule is then re-evaluated on
the held-out fold, whose drugs the predictor has never seen.

What "validated" means here
---------------------------
A rule is a claim that a subpopulation has a systematically different outcome.
That claim is tested on held-out data by three conditions, all of which must
hold (:func:`validate_rules`):

1. **Coverage.** The rule fires on at least ``MIN_COVERAGE`` of held-out rows.
   A rule matching four measurements describes noise.
2. **Direction agreement.** The sign of the effect on the held-out fold matches
   the sign on the training fold. A rule that reverses has not generalized.
3. **Interval excluding zero.** A cluster bootstrap over *drug groups* puts a
   95% interval on the held-out effect that does not contain 0. Clustering is
   essential: held-out rows are many measurements from few drugs, so a
   row-level bootstrap would treat 1,000 correlated points as 1,000
   independent ones and declare almost everything significant.

Rules that fail any condition are reported separately, not discarded silently.

These are empirical associations in 321 literature-sourced formulations. They
are not evidence of a release mechanism and carry no causal claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from .config import SEED

#: Smallest share of held-out rows a rule must fire on to be testable.
MIN_COVERAGE = 0.05

#: Bootstrap resamples used for the held-out effect interval.
N_BOOTSTRAP = 2000

#: Folds in which a condition pattern must pass before it is called stable.
MIN_FOLDS_VALIDATED = 3

_TERM = re.compile(r"^(?P<feature>.+?)\s*(?P<op><=|>=|<|>|==)\s*(?P<value>\S+)$")


@dataclass(frozen=True)
class Condition:
    """One ``feature op threshold`` test."""

    feature: str
    op: str
    threshold: float

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.feature].to_numpy(dtype=float)
        if self.op == "<=":
            return values <= self.threshold
        if self.op == "<":
            return values < self.threshold
        if self.op == ">=":
            return values >= self.threshold
        if self.op == ">":
            return values > self.threshold
        raise ValueError(f"Unsupported operator: {self.op!r}")

    @property
    def signature(self) -> tuple[str, str]:
        """Feature and direction, without the fold-specific threshold.

        Each fold learns its own thresholds, so two folds never produce
        byte-identical rules. Comparing on direction alone is what makes
        cross-fold stability measurable at all.
        """
        return (self.feature, "high" if self.op in (">", ">=") else "low")


def parse_rule(rule: str) -> list[Condition]:
    """Parse an ``imodels`` rule string into conditions.

    Terms are joined by ``" and "``. Feature names here contain spaces (and
    ``LA/GA`` a slash) but never a comparison character, so splitting each term
    at its first operator is unambiguous.
    """
    conditions = []
    for term in rule.split(" and "):
        matched = _TERM.match(term.strip())
        if not matched:
            raise ValueError(f"Could not parse rule term: {term!r}")
        conditions.append(
            Condition(
                matched.group("feature").strip(),
                matched.group("op"),
                float(matched.group("value")),
            )
        )
    return simplify(conditions)


def simplify(conditions: list[Condition]) -> list[Condition]:
    """Collapse repeated tests on one feature to the binding one.

    A boosted tree path often revisits a feature, so RuleFit emits rules like
    ``Log1p Time <= 1.47 and Log1p Time <= 0.0397 and Log1p Time <= 0.00217``.
    Only the tightest bound has any effect, so keeping just it leaves the mask
    bit-identical while making the rule readable.

    Conditions are grouped by ``(feature, operator)`` rather than by direction,
    so the collapse is exact by construction: two tests with the same operator
    differ only in threshold, and the tighter one implies the looser.
    """
    tightest: dict[tuple[str, str], Condition] = {}
    for condition in conditions:
        key = (condition.feature, condition.op)
        held = tightest.get(key)
        if held is None:
            tightest[key] = condition
        elif condition.op in ("<=", "<"):
            tightest[key] = min(held, condition, key=lambda c: c.threshold)
        else:
            tightest[key] = max(held, condition, key=lambda c: c.threshold)
    return list(tightest.values())


def is_time_only(conditions: list[Condition]) -> bool:
    """True if a rule tests nothing but the time axis.

    Such rules are correct and often the highest-coverage ones extracted -- late
    measurements do show more release -- but they restate the shape of a release
    curve rather than saying anything about a formulation, so they are reported
    apart from the rules a formulator could act on.
    """
    return all(condition.feature == "Log1p Time" for condition in conditions)


def split_time_conditions(
    conditions: list[Condition],
) -> tuple[list[Condition], list[Condition]]:
    """Separate a rule's time conditions from its formulation conditions."""
    time = [c for c in conditions if c.feature == "Log1p Time"]
    formulation = [c for c in conditions if c.feature != "Log1p Time"]
    return time, formulation


def rule_mask(conditions: list[Condition], X: pd.DataFrame) -> np.ndarray:
    """Rows satisfying every condition (conjunction)."""
    mask = np.ones(len(X), dtype=bool)
    for condition in conditions:
        mask &= condition.mask(X)
    return mask


# --- Human-readable rendering -------------------------------------------


def humanize(condition: Condition, scaler_lookup: dict[str, tuple[float, float]]) -> str:
    """Render one condition in the units a formulator would recognize.

    Two rewrites happen. Crisp features were standardized inside the fold, so a
    threshold of ``-0.88`` is meaningless on its own; the fold's own mean and
    scale map it back to native units. Membership features are already on
    [0, 1], so a threshold near the crossover is rendered linguistically --
    ``Polymer MW is Medium <= 0.48`` becomes ``Polymer MW is not clearly
    Medium``, which is what that test actually asks.
    """
    feature, op, threshold = condition.feature, condition.op, condition.threshold

    if " is " in feature:
        base, label = feature.rsplit(" is ", 1)
        if op in (">", ">="):
            qualifier = "is mostly" if threshold >= 0.5 else "has some"
            return f"{base} {qualifier} {label}"
        qualifier = "is not clearly" if threshold >= 0.5 else "is not"
        return f"{base} {qualifier} {label}"

    if feature in scaler_lookup:
        mean, scale = scaler_lookup[feature]
        threshold = mean + threshold * scale

    symbol = {"<=": "≤", "<": "<", ">=": "≥", ">": ">"}[op]
    return f"{feature} {symbol} {threshold:.4g}"


def scaler_lookup(representation) -> dict[str, tuple[float, float]]:
    """Per-feature ``(mean, scale)`` from a fitted representation's crisp scaler."""
    scaler = getattr(representation, "scaler_", None)
    if scaler is None:
        return {}
    return {
        name: (float(mean), float(scale))
        for name, mean, scale in zip(
            representation.features_, scaler.mean_, scaler.scale_
        )
    }


def render_rule(
    conditions: list[Condition], lookup: dict[str, tuple[float, float]]
) -> str:
    return "IF " + " AND ".join(humanize(c, lookup) for c in conditions)


# --- Extraction ----------------------------------------------------------


def rulefit_rules(fit, max_rules: int = 40, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Fit RuleFit to mimic the trained predictor on the training fold.

    Returns the surviving rules and a fidelity report measured on the held-out
    fold. Fidelity is agreement with the *predictor*, which is what a surrogate
    is supposed to reproduce; it is not accuracy against the true release.
    """
    from imodels import RuleFitRegressor

    train_target = fit.model.predict(fit.X_train)
    surrogate = RuleFitRegressor(
        max_rules=max_rules, random_state=seed, include_linear=False
    )
    surrogate.fit(
        fit.X_train.values, train_target, feature_names=list(fit.X_train.columns)
    )

    valid_target = fit.model.predict(fit.X_valid)
    valid_surrogate = surrogate.predict(fit.X_valid.values)
    fidelity = {
        "fold": fit.fold,
        "method": "RuleFit",
        "fidelity R2 (valid)": float(r2_score(valid_target, valid_surrogate)),
        "fidelity RMSE (valid)": float(
            mean_squared_error(valid_target, valid_surrogate) ** 0.5
        ),
        "n rules kept": 0,
    }

    table = surrogate._get_rules()
    table = table[(table["coef"] != 0) & (table["type"] == "rule")].copy()
    fidelity["n rules kept"] = int(len(table))

    rows = []
    for _, entry in table.iterrows():
        try:
            conditions = parse_rule(entry["rule"])
        except ValueError:
            continue  # unparseable term; excluded rather than guessed at
        rows.append(
            {
                "fold": fit.fold,
                "method": "RuleFit",
                "rule": entry["rule"],
                "conditions": conditions,
                "coef": float(entry["coef"]),
                "train support": float(entry["support"]),
                "importance": float(entry["importance"]),
            }
        )
    return pd.DataFrame(rows), fidelity


def surrogate_tree_rules(
    fit, max_depth: int = 3, seed: int = SEED
) -> tuple[pd.DataFrame, dict]:
    """Fit a shallow decision tree to mimic the predictor; one rule per leaf.

    Leaves partition the space, so unlike RuleFit these rules are mutually
    exclusive and their coverages sum to 1. That makes them easier to present
    as a decision aid, at the cost of much lower fidelity.
    """
    train_target = fit.model.predict(fit.X_train)
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(
        fit.X_train, train_target
    )

    valid_target = fit.model.predict(fit.X_valid)
    valid_surrogate = tree.predict(fit.X_valid)
    fidelity = {
        "fold": fit.fold,
        "method": f"Surrogate tree (depth {max_depth})",
        "fidelity R2 (valid)": float(r2_score(valid_target, valid_surrogate)),
        "fidelity RMSE (valid)": float(
            mean_squared_error(valid_target, valid_surrogate) ** 0.5
        ),
        "n rules kept": int(tree.get_n_leaves()),
    }

    names = list(fit.X_train.columns)
    structure = tree.tree_
    rows = []

    def walk(node: int, path: list[Condition]) -> None:
        if structure.children_left[node] == -1:
            collapsed = simplify(path)
            rows.append(
                {
                    "fold": fit.fold,
                    "method": f"Surrogate tree (depth {max_depth})",
                    "rule": " and ".join(
                        f"{c.feature} {c.op} {c.threshold:.6g}" for c in collapsed
                    ),
                    "conditions": collapsed,
                    # A leaf's mean prediction stands in for RuleFit's coef: it
                    # says which way this branch pushes the prediction.
                    "coef": float(structure.value[node].ravel()[0] - train_target.mean()),
                    "train support": float(
                        structure.n_node_samples[node] / structure.n_node_samples[0]
                    ),
                    "importance": float(
                        abs(structure.value[node].ravel()[0] - train_target.mean())
                    ),
                }
            )
            return
        feature = names[structure.feature[node]]
        threshold = float(structure.threshold[node])
        walk(structure.children_left[node], path + [Condition(feature, "<=", threshold)])
        walk(structure.children_right[node], path + [Condition(feature, ">", threshold)])

    walk(0, [])
    return pd.DataFrame(rows), fidelity


# --- Validation ----------------------------------------------------------


def _cluster_bootstrap_effect(
    y: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% interval for ``mean(y|rule) - mean(y|not rule)``, resampling drugs.

    Drugs are the independent unit: every measurement of one drug shares that
    drug's descriptors, and each curve contributes many correlated points.
    Resampling rows would understate the interval by roughly the square root of
    the cluster size.
    """
    unique = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}

    effects = np.empty(n_boot)
    effects.fill(np.nan)
    for draw in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_group[group] for group in picked])
        inside, outside = mask[rows], ~mask[rows]
        if inside.sum() == 0 or outside.sum() == 0:
            continue
        effects[draw] = y[rows][inside].mean() - y[rows][outside].mean()

    usable = effects[np.isfinite(effects)]
    if len(usable) < n_boot // 10:
        return (np.nan, np.nan)
    return float(np.percentile(usable, 2.5)), float(np.percentile(usable, 97.5))


def validate_rules(
    rules: pd.DataFrame,
    fit,
    groups: np.ndarray,
    min_coverage: float = MIN_COVERAGE,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """Re-test each training-fold rule on the held-out fold.

    ``groups`` are the drug identifiers of the held-out rows, used as bootstrap
    clusters.
    """
    rng = np.random.default_rng(seed)
    y_train = fit.y_train.to_numpy(dtype=float)
    y_valid = fit.y_valid.to_numpy(dtype=float)
    lookup = scaler_lookup(fit.representation)

    records = []
    for _, entry in rules.iterrows():
        conditions = entry["conditions"]
        train_mask = rule_mask(conditions, fit.X_train)
        valid_mask = rule_mask(conditions, fit.X_valid)

        coverage = float(valid_mask.mean())
        train_effect = (
            y_train[train_mask].mean() - y_train[~train_mask].mean()
            if 0 < train_mask.sum() < len(train_mask)
            else np.nan
        )
        valid_effect = (
            y_valid[valid_mask].mean() - y_valid[~valid_mask].mean()
            if 0 < valid_mask.sum() < len(valid_mask)
            else np.nan
        )

        if coverage >= min_coverage and np.isfinite(valid_effect):
            low, high = _cluster_bootstrap_effect(
                y_valid, valid_mask, groups, n_boot, rng
            )
        else:
            low = high = np.nan

        # The formulation part, tested inside the rule's own time window.
        #
        # Almost every extracted rule pairs a formulation condition with a time
        # condition, and release rises steeply with time. Contrasting the rule
        # against all other rows therefore credits the whole gap to the rule
        # when time alone may explain it. Restricting to rows that already
        # satisfy the time conditions holds time roughly fixed and asks the
        # narrower question: among measurements in this time window, does the
        # formulation condition still separate them?
        time_conditions, formulation_conditions = split_time_conditions(conditions)
        partial_effect = partial_low = partial_high = np.nan
        partial_coverage = np.nan
        if formulation_conditions:
            window = (
                rule_mask(time_conditions, fit.X_valid)
                if time_conditions
                else np.ones(len(fit.X_valid), dtype=bool)
            )
            inside = rule_mask(formulation_conditions, fit.X_valid) & window
            outside = (~rule_mask(formulation_conditions, fit.X_valid)) & window
            if inside.sum() > 0 and outside.sum() > 0:
                partial_coverage = float(window.mean())
                partial_effect = y_valid[inside].mean() - y_valid[outside].mean()
                partial_low, partial_high = _cluster_bootstrap_effect(
                    y_valid[window],
                    rule_mask(formulation_conditions, fit.X_valid)[window],
                    groups[window],
                    n_boot,
                    rng,
                )

        partial_holds = (
            np.isfinite(partial_low)
            and np.isfinite(partial_high)
            and (partial_low > 0 or partial_high < 0)
        )

        enough_coverage = coverage >= min_coverage
        same_direction = (
            np.isfinite(train_effect)
            and np.isfinite(valid_effect)
            and np.sign(train_effect) == np.sign(valid_effect)
        )
        excludes_zero = np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)

        records.append(
            {
                "fold": entry["fold"],
                "method": entry["method"],
                "readable": render_rule(conditions, lookup),
                "signature": tuple(sorted(c.signature for c in conditions)),
                "n conditions": len(conditions),
                "time only": is_time_only(conditions),
                "train support": entry["train support"],
                "valid coverage": round(coverage, 4),
                "train effect": train_effect,
                "valid effect": valid_effect,
                "valid effect CI low": low,
                "valid effect CI high": high,
                "enough coverage": enough_coverage,
                "same direction": bool(same_direction),
                "CI excludes 0": bool(excludes_zero),
                "validated": bool(enough_coverage and same_direction and excludes_zero),
                "time window coverage": partial_coverage,
                "formulation effect in window": partial_effect,
                "formulation CI low": partial_low,
                "formulation CI high": partial_high,
                "formulation part holds": bool(partial_holds),
                "raw rule": entry["rule"],
            }
        )

    return pd.DataFrame(records)


def stable_signatures(
    validated: pd.DataFrame,
    min_folds: int = MIN_FOLDS_VALIDATED,
    require: str = "validated",
) -> pd.DataFrame:
    """Condition patterns that validated in several independent folds.

    A single fold's rule can validate by chance. A pattern that survives in
    three or more folds, each holding out different drugs, is a much stronger
    claim -- though still an association, not a mechanism.
    """
    passed = validated[validated["validated"] & validated[require]]
    if passed.empty:
        return pd.DataFrame(
            columns=["signature", "folds validated", "mean valid effect", "example rule"]
        )

    rows = []
    for signature, part in passed.groupby("signature"):
        directions = np.sign(part["valid effect"])
        rows.append(
            {
                "signature": signature,
                "folds validated": part["fold"].nunique(),
                "consistent direction": bool(
                    len(set(directions[np.isfinite(directions)])) == 1
                ),
                "mean valid effect": float(part["valid effect"].mean()),
                "mean valid coverage": float(part["valid coverage"].mean()),
                "example rule": part.sort_values(
                    "valid coverage", ascending=False
                )["readable"].iloc[0],
            }
        )

    table = pd.DataFrame(rows)
    table["stable"] = (table["folds validated"] >= min_folds) & table[
        "consistent direction"
    ]
    return table.sort_values(
        ["stable", "folds validated", "mean valid coverage"], ascending=False
    ).reset_index(drop=True)


def describe_effect(effect: float, target_label: str) -> str:
    """The THEN half of a rule, stated as a direction rather than a mechanism."""
    direction = "higher" if effect > 0 else "lower"
    return f"THEN {target_label} is {direction} by {abs(effect):.3f} on held-out drugs"
