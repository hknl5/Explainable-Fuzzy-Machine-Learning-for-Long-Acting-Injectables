# Fuzzy Membership Functions — `worktree-membership-functions`

> **Branch scope.** This branch contains one deliverable: leakage-free triangular
> Low / Medium / High membership functions for the PLGA microparticle dataset,
> built from a collaborator's crisp decision-tree thresholds.
> For the overall project overview, see the `main` branch and `proposal.md`.

Part of *Explainable Fuzzy Machine Learning for Long-Acting Injectables* — research
into whether fuzzy representations improve prediction and interpretation of drug
release from drug-loaded PLGA microparticles.

---

## What this branch does

A collaborator fitted a small `DecisionTreeRegressor(max_depth=3)` on four features
(`Time`, `Polymer MW`, `Particle Size`, `LA/GA`) and printed **8 rules** (8 leaves).
Those rules are **crisp**: a formulation either satisfies `Time <= 3.34` or it does
not, with nothing in between.

This notebook takes those 8 rules as its starting point and replaces every hard
threshold with a **fuzzy membership** — instead of "Time is short (yes/no)", it says
"Time is Low with degree 0.8, Medium with degree 0.2, High with degree 0.0". The
membership functions and their plots are the deliverable.

**Leakage control is the point of the notebook, not a side note.** A membership
function is *fitted from data* — its knots are quantiles of the training values.
That makes it the same kind of object as a scaler: compute it on the whole dataset
and information from the rows you later evaluate on has already leaked into the
model's inputs.

Four rules are enforced, and Task 6 proves each one:

| # | Rule | Enforced in |
|---|---|---|
| 1 | Split **first**, by drug (`Drug SMILES`), before computing anything | Task 3 |
| 2 | Knots come from the **training fold only**, applied unchanged to validation | Task 4 |
| 3 | Knots come from **per-formulation** values, not per-measurement values | Task 4 |
| 4 | No clipping, no imputation, no monotonicity forcing; SMILES never a feature | Tasks 1, 6 |

Rule 3 matters because a formulation with 48 sampled time points would otherwise
pull a quantile ten times harder than one with 5.

### Corrected from the exploratory notebook

The collaborator's original pass had issues this branch deliberately does not carry
forward:

- `RobustScaler`, `MinMaxScaler` and `DecisionTreeRegressor` were all `.fit()` on all
  4,913 rows with **no train/test split at all** → fixed by splitting first, so every
  statistic afterwards is a training-fold statistic.
- **No drug grouping.** Rows of the same drug are near-duplicates, so even a random
  split would effectively test the model on its own training data → fixed with
  `GroupKFold` on exact `Drug SMILES`, with printed proof of zero drug overlap.
- `Release` was clipped to [0, 1]. The notebook verifies the clip **made no difference
  to the rules** and drops it, since it destroyed data for no benefit.

---

## Notebook structure

| Task | Content |
|---|---|
| 0 | What the notebook does, and why leakage control is the whole point |
| 1 | Load data, attach `Drug SMILES` as the grouping key |
| 2 | Reproduce the collaborator's 8 rules (the starting point) |
| 3 | The grouped split — the leakage barrier |
| 4 | Build the membership functions: shape, knot provenance, machinery, sanity checks, per-fold fitting, and the tie situation measured rather than assumed |
| 5 | Visualise the memberships, with the crisp thresholds overlaid |
| 6 | **Leakage self-check (must pass)** |
| 7 | Summary, the honest next step, and what the notebook does *not* claim |

---

## Outputs

`membership_knots.csv` — the fitted knots, one row per (fold, feature), with the
`q25_a` / `q50_b` / `q75_c` shoulders and the rule used. Eleven features are
fuzzified across 5 folds:

`Drug MW` · `Drug TPSA` · `Drug LogP` · `Drug Loading Capacity` ·
`Drug Encapsulation Efficiency` · `Polymer MW` · `LA/GA` ·
`Initial Drug-to-Polymer Ratio` · `Particle Size` ·
`Solubility Enhancer Concentration` · `Time`

Knots differ per fold by construction — that is the leakage control working, not
instability.

---

## Files

```text
├── membership_functions.ipynb   # The notebook (Tasks 0-7)
├── membership_knots.csv         # Fitted knots per fold and feature
├── membership_functions.png     # Membership function figure
├── hajar_rules_fuzzy.png        # The collaborator's crisp thresholds vs the fuzzy memberships
├── mp_dataset_initial.xlsx      # Source dataset and metadata
├── mp_dataset_processed.xlsx    # Numeric modeling dataset
└── proposal.md                  # Research context
```

---

## What this branch does *not* claim

- It does **not** claim the 8 rules are valid. It only re-expresses their thresholds
  as memberships. Whether those rules generalise to unseen drugs is a separate
  question, answered on `worktree-rule-validation-study` — where the result is that
  only the Polymer MW rules (R5/R6) replicate in 5 of 5 held-out folds, the Particle
  Size rules (R3/R4) are borderline, and the four pure-`Time` rules are unsupported.
- It does **not** claim fuzzy representation improves accuracy. No model is trained
  or compared here.
- The membership boundaries are **not** pharmaceutical truth. They are quantiles of
  a training fold.

---

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace membership_functions.ipynb
```

Task 6 must print a passing leakage self-check. If it does not, no downstream result
from this branch should be used.
