# What changed from the previous implementation, and why

The previous version was internally sound — leakage control, grouped CV and
honest reporting were all in place — but it had applied fuzzy logic to the wrong
side of the model. This lists every change and the reason for it.

---

## 1. The core misunderstanding

**Before.** Fuzzification was applied only to model **inputs**, and the
"interpretability contribution" was framed as fuzzy *classification* of the
normalized-AUC class (AUC ≤ 0.5 vs > 0.5). proposal.md v2 §6 stated: *"the
primary task is release classification with fuzzy, graded output"*, and §13 made
that the central research question.

**Why that was wrong.** The 69% Medium / 70% High boundary problem — the entire
motivation — is about the **predicted release value**, not about a curve-shape
descriptor. Fuzzifying the inputs does nothing for it: a model can take fuzzy
inputs and still emit a hard label that flips at a cut point. And AUC ≤ 0.5 vs
> 0.5 is a mathematical summary of a rescaled curve; softening *its* boundary
does not give a pharmacist a graded reading of how much drug came out.

**After.** The model predicts **continuous release**, and the **predicted value**
is fuzzified into Low/Medium/High. AUC classification is retained as a benchmark
comparison only. The four layers (input fuzzification / continuous prediction /
output fuzzification / rule extraction) are now stated separately in the proposal
because they succeed and fail independently.

**New:** `fuzzypharma/interpret.py`. **Rewritten:** `proposal.md` (v3).

---

## 2. Output fuzzification did not exist

**Before.** No code anywhere converted a predicted release into memberships.
`fuzzy.py` fuzzified inputs only.

**After.** `ReleaseInterpreter` maps any release value — measured or predicted —
to Low/Medium/High memberships that sum to 1. It is deliberately *not* a
scikit-learn transformer: it applies to a prediction vector, not a design matrix,
which makes it hard to accidentally fit on held-out release values. It attaches
to any regressor without retraining.

**Result (Task G):** 36.9% of held-out predictions receive a graded multi-set
description instead of a single label; 557 rows (11.3%) are genuinely borderline
with no set above 0.65 membership.

---

## 3. Boundary robustness was promised but never measured

**Before.** proposal.md v2 §12 listed boundary robustness as the fuzzy-specific
metric — "so it needs its own evaluation". No artifact computed it.

**After.** `boundary_robustness()` nudges every prediction by ±1 percentage
point and measures both schemes on the same 0–1 scale: a crisp flip costs 1.0,
the fuzzy cost is the total-variation distance between membership vectors.

**Result:** 4.0% of held-out predictions sit within 1 point of a cut point. Their
crisp description changes completely; the fuzzy description moves by 0.057.

**Deliberately also reported:** the "100% of near-boundary cases flip" figure is
**true by construction** (near-boundary is *defined* as straddling a cut point)
and is labelled as such in the output rather than presented as a discovery. The
empirical content is the 4.0% exposure and the 0.057-vs-1.0 contrast. The fuzzy
layer does nothing for the other 96%, and that is stated.

---

## 4. Boundary choices were not held to account

**Before.** No mechanism distinguished a declared threshold from a data-derived
one from an expert one.

**After.** `ReleaseInterpreter.provenance` is a required field carried into every
table it produces, so no membership is ever displayed without its origin.
`boundary_sensitivity_to_knots()` re-runs the whole robustness measurement under
four alternative cut-point sets.

**Result:** the fuzzy-vs-crisp contrast survives every boundary choice; the
*number of formulations affected* does not, and is therefore reported as a
property of the corpus rather than of the method. No expert definition was
available, so the framework is explicitly labelled parameterised and the choice
is flagged for supervisor decision.

---

## 5. Rule extraction was a surrogate of XGBoost, with time competing

**Before.** `rulefit_rules` / `surrogate_tree_rules` fitted RuleFit and a shallow
tree to **XGBoost's own predictions**, with `Log1p Time` present as a live
feature. Time then won most splits, and the fix was a *post-hoc* window
correction applied after the fact. Outcome: 234 rules → 14/60 formulation rules
survived → **none replicated across ≥3 folds**.

**Why that was wrong.** A surrogate describes the model, not the data, so it
inherits every one of the model's errors and its rules are one step removed from
any claim about formulations. And discounting time afterwards is weaker than
never letting it compete: the tree has already spent its depth budget on time
before the correction runs.

**After.** `phase_tree_rules()` fits a shallow tree directly on **measured**
release, over the fuzzy membership columns of **all ten** static descriptors, with
**time absent from the feature matrix entirely**. Time is held fixed by
restricting to a release phase instead. `tree_text()` preserves the exact
`export_text` output so the numerical criterion behind each paraphrased rule can
be checked.

**Kept:** the original surrogate path (Task E) still runs, since "what is the
trained model doing" is a legitimate separate question.

---

## 6. No time-phase structure existed

**Before.** Nothing partitioned the data temporally.

**After.** `fuzzypharma/phases.py` defines Early (0–3 d) / Middle (3–14 d) /
Late (>14 d), each with **stated provenance**:

- 3 d is **supported** by the published ≤20%-within-3-days criterion.
- 14 d is **data-derived** (66th percentile of measured times) and is labelled
  **not a pharmaceutical threshold — requires expert selection**, not quietly
  presented as biology.

Alternative boundary pairs are provided for sensitivity analysis.

---

## 7. "Validated" was one bar, not three

**Before.** A rule was `validated` or not. The proposal asked for
candidate / replicated / validated finding to be distinguished.

**After.** `classify_evidence()` assigns tiers with thresholds declared in
advance: replication in ≥3 of 5 drug-grouped folds with consistent direction,
plus a held-out effect of ≥0.05 fractional release (below that, a "difference" is
inside the scatter of curves pooled from 113 publications). Each rule also
carries the reason it stopped where it did.

**Result (Task H), reported as the negative result it is:**

| tier | n rules |
|---|---|
| validated finding | **0** |
| replicated | 0 |
| candidate | 9 |
| rejected | 105 |

All 9 candidates replicated in exactly 1 of 5 folds. This is a cleaner negative
than the previous one: time can no longer be laundering the result, so the
conclusion is now specifically that **ten static descriptors do not resolve a
replicable formulation–release rule at the bar set in advance**.

---

## 8. The alternative target was cited but never run

**Before.** proposal.md v2 §3.1 discussed the ≤20%-within-3-days criterion as
related work. No code implemented it.

**After.** `slow_release_class()` + `run_slow_release_classification()` run it as
a **separate experiment** (Task I) on the same folds and model, so the only
difference is the target. Its docstring states that it is a different question
from the AUC class — real elapsed time vs rescaled curve shape — and that the
threshold is adopted **because a source defines it, not because it improves
metrics**. Formulations whose curve ends before day 3 would be dropped, not
imputed (in this dataset, none are).

---

## 9. There was no pharmacist-facing output

**Before.** Results existed as metric tables.

**After.** `interpretation_table()` writes one row per held-out prediction with
its graded description, and `final_explanation()` assembles the end-to-end
display from real model output.

**Note on what it shows.** The "main explanatory pattern" line is **deliberately
left empty**, because no rule reached the validated-finding tier. Filling it with
a candidate rule would be exactly the failure mode this study exists to avoid: it
would read as a pharmaceutical conclusion while resting on a pattern that did not
replicate.

---

## What was already correct and was kept unchanged

- `GroupKFold(5)` on exact drug SMILES, with zero-overlap asserted.
- All distribution-dependent fitting inside the training fold, including knots
  fitted on formulation rows rather than measurement rows.
- Identity/provenance columns (`Drug`, `Drug SMILES`, `DOI`, `Formulation Index`)
  excluded from every feature matrix.
- `Formulation Method` one-hot encoded against a declared vocabulary — nominal
  categories were **already** not being forced into Low/Medium/High.
- All ten static features already used by the package. The four-feature
  restriction (Time, Polymer MW, Particle Size, LA/GA) existed only in the
  exploratory notebook, not in `fuzzypharma/`.
- No imputation, no release clipping, no forced monotonicity.
- Paired per-fold comparison against the crisp baseline.
- Honest reporting of negative results throughout.
