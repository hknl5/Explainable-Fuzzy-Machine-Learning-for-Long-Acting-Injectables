# Fuzzy Release Classification — `worktree-fuzzy-classification`

> **Branch scope.** Boundary-robust fuzzy classification of PLGA release behaviour:
> does a *graded* class output describe near-boundary formulations more truthfully
> than a hard label? For the overall project overview, see the `main` branch and
> `proposal.md`.

Part of *Explainable Fuzzy Machine Learning for Long-Acting Injectables*.

**All numbers below are produced by `fuzzy_classification.ipynb`; none is hand-typed.**
The full generated report is in [`FINDINGS.md`](FINDINGS.md).

---

## Setup

| | |
|---|---|
| Formulations | 321 |
| Drug groups (exact SMILES) | 88 |
| Split | `GroupKFold(5)` on exact SMILES; zero drug overlap asserted |
| Target | AUC of the normalized release curve, cut at 0.5 |
| **Base rate** | **0.7445 (2.91:1) — majority-class accuracy is 0.7445** |
| Time | **not** a feature; enters only through the target definition |
| Preprocessing | no imputation, no `Release` clipping (max observed 1.0816) |

> Read every accuracy against **0.7445**, not against 0.50.

---

## Headline result

**Fuzzy representation did not improve classification accuracy — and that is
reported as a result rather than worked around.**

Six arms (crisp / fuzzy / hybrid features × hard / soft labels), identical folds and
seeds. **6 of 6 arms score below the majority-class baseline on accuracy.**

| arm | label | accuracy | balanced acc | f1 | AUROC |
|---|---|---|---|---|---|
| crisp | hard | 0.7004 | 0.4837 | 0.8126 | 0.5583 |
| crisp | soft | 0.7098 | 0.5007 | 0.8196 | 0.6026 |
| fuzzy | hard | 0.6848 | 0.4844 | 0.8007 | 0.6090 |
| fuzzy | soft | 0.7067 | 0.5168 | 0.8108 | 0.5982 |
| hybrid | hard | 0.7037 | 0.5166 | 0.8141 | 0.5742 |
| **hybrid** | **soft** | **0.7254** | **0.5335** | **0.8273** | 0.6047 |

Paired per-fold win counts on accuracy are **1–2 of 5** — that is what noise looks
like. On AUROC the direction *is* consistent (4–5 of 5 folds, mean gain ≈ +0.04 to
+0.05), but it is a *ranking* gain in a weak regime (≈0.56 → ≈0.61) that never lifts
any arm above the baseline, computed on ~64 formulations per fold. **Recorded as a
lead, not as the contribution.**

## The actual contribution — boundary robustness

Perturbation: recompute each curve's AUC from half its interior time points — a change
in *reporting schedule*, not in the formulation.

| Quantity | Value |
|---|---|
| Fuzzy boundary half-width `w` (mean over folds) | 0.0570 |
| Formulations receiving a graded reading (0 < µ < 1) | **63 of 321 (19.6%)** |
| **Crisp labels that flip under the perturbation** | **4 of 321 (1.25%)** |
| Crisp mean absolute change | 0.0125 |
| **Fuzzy mean absolute change** | **0.0077** |
| Sensitivity at the cutoff, crisp | infinite (step) |
| Sensitivity at the cutoff, fuzzy | 1/(2w) = 8.77 |
| MAE(model output, graded target) near boundary | 0.3866 |
| MAE(model output, hard label) near boundary | 0.5128 |

The crisp label changes by the maximum possible amount (1.0) on 4 formulations in
response to evidence that barely moved; the fuzzy membership moves by 0.0077 on
average. Near the boundary the model's own output is **0.1262 closer** to the graded
target than to the hard one. For the 63 formulations sitting near the cutoff, a
reading like *"fast 0.52 / slow 0.48"* is a more truthful description than a
capitalised **FAST**.

Figure: `figures/fig2_boundary_robustness.png`.

## Rules

- Trees refit inside each of the 5 training folds, formulation descriptors only, no `Time`.
- Crisp thresholds converted to fuzzy IF–THEN antecedents using each fold's own memberships.
- Rules extracted: **35** · passed the held-out test in their own fold: **2** ·
  **replicating in ≥ 3 of 5 folds: 0**

**No rule is a validated finding. Every rule is a LEAD.** This reproduces the earlier
result (`proposal.md` §9): tree→fuzzy conversion did not rescue the rules.

## SHAP — why fuzzifying the inputs costs information

Per-column mean |SHAP| in the hybrid arm, which is offered both encodings and picks:

- crisp form: **0.2482** per column (10 columns, 56.0% of total credit)
- fuzzy form: **0.0701** per column (26 columns, 41.1% of total credit)

**Per column a crisp feature carries 3.54× the weight of a fuzzy one.** Offered both,
the model prefers crisp — fuzzifying the *inputs* subtracts information here.

## Improvement attempts — all rejected

The baseline configuration (3 sets, quantile knots, α = 0.25) is retained.

| Attempt | Reason tried | Result |
|---|---|---|
| 5 fuzzy sets | long-tailed features saturate at 3 sets | mean ΔAUROC −0.0188 — rejected |
| uniform knots | quantile knots crowd where data is dense | mean ΔAUROC −0.0433 — rejected |
| α = 0.10 | narrower graded band | mean ΔAUROC −0.0083 — rejected |
| α = 0.50 | wider graded band | mean ΔAUROC +0.0036 — rejected |

---

## Honest conclusion

Fuzzy representation did **not** improve classification accuracy. The deeper finding
is that *no* arm beats the majority-class baseline of 0.7445: **ten static descriptors
do not determine the AUC class of a curve for a drug the model has never seen.** SHAP
supports the mechanism. The binding constraint is **information, not encoding**.

What *did* work is the contribution the project actually claims — the fuzzy class
**output**: a clearly measured positive result on boundary robustness, sitting
alongside a clearly stated null result on accuracy.

---

## Files

```text
├── fuzzy_classification.ipynb   # The notebook — produces every number in FINDINGS.md
├── FINDINGS.md                  # Generated report (full tables)
├── figures/
│   ├── fig1_arm_comparison.png      # Six-arm comparison
│   ├── fig2_boundary_robustness.png # The contribution
│   └── fig3_shap.png                # Crisp vs fuzzy SHAP credit
├── mp_dataset_initial.xlsx      # Source dataset and metadata
├── mp_dataset_processed.xlsx    # Numeric modeling dataset
└── proposal.md                  # Research context
```

## Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace fuzzy_classification.ipynb
```

## Related

The AUC class is a mathematical summary of the normalized curve — **not** a clinical
or regulatory burst-release label. For validation of the 8 predefined rules on unseen
drugs, see `worktree-rule-validation-study`.
