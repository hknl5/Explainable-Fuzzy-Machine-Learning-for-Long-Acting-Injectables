# Validation of Eight Predefined Fuzzy Rules on Unseen Drugs

A leakage-controlled, time-confound-controlled replication study of the eight
predefined rules for PLGA microparticle drug release.

**Notebook:** [`validation_of_8_predefined_fuzzy_rules.ipynb`](validation_of_8_predefined_fuzzy_rules.ipynb)
**Results table:** [`rule_validation_summary.csv`](rule_validation_summary.csv)
**Figures:** [`figures/rule_validation/`](figures/rule_validation/)

---

## 1. What this study is

One question, asked once:

> **Do the eight predefined rules describe relationships that hold for drugs the
> analysis has never seen?**

It is a *validation* study. It is **not** a rule-discovery study, a prediction
benchmark, or a model-tuning exercise.

| This study **does** | This study deliberately **does not** |
|---|---|
| Identify the 8 rules from their original source and reproduce them exactly | Modify, merge, simplify, re-fit or add to the rules |
| Test each rule on held-out drugs via 5-fold `GroupKFold` on exact SMILES | Discover new rules (no tree search, RuleFit, or SHAP) |
| Control for the time confound explicitly, three ways | Use time as a condition inside any rule |
| Fit every data-dependent quantity on the training fold only | Tune anything to improve a score |
| Report per-fold direction and replication honestly | Reuse conclusions from any previous experiment |

**Independence.** Nothing is carried over from earlier notebooks, scripts,
generated rule sets, or result files in this repository. The only inputs are the
two original dataset workbooks and the definition of the eight rules. Every
number in the notebook is computed inside it.

---

## 2. Where the eight rules come from

The rules had to be identified exactly, not guessed. Their provenance is recorded
in the project history in two places.

The proposal (§9, *Rule Generation — Status and Requirements*):

> *"A collaborator extracted 8 rules from a `DecisionTreeRegressor(depth=3)` on
> [Time, Polymer MW, Particle Size, LA/GA]. These are the starting point for
> Layer 3."*

and the membership-function work records the same origin:

> *"A collaborator fitted a small decision tree (`max_depth=3`) on four features —
> `Time`, `Polymer MW`, `Particle Size`, `LA/GA` — and printed 8 rules (8 leaves)."*

So the eight rules are, by definition, **the eight leaves of that tree** — a fully
specified, deterministic object:

| Element | Value |
|---|---|
| Estimator | `DecisionTreeRegressor(max_depth=3)` |
| Inputs | `Time`, `Polymer MW`, `Particle Size`, `LA/GA` |
| Target | `Release` |
| Fitted on | all 4,913 measurement rows of `mp_dataset_processed.xlsx` |
| Leaves | 8 → **the 8 rules** |

Section 3 of the notebook reproduces the tree and **asserts** its seven thresholds
match the archived values — identical with or without the collaborator's
`Release` clip. That assertion is what licenses the rest of the study. Recovering
a fixed, externally specified object is not rule discovery: the tree is never
re-fitted inside a fold and never used to search for anything new.

### The eight rules, verbatim

```
R1:  IF Time <= 0.0022                                        THEN Release = 0.0001
R2:  IF 0.0022 < Time <= 0.1071                               THEN Release = 0.1410
R3:  IF 0.1071 < Time <= 3.3365  AND Particle Size <= 9.5050  THEN Release = 0.4383
R4:  IF 0.1071 < Time <= 3.3365  AND Particle Size >  9.5050  THEN Release = 0.2597
R5:  IF 3.3365 < Time <= 10.8857 AND Polymer MW    <= 21.0000 THEN Release = 0.6598
R6:  IF 3.3365 < Time <= 10.8857 AND Polymer MW    >  21.0000 THEN Release = 0.4549
R7:  IF 10.8857 < Time <= 17.8903                             THEN Release = 0.6629
R8:  IF Time > 17.8903                                        THEN Release = 0.7749
```

---

## 3. The structural finding — visible before any statistics

**The tree spends 5 of its 7 splits on `Time`.**

| Rule | Formulation condition | Testable formulation claim? |
|---|---|---|
| R1, R2, R7, R8 | *none* | ✗ — pure Time |
| R3, R4 | `Particle Size` vs 9.5050 | ✓ |
| R5, R6 | `Polymer MW` vs 21.0000 | ✓ |

`LA/GA` is never used by the tree at all.

Furthermore R3/R4 and R5/R6 are **sibling pairs** — within one time window they
partition the data on a single threshold. So the rule set contains:

- **2 distinct formulation claims**, each stated twice from opposite sides, and
- **4 statements about elapsed time**.

This is the single most important fact about the rule set, and it constrains what
any validation can possibly show.

### How the study handles it without altering the rules

- Each rule's `Time` interval is treated as a **temporal context (phase
  restriction)** — the sanctioned use of time. Time restricts *where* a rule is
  evaluated.
- Time is **never** treated as a rule condition that can earn credit for an effect.
- For R3–R6 the claim under test is the formulation condition, with time held fixed.
- For R1, R2, R7, R8 removing time's credit leaves **no formulation claim at all**.
  They are still evaluated and reported in full, but their verdict is structural,
  not statistical.

---

## 4. Design

### Splitting

5-fold `GroupKFold` grouped on **exact drug SMILES**. The generalisation target is
an unseen drug, so the split must be at drug level. This matters more than usual
here: 321 formulations come from only 88 drugs, and a single drug accounts for 49
formulations. A row-level split would test a rule largely on drugs it had already
seen and make almost anything look supported. The notebook asserts zero
train/validation drug overlap in every fold before reporting any result.

### Leakage discipline

| Quantity | Fitted on |
|---|---|
| Time-adjustment bin edges and within-bin release means | training fold only |
| Fuzzy membership knots | training fold only, **per formulation** |
| The 8 rule thresholds | **not fitted at all** — externally predefined and fixed |

The last row is the point of the study: the rules are a *prior hypothesis*, held
constant across folds. Re-fitting them per fold would be rule discovery.

Membership knots use per-formulation values, not per-measurement values, so a
formulation with 48 sampled time points does not pull a quantile ten times harder
than one with 5.

### Three controls for time

Cumulative release only increases, so any rule conditioning on `Time` will appear
to work. In this dataset `|r|` with Release is **0.678 for Time** versus
0.035–0.054 for every rule feature — roughly **17×**.

- **Control A — restriction (primary, R3–R6).** Compare a rule against its
  sibling: same time window, opposite side of the same threshold. Time is held
  fixed by construction, so the contrast is purely the formulation condition.
- **Control B — time-adjusted residuals (uniform, all 8 rules).** On the training
  fold only, estimate mean release within each decile of `log(1+Time)`; subtract
  that fitted profile from held-out release. A rule whose apparent effect was
  elapsed time collapses toward zero.
- **Control C — release phases.** The rules' own windows are arbitrary CART cut
  points, so the two formulation claims are re-tested inside *a priori* phases.

### Phase definition

From established triphasic PLGA behaviour, not fitted to this dataset:

| Phase | Bound | Physical justification |
|---|---|---|
| **Early** | `Time <= 1 day` | Burst — surface-associated drug; governed by surface area and wetting |
| **Middle** | `1 < Time <= 14 days` | Diffusion-controlled transport through the intact matrix |
| **Late** | `Time > 14 days` | Erosion-controlled — bulk hydrolytic degradation of PLGA |

Both boundaries are fixed in advance and identical in every fold, so they cannot
leak. **Time defines the phase and nothing else.**

### Statistics

95% intervals come from a **cluster bootstrap resampling whole drugs**, not rows.
Held-out rows are many correlated measurements from few drugs; a row-level
bootstrap would treat ~1,000 correlated points as independent and declare nearly
everything significant.

A fold-level contrast is reported as `n/a` if either side has fewer than **3
distinct drugs** — a difference computed across two drugs is a statement about
those two drugs. These appear as gaps, never silently dropped.

### Replication criterion — fixed in advance

> A rule is **replicated** only if its time-controlled effect reproduces the
> **same direction in at least 3 of the 5 held-out folds**.

Working on the full dataset, or in a single fold, does not count.

---

## 5. Results

| Rule | Coverage | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Replicated? |
|---|---|---|---|---|---|---|---|
| R1 | 6.5% (18 drugs) | −0.091 | −0.005 | −0.144 | −0.124 | −0.001 | N/A — no formulation condition |
| R2 | 0.7% (4 drugs) | n/a | +0.104 | n/a | −0.016 | n/a | N/A — no formulation condition |
| R3 | 4.5% (4 drugs) | n/a | +0.194 ✓ | n/a | +0.254 ✓ | +0.116 ✓ | YES (3/5) — borderline |
| R4 | 18.7% (15 drugs) | n/a | −0.194 ✓ | n/a | −0.254 ✓ | −0.116 ✓ | YES (3/5) — borderline |
| R5 | 11.4% (9 drugs) | +0.360 ✓ | +0.364 ✓ | +0.047 ✓ | +0.109 ✓ | +0.194 ✓ | **YES (5/5)** |
| R6 | 16.4% (12 drugs) | −0.360 ✓ | −0.364 ✓ | −0.047 ✓ | −0.109 ✓ | −0.194 ✓ | **YES (5/5)** |
| R7 | 15.4% (16 drugs) | −0.012 | −0.137 | +0.136 | +0.025 | −0.059 | N/A — no formulation condition |
| R8 | 26.4% (13 drugs) | −0.019 | +0.146 | −0.110 | −0.088 | +0.136 | N/A — no formulation condition |

Fold columns are the time-controlled effect. ✓ = matches the rule's own predicted
direction. Full untruncated interpretations are in `rule_validation_summary.csv`.

### Supported

**R5 / R6 — polymer molecular weight** (one claim, both sides).
*Within a fixed time window, lower-MW PLGA (≤ 21 kDa) releases more drug.*
Direction in **5 of 5 folds**, mean **+0.215**, CI excludes 0 in 2/5 folds, fires
on 9–12 distinct drugs per fold, holds **5/5 in the Middle (diffusion) phase**,
and is unchanged under graded fuzzy membership. Consistent with faster hydrolytic
degradation of lower-MW PLGA — though the data are observational and support no
causal claim.

### Weak / inconsistent

**R3 / R4 — particle size** (one claim, both sides).
*Within a fixed time window, smaller particles (≤ 9.505 µm) release more drug.*
Direction held in **all 3 evaluable folds** (mean +0.188), which meets the ≥3/5
criterion — but *exactly*, and with real caveats:

- **Folds 1 and 3 could not be tested**: R3 fired on only **2 distinct held-out
  drugs**, below the 3-drug guard.
- **No individual fold's CI excludes zero.**
- **Phase-dependent**: holds 3/3 in Early (+0.098) and Middle (+0.182), but
  **disappears in Late erosion** (1/3 folds, mean −0.029).
- Coverage is low and unstable (0.6%–10.4% of held-out rows).

A promising lead requiring more drugs, **not an established finding**.

### Unsupported

**R1, R2, R7, R8 — no formulation content.** All four show large naive effects
(R1 at −0.56; R8 at +0.34 with 5/5 folds positive) that collapse under time
control, losing **53%–87%** of their magnitude. R2, R7 and R8 also lose any
consistent sign.

R1 is a partial exception worth stating precisely: its residual stays weakly
negative in all 5 folds (mean −0.07). That is **not** a formulation effect — R1
has no formulation term that could produce one. Release rises near-vertically out
of `t = 0`, and a 10-bin decile adjustment cannot flatten a step that steep inside
its lowest bin. The residual is an artefact of the correction's resolution.

Together these four restate the shape of the cumulative release curve — release
is ≈0 at t=0 and high after ~18 days — which is true, trivial, and carries no
information about how to formulate.

---

## 6. Conclusion

**Two of the eight rules (R5, R6) are supported on unseen drugs. Two more (R3, R4)
show consistent but weak evidence. The remaining four (R1, R2, R7, R8) are
unsupported as formulation rules, because they contain no formulation condition
at all.**

Because of the sibling structure, that is really: **one of the two formulation
relationships encoded in this rule set generalises to unseen drugs.**

The dominant limitation is structural and was visible before any statistics: a
depth-3 tree fitted with `Time` among its inputs spent 5 of 7 splits on time. Any
future rule extraction for this purpose should **exclude time from the learner**
and fit within release phases, so formulation descriptors compete only against
each other.

On fuzzification specifically: crisp and fuzzy representations reached the **same
verdict for every rule**. For R5/R6 they agree exactly (5/5 both). For R3/R4 the
graded version is directionally consistent in 5/5 and 4/5 folds — *more* folds
than the crisp test, because graded membership uses every held-out row in the
window instead of discarding folds where the hard threshold isolated two drugs.
That is the practical benefit fuzzification delivered here: it made a thinly
covered rule testable, without changing any conclusion. It produced robustness and
coverage, not new findings — and it rescued none of the four pure-Time rules,
because no encoding can supply formulation content a rule never contained.

---

## 7. Limitations

- **88 drugs, unevenly distributed.** One drug contributes 49 formulations. Some
  fold-level contrasts rest on very few distinct drugs — hence the drug-clustered
  bootstrap and the 3-drug guard, and hence R3's discount.
- **Observational data from 113 publications.** All effects are associations. No
  causal or mechanistic claim is made, and nothing here is a regulatory or
  clinical statement about burst release.
- **The rule thresholds were originally derived on the full dataset**, so they have
  seen every drug tested here. This biases the study *toward* finding support —
  which makes the four failures strong, and the R5/R6 result slightly optimistic.
- **The predefined time windows are arbitrary CART cut points.** The phase analysis
  is the guard against that, and R5/R6 passed it.
- `Time` is treated as **days** throughout, consistent with the recorded duration
  range (3.003–237.732) and the benchmark's reported "72 h minimum, 238 days
  maximum".

---

## 8. Reproducing

```bash
jupyter nbconvert --to notebook --execute --inplace \
    validation_of_8_predefined_fuzzy_rules.ipynb
```

The notebook is self-contained: it imports nothing from this repository and reads
only the two dataset workbooks. It is **deterministic** — executing from cleared
outputs reproduces `rule_validation_summary.csv` byte-for-byte. It also asserts,
and will fail rather than proceed, if:

- the dataset is not 4,913 rows / 321 formulations / 88 SMILES,
- the reproduced tree thresholds do not match the archived rule set,
- any fold has train/validation drug overlap.

Two helper scripts regenerate the exported images from committed results without
re-running the analysis:

| Script | Purpose |
|---|---|
| `export_evidence.py` | Extracts raw notebook outputs. Figures are the PNG bytes embedded in the `.ipynb` (byte-identical); text and tables are saved verbatim as `.txt` and typeset to `.png` in plain monospace. |
| `export_figures.py` | Builds the presentation set. Figures 2–3 come from `rule_validation_summary.csv`; the tree figure asserts the archived thresholds before drawing; the time figure asserts its correlations against the notebook's printed values. |

---

## 9. Notebook map

| Cell | Produces |
|---|---|
| `In[2]` | Dataset load and integrity assertions |
| `In[3]` | Source tree reproduced; thresholds asserted against the archived set |
| `In[4]` | The 8 rules printed verbatim |
| `In[5]` | Rule structure audit + full-dataset behaviour |
| `In[6]` | Fold table + leakage proof |
| `In[7]` | **Figure** — time dominance; correlation printout |
| `In[8]` | Phase definition table |
| `In[9]` | **Figure** — fuzzy membership functions; knots per fold |
| `In[12]` | Per-rule, per-fold results on held-out drugs |
| `In[13]` | **Figure** — naive vs time-controlled per rule/fold; shrinkage table |
| `In[14]` | Phase summary by claim |
| `In[15]` | **Figure** — phase effect by fold |
| `In[16]` | Crisp vs fuzzy agreement |
| `In[17]` | **Final summary table** (written to `rule_validation_summary.csv`) |
| `In[18]` | **Figure** — verdict dot plot |
| `In[19]` | **Figure** — `plot_tree()` of the already-fitted source tree *(evidence cell)* |
| `In[20]` | Untruncated final table *(evidence cell)* |

`In[19]` and `In[20]` were appended after the study was complete, to export
evidence. They render existing objects (`source_tree`, `final`) and recompute
nothing; re-executing the notebook with them present leaves every original output
byte-identical.

---

## 10. Figure index

### `figures/rule_validation/notebook_evidence/`

Raw notebook output. Figures are byte-identical to the `.ipynb`; text is verbatim,
with `.txt` sources alongside.

| File | Cell |
|---|---|
| `01_decision_tree_plot_from_notebook.png` | `In[19]` — `plot_tree()` figure |
| `01_decision_tree_export_text_In3.*` | `In[3]` — `export_text()` (text, not a figure) |
| `02_the_8_predefined_rules_In4.*` | `In[4]` |
| `02b_rule_structure_table_In5.*` | `In[5]` |
| `03_fold_by_fold_results_In12.*` | `In[12]` |
| `03b_per_rule_per_fold_figure_In13.png` | `In[13]` |
| `04_phase_summary_by_claim_In14.*` | `In[14]` |
| `04b_phase_effect_by_fold_figure_In15.png` | `In[15]` |
| `05_time_dominance_figure_In7.png` | `In[7]` |
| `05b_time_correlations_In7.*` | `In[7]` |
| `05c_time_control_shrinkage_In13.*` | `In[13]` |
| `06_final_summary_table_In17.*` | `In[17]` (pandas-truncated, as printed) |
| `06_final_summary_full.png` | `In[20]` (untruncated) |

### `figures/rule_validation/presentation/`

Redrawn for slides from the committed results. Useful for talks; **cite the
`notebook_evidence/` versions as evidence.**

| File | Content |
|---|---|
| `01_original_decision_tree.png` | Tree with Time vs formulation splits colour-coded by verdict |
| `02_rules_validation_summary.png` | All 8 verdicts + the 2 underlying claims |
| `03_fold_validation.png` | The 4 formulation rules across 5 folds |
| `04_time_dominance.png` | Time confounding |
