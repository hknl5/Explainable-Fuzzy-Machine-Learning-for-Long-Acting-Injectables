
# Research Proposal (v2 — Detailed)

## Explainable Fuzzy Machine Learning for Interpretable Drug-Release Classification in PLGA Long-Acting Injectables

> This is an updated, detailed version of the original proposal. It preserves
> the original structure and all safety constraints, and integrates: (a) the
> decisions the team has now settled, (b) the empirical findings from the
> completed leakage-controlled study, and (c) a literature review of current
> PLGA + fuzzy work with sources. Text that is a **new decision** or a
> **finding** is marked so collaborators can see what changed.

---

## 1. Research Domain and Problem

This research investigates fuzzy logic, machine learning, and explainable AI for
predicting drug-release behavior from polymer-based long-acting injectable (LAI)
formulations, specifically drug-loaded PLGA (poly(lactide-co-glycolide))
microparticles.

After administration, the drug is released gradually from PLGA microparticles
over an extended period. Release behavior depends on many interacting
characteristics of the drug, the polymer, the formulation composition, the
manufacturing method, the resulting microparticle properties, and the in-vitro
release conditions. Developing an effective formulation normally requires
repeated, time-consuming laboratory experiments. Machine learning can learn the
relationships between formulation characteristics and release outcomes, but those
relationships are nonlinear, high-dimensional, and partly uncertain — which is
the motivation for a fuzzy approach.

**The specific gap this project targets (refined).** Existing ML models for this
dataset classify release behavior using *crisp* thresholds — a formulation is
labelled one class on one side of a hard cutoff and the opposite class on the
other. A value just below the cutoff and a value just above it are treated as
completely different, even though they are physically almost identical. This
project replaces that brittle crisp boundary with a *fuzzy* representation that
assigns graded, overlapping membership (e.g. Low / Medium / High), so that
near-boundary formulations receive a graded, interpretable description rather
than an arbitrary hard flip.

---

## 2. Benchmark Study

The principal benchmark paper is:

> Robles, K. N., & Samad, M. D. (2026). *Predicting early and complete drug
> release from long-acting injectables using explainable machine learning.*
> International Journal of Pharmaceutics. (Preprint: arXiv:2601.02265.)

The benchmark uses explainable ML to predict drug release from static material
and formulation characteristics of drug-loaded PLGA microparticles, across three
tasks:

- **Task 1 — Early Drug-Release Prediction (regression):** predict fractional
  cumulative release at 24 h, 48 h, and 72 h. Output is a fraction in [0, 1].
- **Task 2 — Release-Profile Classification (binary):** each normalized release
  curve is summarized by its area under the curve (AUC); classes are AUC ≤ 0.5
  vs AUC > 0.5. This is a mathematical descriptor of the normalized curve and
  **must not** be read as a clinical burst-release diagnosis.
- **Task 3 — Complete Release-Curve Prediction:** predict the full standardized
  curve from static descriptors, comparing time-dependent and time-independent
  approaches.

---

## 3. Related Work and Literature Positioning (NEW)

A focused review was conducted to place this project relative to current work.
Three findings matter for scope.

**3.1 The same dataset has recent published models.** At least two 2026 studies
use the same 321-profile / 89-drug PLGA dataset:

- *Interpretable Two-Stage Machine Learning for Early and Full Drug Release
  Prediction in PLGA Microspheres* (Pharmaceutics, 19(5):767, May 2026). It
  first classifies slow-release behaviour (≤ 20% release within 3 days), then
  feeds the predicted early-release probability into a regression model. XGBoost
  achieved the lowest MAE (0.126) and highest Pearson r (0.831); SHAP identified
  drug and polymer molecular weight and polymer concentration as influential.
- The benchmark itself (arXiv:2601.02265, 2026): early release at 24/48/72 h,
  release-profile-type classification, and complete-curve prediction, with an
  explicit data-transformation + explainable-ML pipeline.

*Implication:* prediction and SHAP-based explanation on this dataset are already
published. The team's completed experiments reproduced comparable numbers
(r ≈ 0.83), which is consistent with this. Prediction accuracy is therefore
**not** the novel contribution.

**3.2 Tree → fuzzy-rule conversion is an established method.** The pipeline of
(a) extracting crisp rules from a decision tree, (b) transforming them into a
fuzzy model, and (c) optimizing the fuzzy parameters is documented in the fuzzy
systems literature (e.g. Katsis et al., *Constructing a fuzzy inference
framework using crisp decision trees*, ScienceDirect; Suarez & Lutsko, *Globally
Optimal Fuzzy Decision Trees*; *Decision Trees based Fuzzy Rules*, 2016). The
stated rationale in that literature — that crisp trees have sharp decision
boundaries that fuzzification "softens" — is exactly this project's motivation.

**3.3 The novel contribution (refined).** No study on this dataset applies a
*fuzzy* representation to make the classification interpretable and to soften the
crisp decision boundary. **That is this project's contribution: not better
prediction, but interpretable, boundary-robust fuzzy classification, compared
head-to-head against the crisp baseline.**

---

## 4. Dataset

Original dataset: Bao et al., *A dataset on formulation parameters and
characteristics of drug-loaded PLGA microparticles* — compiled from 113
publications; 321 in-vitro release experiments; 4,913 release time points;
formulation, drug, polymer, and microparticle descriptors plus complete
in-vitro release profiles.

**Established facts from the team's data audit (findings):**

- The 4,913 rows are **measurements**, not independent samples; they correspond
  to **321 formulations**, each a curve over time. Static descriptors are
  constant within a formulation.
- **88 unique drug SMILES vs 89 drug names** — two names share one recorded
  structure. Grouping on SMILES is the stricter, safer choice.
- **Zero missing values** (confirmed cell-by-cell) → no imputation.
- Some Release values exceed 1 (max ≈ 1.08); some curves decrease. These are
  **not** clipped or forced monotonic in the main pipeline (they may be real
  measurements; flagged for source review).
- Release durations span 72 h to 238 days (mean ≈ 30 ± 25 days).

**Resolved open item (decision):** the recorded Time unit is **days**. (Earlier
this was uncertain; it is now confirmed. This affects any early-release target
that references 24/48/72 hours — those must be expressed consistently with the
day-based axis.)

---

## 5. Input Features

Formulation Method (categorical → one-hot); Drug MW; Drug TPSA; Drug LogP;
Polymer MW; LA/GA ratio; Initial Drug-to-Polymer Ratio (DPR); Particle Size;
Encapsulation Efficiency (EE); Loading Capacity; Solubility Enhancer
Concentration.

Formulation Index, Drug name, and Drug SMILES are identifiers / grouping keys —
**never model inputs.**

**Feature-level findings (from audit + SHAP):**

- Long-tailed features (Polymer MW, Drug MW, Particle Size) saturate under
  quantile fuzzification (top quartile collapses to one label).
- `LA/GA` and `Solubility Enhancer Concentration` are near-degenerate (dominated
  by a single repeated value; ~71% and ~64% respectively). A Low/Med/High split
  is near-meaningless for these two; a two-level or categorical treatment is
  more honest. **(Decision to confirm with the team.)**
- SHAP on the hybrid arm: the model prefers the **crisp** form for long-tailed
  features and the **fuzzy** form for tie-heavy features (Drug TPSA, Initial
  DPR) — it routes around whichever encoding loses information.

---

## 6. Proposed Methodology (UPDATED — three explicit layers)

The approach is a **hybrid crisp/fuzzy feature-representation framework with an
explicit fuzzy-rule layer**, built to make classification interpretable.

**Layer 1 — Fuzzification.** Convert numerical descriptors into triangular
Low/Medium/High membership values. Knot placement is fitted **on the training
fold only**, on per-formulation values (so long curves do not dominate the
quantiles). Tied/degenerate features use a declared fallback or a reduced
(two-level/categorical) treatment.

**Layer 2 — Predictor.** XGBoost is the primary model (for regression and
classification); GRU/LSTM remains optional for full-curve prediction. Three
feature representations are compared under identical folds and seeds:
crisp-only, fuzzy-only, hybrid.

**Layer 3 — Fuzzy rule generation (the interpretability contribution).** Build
on the team's decision-tree rule extraction (see §9), then **convert the crisp
tree thresholds into fuzzy IF–THEN rules** using the membership functions from
Layer 1 — following the published tree→fuzzy method (§3.2). Output rules read as,
e.g., "IF Particle Size is Low AND Drug LogP is Low THEN release is High (with
graded membership)."

**Note on task focus (decision).** Because the team's goal is *interpretable
classification* — softening the crisp class boundary — the primary task is
**release classification with fuzzy, graded output**, not point prediction. The
fuzzy layer operates both on the input descriptors and on the graded class
assignment (so a near-boundary formulation is, e.g., "fast 0.55 / slow 0.45"
rather than flipped hard at a cutoff).

---

## 7. Handling of Time (UPDATED — decision + finding)

**Finding (from the completed study):** time dominates every predictor — mean
|SHAP| ≈ 0.219, roughly six times the next feature. Rules that pair a
formulation condition with a time condition get credited for an effect that time
alone explains.

**Decision:** for the classification contribution, time is treated as part of
the **target definition** (the class is derived from the curve/early-release
behaviour), not mixed in as an ordinary formulation feature — this keeps the
fuzzy contribution about *formulation characteristics*, which is what a
pharmacist needs, and prevents time from swamping the fuzzy signal. Where a
predictive task does require time (e.g. pointwise release), time is kept on a
separate axis (log1p) rather than blended with descriptors.

---

## 8. Leakage Control (constraint — non-negotiable)

- Split by **drug (exact SMILES)** with `GroupKFold`; all formulations of one
  drug stay in one fold. Zero cross-fold drug overlap is asserted.
- All distribution-dependent steps — scaling, fuzzy knots, feature selection,
  resampling, rule extraction — are fit on the **training fold only**.
- No imputation; no Release clipping; no forced monotonicity.
- Identity/provenance columns are never features.
- An explicit leakage self-check must PASS before any result is reported.

*(Status: the membership-function notebook implements all of the above; 7/7
leakage assertions pass.)*

---

## 9. Rule Generation — Status and Requirements (UPDATED)

**What exists.** A collaborator extracted 8 rules from a
`DecisionTreeRegressor(depth=3)` on [Time, Polymer MW, Particle Size, LA/GA].
These are the starting point for Layer 3.

**Requirements before these count as findings (constraint):**

1. The tree must be trained **inside each training fold**, not on the full
   dataset (the collaborator's exploratory tree used all data → exploratory only).
2. Rules must be **validated on held-out drugs**. In the team's completed
   experiment, 234 rules → 181 passed a naive test, but after a time-window
   control only 14/60 formulation rules survived and **none replicated across
   ≥ 3 folds**. The collaborator's rules are exposed to the same risk and must pass the
   same tests.
3. Rules must center on **formulation descriptors**, not time (see §7). Note the
   depth-3 tree spent most of its splits on Time and never used LA/GA — a sign
   the crisp tree over-resolves the early curve.
4. Only after fold-wise validation may any rule be reported to a pharmaceutical
   audience. **Do not claim validated fuzzy IF–THEN rules before this.**

---

## 10. Candidate Models

- XGBoost — primary, for tabular regression and classification.
- GRU/LSTM — optional, for complete-curve prediction.
- Decision tree (shallow) — for rule extraction / surrogate, feeding Layer 3.
- Rule-extraction methods: RuleFit and/or surrogate tree, then tree→fuzzy
  conversion.

The final fuzzy architecture is **not** described as ANFIS / Mamdani / Sugeno
unless one is explicitly selected and implemented. Current description: a hybrid
crisp/fuzzy feature-representation framework with an explicit tree→fuzzy rule
layer.

---

## 11. Explainability (UPDATED)

SHAP is used for global importance, effect direction, per-formulation
explanations, and crisp-vs-fuzzy feature comparison. **The explicit
rule-generation method is now selected:** decision-tree extraction (RuleFit /
surrogate) followed by tree→fuzzy conversion, fit per training fold and validated
on held-out folds before any rule is reported. This satisfies the original
proposal's requirement that IF–THEN rules not be claimed without a validated,
explicit method.

---

## 12. Evaluation Metrics

- **Classification (primary):** accuracy, precision, recall, F1, AUROC,
  confusion matrix — read against the ~2.9:1 class imbalance (≈ 74.5% of
  formulations are AUC > 0.5), **not** against 50%.
- **Regression (secondary):** RMSE, MAE, Pearson r, R².
- **Fuzzy-specific:** boundary-robustness — show that near-cutoff formulations
  receive graded membership rather than a hard flip (this is the contribution,
  so it needs its own evaluation, e.g. membership smoothness across the boundary
  and agreement of near-boundary cases).
- **Comparison discipline:** crisp vs fuzzy vs hybrid on identical folds/seeds;
  use paired (per-fold) comparison, since fold difficulty dominates the ± sd and
  hides real differences.

---

## 13. Research Question

**Central question:** Can a fuzzy representation of drug, polymer, formulation,
and microparticle characteristics make PLGA release *classification* more
interpretable and more robust at the decision boundary than a crisp-threshold
model — without loss of validity?

Subquestions: which features benefit from fuzzy representation; whether fuzzy
membership gives clearer, boundary-robust class descriptions; which
membership-function design performs best; whether crisp+fuzzy beats either alone;
whether the method generalizes to unseen drugs.

---

## 14. Findings So Far (NEW — honest summary)

From the completed leakage-controlled study (`run_study.py`) and the
membership-function notebook:

- **Fuzzy did not improve prediction, mostly.** In the working regime
  (time-dependent, R² ≈ 0.47–0.49), all arms are within 0.006 RMSE of crisp
  (paired win-count 2/5–4/5 = noise). The one consistent gain (time-independent,
  quantile fuzzification, 5/5 folds at 24 h & 48 h, ≈ 8%) occurs where **every**
  arm has negative R² — fuzzification made a failing model fail less.
- **No rule survived validation** (see §9).
- **The binding constraint is information, not encoding:** ten static descriptors
  carry limited information about a curve for an unseen drug. This — not the
  fuzzy encoding — sets the performance ceiling.

These results reframe the project honestly: the contribution is
**interpretability and boundary behaviour**, evaluated on its own terms, rather
than a claim of higher accuracy.

---

## 15. Research Constraints (unchanged — still binding)

- Do not invent pharmaceutical interpretations.
- Distinguish verified findings from hypotheses.
- Do not treat mathematical AUC classes as clinical burst-release labels.
- Prevent leakage at every stage; keep same-drug formulations in one fold.
- Fit scaling, fuzzification, selection, resampling on training data only.
- Preserve benchmark preprocessing/evaluation when comparing.
- Report negative or insignificant results honestly.
- Do not assume fuzzy features will improve accuracy.
- Do not claim the final fuzzy architecture is selected beyond what is stated.
- Do not claim IF–THEN rules until extraction is implemented **and validated**.
- Explain decisions clearly; provide reproducible code.

---

## 16. Current Status and Next Step

**Decided:** benchmark & dataset; leakage-controlled grouped split; membership
functions (train-fold-only, triangular Low/Med/High); tree→fuzzy rule method;
time treated as target-definition for the classification task; Time unit = days;
fuzzy-as-representation (not ANFIS).

**Still open:** membership-function count/shape per feature; treatment of the two
degenerate features (LA/GA, Solubility Enhancer); exact fuzzy class-boundary
formulation for the output; final ablation design.

**Immediate next step:** implement the fuzzy classification task (graded output)
that builds on the collaborator's rules inside the grouped split, convert the tree
thresholds to fuzzy IF–THEN using the Layer-1 memberships, and compare crisp vs
fuzzy classification on boundary-robustness and standard metrics — after reading
the two 2026 same-dataset studies (§3.1) to state the delta precisely.

---

## Sources

1. Robles & Samad (2026), *Predicting early and complete drug release from LAIs
   using explainable ML*, Int. J. Pharmaceutics — benchmark. arXiv:2601.02265.
2. *Interpretable Two-Stage ML for Early and Full Drug Release Prediction in
   PLGA Microspheres*, Pharmaceutics 19(5):767 (May 2026). doi:10.3390/ph19050767
3. Bao et al., *A dataset on formulation parameters and characteristics of
   drug-loaded PLGA microparticles* — source dataset.
4. Katsis et al., *On constructing a fuzzy inference framework using crisp
   decision trees*, ScienceDirect — tree→fuzzy method.
5. Suarez & Lutsko, *Globally Optimal Fuzzy Decision Trees for Classification and
   Regression* — fuzzy decision-tree theory.
6. *Decision Trees based Fuzzy Rules* (2016), ResearchGate — auto-generating
   fuzzy rules + membership functions from a tree.
7. *Machine Learning for Predicting Drug Release Behavior of PLGA Microspheres*,
   PMC12919544 — additional PLGA ML context.
