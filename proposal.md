
# Research Proposal (v3)

## Interpretable Fuzzy Representation of Drug Release from PLGA Long-Acting Injectables

> **What changed in v3.** v2 framed the contribution as *fuzzy classification*
> of the normalized-AUC class (AUC ≤ 0.5 vs > 0.5) and put the fuzzy layer on the
> model's **inputs**. That misread the research idea. The point of fuzzy logic
> here is **pharmaceutical interpretability and boundary robustness of the
> predicted release itself**. v3 therefore predicts **continuous release** and
> fuzzifies the **output**, and demotes AUC classification to a benchmark
> comparison. See `CHANGES.md` for the line-by-line delta.

---

## 1. The problem this project addresses

A crisp categorisation of release behaviour has to put a cut point somewhere,
and formulations land on either side of it:

| Predicted release | Crisp category |
|---|---|
| 69% | Medium |
| 70% | High |

One percentage point — well inside the measurement scatter of curves pooled from
113 separate publications — changes the description completely. The two
formulations are physically almost identical and are reported as categorically
different.

A fuzzy representation of the same two values instead reads:

| Predicted release | Low | Medium | High |
|---|---|---|---|
| 69% | 0.00 | 0.55 | 0.45 |
| 70% | 0.00 | 0.48 | 0.52 |

Nothing about the underlying prediction has changed. What has changed is that a
borderline formulation is now *described as borderline*, which is the honest
description and the one a formulator can act on. The graded description also
carries information the hard label destroys: "Medium 0.48 / High 0.52" says this
formulation is a coin-flip away from a different characterisation, and
"Medium 0.95 / High 0.05" says it is not.

**This is the contribution.** It is a contribution to interpretation, not to
accuracy, and the proposal does not claim otherwise anywhere.

---

## 2. Research question and objectives

**Central question.** Can drug release from PLGA microparticles be predicted as
a continuous quantity and then *expressed* through overlapping fuzzy linguistic
memberships, so that near-boundary release behaviour receives a graded,
interpretable description rather than an arbitrary hard category — and can
interpretable formulation–release rules be extracted from that representation
and shown to hold on unseen drugs?

**Objectives.**

1. Fuzzify the static formulation and drug descriptors into overlapping
   Low/Medium/High memberships, fitted on training folds only.
2. Predict **continuous** cumulative release.
3. Fuzzify the **predicted release** into Low/Medium/High memberships, and
   quantify the boundary robustness this buys.
4. Extract human-readable IF–THEN rules relating formulation characteristics to
   release behaviour, within controlled temporal contexts, and validate them on
   unseen drugs.
5. Report honestly whether fuzzy representation helps prediction — including if
   it does not.

**Explicitly not an objective:** showing that fuzzy features improve predictive
accuracy. That is tested (§8 A) and reported either way, but it is not the
claim.

---

## 3. The four layers, kept distinct

These are four separate things that v2 conflated. The distinction matters
because they succeed and fail independently.

**Layer 1 — Input fuzzification.** Each suitable static numerical descriptor is
represented by overlapping triangular Low/Medium/High memberships. This is a
choice of *feature encoding*; it changes what the model sees.

**Layer 2 — Continuous release prediction.** XGBoost predicts fractional
cumulative release as a real number. No categorisation happens here at all.

**Layer 3 — Output fuzzification.** The predicted release value is mapped to
Low/Medium/High memberships. **This layer is the contribution**, it is
independent of Layers 1–2, and it can be applied to any regressor's output
without retraining — including a purely crisp one. A model with better accuracy
would produce a better-centred membership vector; the boundary behaviour would
be unchanged.

**Layer 4 — Rule extraction.** A shallow decision tree over the fuzzy
representation proposes IF–THEN rules, which are then validated on held-out
drugs. The tree is a *rule-discovery mechanism*, not a fuzzy inference system,
and this proposal does not describe it as one.

```
raw formulation / drug descriptors
        │
        ├─ leakage-safe preprocessing (train-fold only)
        │
   [L1] ├─ fuzzy memberships for suitable input features
        │
   [L2] ├─ continuous release prediction  ──────────────┐
        │                                              │
   [L3] ├─ fuzzy Low/Medium/High interpretation ────────┤
        │   of the PREDICTED release                    │
        │                                              ▼
   [L4] └─ shallow tree rule extraction        pharmacist-readable
           WITHIN a release phase              graded explanation
                    │
                    └─ rule validation on unseen drugs
                       (candidate → replicated → validated finding)
```

---

## 4. Dataset

Bao et al., *A dataset on formulation parameters and characteristics of
drug-loaded PLGA microparticles*: 113 publications, 321 in-vitro release
experiments, 4,913 release time points.

Established by the team's audit:

- The 4,913 rows are **measurements**, not independent samples: 321 formulations,
  each a curve. Static descriptors are constant within a formulation. Four
  formulations record a repeated time; those release values are averaged and the
  affected rows are flagged.
- **88 unique drug SMILES** vs 89 drug names. Grouping on SMILES is the stricter
  choice and is what is used.
- **Zero missing values** — no imputation.
- Some Release values exceed 1 (max ≈ 1.08) and some curves decrease. Neither is
  clipped nor forced monotonic; both are flagged for source review.
- `Time` is recorded in **days** (confirmed against the benchmark's reported
  duration statistics: min 72 h, max 238 days, mean 30 ± 25).

---

## 5. Input features (Layer 1)

Rule discovery is **not** restricted to a hand-picked subset. All ten static
numerical descriptors enter, and the tree decides which are useful:

Drug MW · Drug TPSA · Drug LogP · Polymer MW · LA/GA · Initial Drug-to-Polymer
Ratio · Particle Size · Drug Encapsulation Efficiency · Drug Loading Capacity ·
Solubility Enhancer Concentration

**Categorical variables are not fuzzified.** `Formulation Method` is nominal —
O/W, S/O/W, W/O/W, S/W/O/W have no ordering, so a Low/Medium/High membership over
them would be meaningless. It is one-hot encoded against a declared vocabulary
(so feature width is equal in every fold) and enters all arms unchanged.

`Formulation Index`, `Drug`, `Drug SMILES` and `DOI` are identity and provenance.
**They are never model inputs.**

**Known feature-level problems, carried forward honestly.** `LA/GA` (71% one
value) and `Solubility Enhancer Concentration` (64% exactly zero) are
near-degenerate; a three-way split is close to meaningless for them and a
declared fallback recomputes knots over distinct values. Long-tailed features
(Polymer MW, Drug MW, Particle Size) saturate under quantile knots — the top
quartile collapses into one label. Both quantile and equal-width knots are run
as separate arms rather than one being asserted as correct.

---

## 6. Output fuzzification (Layer 3) — and the honesty problem at its centre

The predicted release value is mapped to Low/Medium/High through the same
shoulder-triangle (Ruspini) memberships, so the three degrees sum to 1 and read
directly as shares.

**Where the boundaries come from is the hard question, and it is not settled by
this study.** Three origins are possible and they are not interchangeable:

| Origin | Status |
|---|---|
| **Design** — declared (an even split of the [0,1] release axis, 1/3 and 2/3) | Symmetric and inspectable. **Not a pharmaceutical fact.** |
| **Data-derived** — tertiles of the training fold's release distribution | Honest about the corpus. "Medium" then means "middle third of what these 113 papers measured" — a statement about the corpus, not about therapy. |
| **Expert** — supplied by a pharmacist for a stated drug, route and indication | What is actually needed. **Not available.** |

No clinically meaningful Low/Medium/High definition was available, so **this
remains a parameterised framework** and the boundary selection is flagged as a
**supervisor / pharmaceutical-expert decision**. Rather than assume the choice
away, the study *measures its influence*: every boundary-dependent result is
re-run under four alternative cut-point sets, and the write-up separates
conclusions that survive all four from those that move with the choice.

No membership boundary in this work should be presented as pharmaceutical truth.

---

## 7. Handling of time (Layer 4)

**The problem.** Cumulative release only increases. A rule learner given raw time
spends its splits on time and returns *IF Time is large THEN release is high* —
true, high-coverage, and useless. In the v2 study time carried a mean |SHAP| ≈
0.219, roughly six times the next feature, and the rules it produced paired a
formulation condition with a time condition so that time got credit for the
whole effect.

**v2's response was too weak.** It let time compete and then applied a post-hoc
window correction. **v3 removes time from the rule learner entirely** and instead
partitions measurements into release phases, fitting a separate shallow tree
inside each. Time is held roughly fixed by restriction, cannot win a split, and
every rule that emerges is about formulation by construction rather than by
correction.

| Phase | Bound | Provenance |
|---|---|---|
| Early | 0–3 d | **Supported**: the published ≤20%-within-3-days early-release criterion for PLGA microspheres (Pharmaceutics 19(5):767, 2026) treats 3 days as meaningful. |
| Middle | 3–14 d | **Data-derived, not pharmaceutical.** 14 d is the 66th percentile of measured times and gives three comparably sized phases. **Requires expert selection.** |
| Late | > 14 d | Remainder; inherits the same unvalidated 14-day boundary. **Requires expert selection.** |

The 14-day boundary is a dataset property, and the proposal says so rather than
dressing it as biology. Alternative boundary pairs (1/7 d, 7/28 d, corpus
tertiles) are available for sensitivity analysis.

---

## 8. Evaluation — five questions, answered separately

These are separate questions with separate answers, and conflating them is what
produced the v2 framing.

**A. Predictive performance.** Does fuzzy or hybrid representation improve
continuous release prediction over crisp features? Paired per-fold comparison on
identical folds and seeds (the across-fold sd is dominated by fold difficulty,
which is common to all arms and cancels in a paired difference). **Reported
honestly either way.**

**B. Boundary robustness.** Every predicted release is nudged by ±1 percentage
point — the size of the step in the 69%/70% example — and both schemes are asked
how much their description changed. A crisp flip costs 1.0 (the formulation is
reassigned); the fuzzy shift is the total-variation distance between membership
vectors, on the same 0–1 scale, so the two are directly comparable. Also
reported: **how many** formulations are near a boundary at all, since that bounds
how much the contribution can matter.

**C. Interpretability.** What share of held-out predictions receive a graded
multi-set description rather than a single hard label, and what do borderline
cases look like when written out.

**D. Rule usefulness.** Can shallow-tree extraction find reproducible
formulation–release relationships on unseen drugs? Three declared tiers:

- **candidate** — a tree produced it; nothing more. A hypothesis.
- **replicated** — passed coverage, direction and a drug-clustered bootstrap
  interval on its own held-out fold, and an equivalent pattern did so in ≥ 3 of 5
  folds with consistent direction.
- **validated finding** — replicated *and* held-out effect ≥ 0.05 fractional
  release. A rule can be statistically reliable and still too small to matter.

Rules that only reflect elapsed time are excluded structurally (§7). **No
IF–THEN rule is presented as a pharmaceutical conclusion below the top tier.**

**E. Temporal consistency.** Do different formulation characteristics become
important in Early, Middle and Late phases?

**Secondary benchmark.** AUC ≤ 0.5 vs > 0.5 classification is retained for
comparability with prior work on this dataset. It is a mathematical descriptor of
a normalized curve and **must not be read as a clinical burst-release
diagnosis**. It is no longer the headline contribution.

---

## 9. Leakage control (non-negotiable)

- Split by **drug (exact SMILES)** with `GroupKFold(5)`; every formulation of a
  drug stays in one fold; zero cross-fold overlap is asserted before any result.
- **Every data-dependent operation happens inside the training fold**: scaling,
  fuzzy knot fitting, output membership fitting, feature selection, class
  balancing, tree fitting, rule discovery, threshold learning.
- Knots for measurement-level tasks are fitted on the training fold's
  **formulation** rows, not measurement rows, so a 48-point curve does not pull
  the quantiles ten times harder than a 5-point curve.
- No imputation, no release clipping, no forced monotonicity.
- Identity/provenance columns are never features.

---

## 10. Alternative target — a separate experiment

Slow release defined as **≤ 20% released by day 3** (Pharmaceutics 19(5):767,
2026) is run as a **separate experiment**, not as a replacement for the
normalized-AUC task. The two ask genuinely different questions: the AUC class
summarises curve *shape* after the time axis is rescaled to [0,1] and is blind to
elapsed time, while this criterion is anchored to real days, which is what an
early-burst concern is actually about.

The threshold is adopted **because a published source defines it**, not because
it improves any metric. Choosing a target for its metrics would be selecting the
question to fit the answer.

---

## 11. Research constraints (binding)

- Do not invent pharmaceutical interpretations, thresholds, or validated rules.
- Do not present membership boundaries as pharmaceutical truth.
- Do not describe a crisp decision tree as a fuzzy inference system.
- Do not treat mathematical AUC classes as clinical burst-release labels.
- Do not claim fuzzy representation improves accuracy unless experiments show it.
- Distinguish candidate / replicated / validated at every mention of a rule.
- Prevent leakage at every stage; keep same-drug formulations in one fold.
- Report negative and insignificant results honestly.
- Flag every methodological choice requiring supervisor/expert judgement.

---

## 12. Decisions requiring supervisor / pharmaceutical-expert judgement

Listed explicitly because none can be settled from the data alone:

1. **Low/Medium/High release boundaries** (§6) — currently declared, not clinical.
2. **The 14-day Middle/Late phase boundary** (§7) — currently a corpus percentile.
3. **Treatment of the two near-degenerate features**, LA/GA and Solubility
   Enhancer Concentration — whether a three-way split is defensible at all.
4. **Whether release values > 1 and non-monotonic segments** are real
   measurements or transcription errors.
5. **Whether the ≤20%/3-day criterion** is the right early-release definition for
   this project's intended application.

---

## Sources

1. Robles & Samad (2026), *Predicting early and complete drug release from LAIs
   using explainable ML*, Int. J. Pharmaceutics — benchmark. arXiv:2601.02265.
2. *Interpretable Two-Stage ML for Early and Full Drug Release Prediction in
   PLGA Microspheres*, Pharmaceutics 19(5):767 (2026) — source of the
   ≤20%-within-3-days criterion.
3. Bao et al., *A dataset on formulation parameters and characteristics of
   drug-loaded PLGA microparticles* — source dataset.
4. Katsis et al., *On constructing a fuzzy inference framework using crisp
   decision trees* — tree-based rule extraction over fuzzy representations.
5. Suarez & Lutsko, *Globally Optimal Fuzzy Decision Trees for Classification and
   Regression*.
6. Ruspini (1969), *A new approach to clustering* — the sum-to-one partition used
   for both input and output memberships.
