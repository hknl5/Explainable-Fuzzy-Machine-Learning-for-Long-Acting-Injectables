# What to double-check with my collaborator

Ordered by how much the answer would change the results.

## 1. Time units — the single highest-impact assumption

The workbooks label no units anywhere. Everything treats `Time` as **days**,
which makes the 24/48/72 h early-release targets equal to days 1, 2 and 3.

The evidence is strong but circumstantial: durations run 3.003 to 237.732, and
the benchmark paper reports "minimum 72 h, maximum 238 days, mean 30 ± 25 days",
which this workbook reproduces to three decimals.

**If `Time` were hours instead, every early-release target in Task 3 would be
wrong** — they would land at 24/48/72 in the workbook's own units, deep inside
the curves rather than at their start, and the Task 1 results would be measuring
something else entirely. Worth one explicit confirmation.

Related: are the `Release` values fractions (0–1) or percentages already divided?
108 rows exceed 1.0 (max 1.0816), which we preserve rather than clip.

## 2. Duplicate-time policy for formulations 52 and 305

Four formulations record a repeated time point. We average the repeated release
values in the derived curve table and flag every affected row.

- **136 and 148** repeat the *same* value, so averaging is exact and harmless.
- **52 and 305** record *two different* release values at one time. Averaging
  them is a stated choice, not a fact.

Are those genuine replicate measurements (average is right), or a transcription
error where one of the two should be dropped (average is wrong)? Checking against
the source publications for those two would settle it. The flag is retained so
the analysis can be rerun with them excluded either way.

## 3. Her earlier 8 rules — provenance

Three specific questions, because the answers determine whether those rules can
be reported at all:

- **Did they come from a decision tree, a fuzzy inference system, or something
  else?** The distinction matters for how the paper describes the method.
- **Were the boundaries/thresholds fitted on training data only, or on the full
  dataset?** If full-dataset quantiles or thresholds were used, the rules have
  seen the held-out data and their reported support is optimistic.
- **Were they validated on held-out drugs, or only on the data they were
  extracted from?** Grouping matters specifically: 321 formulations come from
  only 88 drugs, and one drug alone accounts for 49 formulations, so a rule can
  look well-supported while really describing one drug.

Context for why this is worth asking: in this study, **181 of 234** extracted
rules passed a naive held-out test, but only **14** survived once the formulation
condition was tested inside its own time window, and **none** replicated across
3 of 5 folds. A rule set that has not been through those controls will look much
stronger than it is.

## 4. Drug identity — `dexamethasone` vs `β-methasone`

These two drug names carry an **identical** recorded SMILES string, which is why
89 drug names collapse to 88 groups (matching the benchmark's 88). The SMILES
appears to lack the stereochemistry that distinguishes them.

We group them together, which is the conservative choice — it cannot leak, it can
only make the task slightly harder. But if they are genuinely different molecules,
a curated drug identifier would be better than exact SMILES. Does she have one?

## 5. Whether `Profile Duration` may be used as a model input

Currently it is **not** used as a predictor. It is a property of how long the
experiment ran, not of the formulation, so using it to predict release felt like
importing knowledge of the experimental design rather than the product. It is
retained in the tables if we decide otherwise.

Does the benchmark use it? This is a defensible call either way but should be
made deliberately and stated.

## 6. Fold assignment — two implementations now exist

`fuzzypharma/folds.py` has a custom deterministic partitioner written earlier,
with a documented reason: `StratifiedGroupKFold` produced *different folds for
the same seed on different scikit-learn versions* (38–101 formulations per fold
on 1.2.2 vs 63–65 on 1.8.0), meaning the notebook kernel and the shell
interpreter silently disagreed about which drugs were held out.

This study uses `sklearn.model_selection.GroupKFold` on exact SMILES, per the
stated protocol. Both give zero drug leakage; they differ in how evenly they
balance fold size and AUC-class rate. Which should be the project standard? Worth
settling once, since all cross-arm comparisons depend on the folds being fixed.

## 7. Lower-priority

- **Formulation method `S/W/O/W` has exactly one formulation** in the whole
  dataset. It is one-hot encoded against a declared vocabulary so the feature
  width stays constant across folds, but no model can learn anything from n=1.
  Should it be merged or dropped?
- **The AUC class is computed on normalized time**, so a 3-day and a 238-day
  profile are summarized identically and duration is discarded. This follows the
  benchmark, but classification performance is poor across all arms (AUROC
  0.55–0.59) and this may be why. Is a duration-aware variant worth testing?
- **No GRU was implemented.** It is listed as optional in the plan and remains a
  candidate for the complete-curve task; the proposal reflects that it has not
  been done.
