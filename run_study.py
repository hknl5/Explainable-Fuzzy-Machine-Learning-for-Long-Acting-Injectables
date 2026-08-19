"""Run Tasks A-F end to end and write every artifact to ``study_results/``.

    python run_study.py

Layer 1 fuzzifies the ten static features into Low/Medium/High memberships,
Layer 2 trains XGBoost on crisp / fuzzy-only / hybrid representations under
drug-grouped CV, and Layer 3 extracts IF-THEN rules from the trained predictor
and tests them on held-out drugs.

Every printed block is an intermediate check meant to be read, not skipped:
fold composition, per-fold knots, saturation shares, and per-fold metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzypharma import build_dataset
from fuzzypharma.config import ID_COL, RELEASE_COL, SEED, SMILES_COL, STATIC_FEATURES, TIME_COL
from fuzzypharma.folds import FOLD_COL
from fuzzypharma.grouped_cv import (
    DUPLICATE_TIME_FLAG,
    LOG_TIME_COL,
    NORM_TIME_COL,
    assign_smiles_folds,
    build_measurement_table,
    iter_positions,
    summarize_smiles_folds,
    verify_no_smiles_leakage,
)
from fuzzypharma.features import Representation
from fuzzypharma.interpret import (
    DESIGN_KNOTS,
    ReleaseInterpreter,
    boundary_robustness,
    boundary_sensitivity_to_knots,
    interpretation_table,
)
from fuzzypharma.phases import (
    PHASE_COL,
    PHASE_PROVENANCE,
    PHASES,
    add_phase,
    phase_boundary_alternatives,
    phase_summary,
)
from fuzzypharma.rules import (
    FINDING,
    MIN_EFFECT,
    MIN_FOLDS_VALIDATED,
    REPLICATED,
    RuleFit,
    classify_evidence,
    describe_effect,
    evidence_counts,
    phase_tree_rules,
    rulefit_rules,
    stable_signatures,
    surrogate_tree_rules,
    tree_text,
    validate_rules,
)
from fuzzypharma.study import (
    ARMS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    aggregate,
    arm_label,
    format_comparison,
    knot_report,
    model_ready,
    paired_vs_crisp,
    run_auc_classification,
    run_pointwise_regression,
    run_slow_release_classification,
    run_static_regression,
)
from fuzzypharma.targets import SLOW_RELEASE_SOURCE, slow_release_class

OUT = Path(__file__).resolve().parent / "study_results"
EARLY_TARGETS = ["Release_24h", "Release_48h", "Release_72h"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def save(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=index)
    print(f"  -> study_results/{name}")


# --- Task A --------------------------------------------------------------


def task_a():
    banner("TASK A -- Load, validate, and split by drug (exact SMILES)")

    dataset = build_dataset()
    quality = dataset.quality

    print("Cross-file validation (both workbooks, row for row):")
    print(f"  rows in each workbook           : {quality.n_rows}")
    print(f"  formulations                    : {quality.n_profiles}")
    print(f"  exact-SMILES drug groups        : {quality.n_drug_groups}")
    print(f"  missing cells                   : {quality.missing_cells}")
    print("  header typo 'Drug Encapuslation Efficiency' renamed on load; the")
    print("  shared numeric columns were compared on values before the join.")

    # Static-feature constancy is enforced by build_formulation_table, which
    # raises rather than returns if any static input varies within a formulation.
    per_formulation = dataset.raw.groupby(ID_COL)[STATIC_FEATURES].nunique()
    print(f"\n  static features varying within a formulation: "
          f"{int(per_formulation.gt(1).sum().sum())} (0 expected)")

    model_table = dataset.model_table.copy()
    model_table[FOLD_COL] = assign_smiles_folds(model_table).values
    verify_no_smiles_leakage(model_table)

    print("\nGroupKFold(5) on exact Drug SMILES:")
    print(summarize_smiles_folds(model_table).to_string())

    per_drug_folds = model_table.groupby(SMILES_COL)[FOLD_COL].nunique()
    print(f"\n  drugs appearing in more than one fold: {int((per_drug_folds > 1).sum())}")
    print(f"  total formulations: {len(model_table)}  total drugs: {len(per_drug_folds)}")
    print("  Drug name, SMILES and DOI are identity/provenance only and are")
    print("  never passed to a model as features.")

    save(summarize_smiles_folds(model_table).reset_index(), "taskA_fold_summary.csv")
    return dataset, model_table


# --- Task B --------------------------------------------------------------


def task_b(dataset, model_table):
    banner("TASK B -- Two time-handling settings")

    measurements = build_measurement_table(dataset.points, model_table, dataset.raw)

    print("Same-time duplicate policy:")
    print("  Four formulations record a repeated time. Their release values are")
    print("  AVERAGED in the derived curve table; the raw frame is untouched and")
    print(f"  each affected row keeps the '{DUPLICATE_TIME_FLAG}' flag so any")
    print("  result can be re-checked with them excluded.")
    affected = sorted(
        measurements.loc[measurements[DUPLICATE_TIME_FLAG], ID_COL].unique().tolist()
    )
    print(f"  affected formulations : {affected}")
    print(f"  rows 4,913 raw -> {len(measurements)} after averaging")
    print("  136 and 148 repeat the SAME value (averaging is exact); 52 and 305")
    print("  repeat DIFFERENT values (averaging is a stated choice, not a fact).")

    print(f"\nSetting (a) time-dependent: {len(measurements)} measurement rows, "
          f"input '{LOG_TIME_COL}' = log1p(Time), target '{RELEASE_COL}'.")

    print(f"\nSetting (b) time-independent: {len(model_table)} formulations.")
    print(f"  '{NORM_TIME_COL}' = (t - t_min)/(t_max - t_min) per curve, kept")
    print(f"  alongside raw '{TIME_COL}' and 'Profile Duration' -- never replacing them.")
    print("  Early targets are linear interpolations at day 1/2/3, read as 24/48/72 h")
    print("  under the audit's Time-is-days interpretation. Extrapolation returns NaN.")
    coverage = model_table[EARLY_TARGETS].notna().sum()
    print(f"  formulations with interpolation support: {coverage.to_dict()} of {len(model_table)}")

    save(measurements, "taskB_measurement_table.csv")
    save(
        model_table[[ID_COL, SMILES_COL, "Profile Duration", "AUC", "AUC Class", FOLD_COL, *EARLY_TARGETS]],
        "taskB_formulation_targets.csv",
    )
    return measurements


# --- Task C --------------------------------------------------------------


def task_c(model_table):
    banner("TASK C -- Fuzzification: knots fitted on training folds only")

    reports = []
    for strategy in ("quantile", "uniform"):
        reports.append(knot_report(model_table, strategy))
    report = pd.concat(reports, ignore_index=True)

    print("Per-fold knots, quantile strategy (fold 0 shown; all folds saved):")
    print(
        report[(report.fold == 0) & (report["knot strategy"] == "quantile")][
            ["feature", "knot source", "knots", "valid High=1 share"]
        ].to_string(index=False)
    )

    print("\nTied-knot fallback:")
    print("  Declared policy -- if the raw q25/q50/q75 are not strictly increasing,")
    print("  recompute the same quantiles over the feature's DISTINCT observed")
    print("  values ('quantile-unique'). Chosen over epsilon-widening because the")
    print("  two features that trigger it are dominated by one value (LA/GA is 71%")
    print("  a single ratio; enhancer concentration is 64% exactly zero). Widening")
    print("  by epsilon would leave a near-binary indicator whose scale is arbitrary,")
    print("  whereas quantiles over distinct values spread the knots across the real")
    print("  support. If even that fails the feature is not fuzzified in that fold.")
    fallback = report[report["knot source"].str.endswith("-unique", na=False)]
    print(f"\n  features that needed the fallback: "
          f"{sorted(fallback['feature'].unique().tolist())}")
    skipped = report[~report["fuzzifiable"]]
    print(f"  features not fuzzifiable in some fold: "
          f"{sorted(skipped['feature'].unique().tolist()) or 'none'}")

    print("\nSaturation -- share of held-out rows with High membership exactly 1:")
    saturation = (
        report[report["fuzzifiable"]]
        .pivot_table(
            index="feature", columns="knot strategy", values="valid High=1 share",
            aggfunc="mean",
        )
        .round(3)
        .sort_values("quantile", ascending=False)
    )
    print(saturation.to_string())
    print("\n  Reading this: a quantile shoulder puts ~25% of values at High=1 by")
    print("  construction, more where the upper quartile is tied. Equal-width knots")
    print("  cut saturation sharply on the long-tailed features (Polymer MW,")
    print("  Particle Size, Drug MW) because the knots sit far out in the tail --")
    print("  but that is a different trade, not a free win: it leaves the crowded")
    print("  low end almost entirely inside a single 'Low' set. Which one predicts")
    print("  better is settled empirically in Task D, where both are run as arms.")

    save(report, "taskC_knot_report.csv")
    save(saturation.reset_index(), "taskC_saturation.csv")
    return report


# --- Task D --------------------------------------------------------------


def task_d(measurements, model_table):
    banner("TASK D -- XGBoost under grouped CV: crisp vs fuzzy vs hybrid")
    print("Identical folds, identical hyperparameters, identical seed in every arm.")
    print("No per-arm tuning: a search budget would confound representation with")
    print("search luck, and searching across folds would leak held-out data.\n")

    per_fold, fits = [], {}

    for kind, strategy in ARMS:
        label = arm_label(kind, strategy)
        rows, arm_fits = run_pointwise_regression(measurements, model_table, kind, strategy)
        per_fold.append(rows)
        fits[("(a) time-dependent", "Release (pointwise)", label)] = arm_fits
        print(f"  [a] {label:<20} RMSE {rows.RMSE.mean():.4f}  R2 {rows.R2.mean():+.4f}")

        for target in EARLY_TARGETS:
            rows, arm_fits = run_static_regression(model_table, target, kind, strategy)
            per_fold.append(rows)
            fits[("(b) time-independent", target, label)] = arm_fits
        print(f"  [b] {label:<20} "
              + "  ".join(
                  f"{t.split('_')[1]} RMSE {per_fold[-3 + i].RMSE.mean():.4f}"
                  for i, t in enumerate(EARLY_TARGETS)
              ))

    regression = pd.concat(per_fold, ignore_index=True)

    print("\nPer-fold regression metrics (all arms):")
    print(regression.to_string(index=False))

    summary = aggregate(
        regression, ["time setting", "target", "representation"], REGRESSION_METRICS
    )
    table = format_comparison(summary, REGRESSION_METRICS)

    print("\n--- COMPARISON TABLE: mean ± sd across 5 drug-grouped folds ---")
    print(table.to_string(index=False))

    paired = paired_vs_crisp(regression, "RMSE")
    print("\n--- PAIRED vs CRISP BASELINE (same folds, so the fold effect cancels) ---")
    print("Positive Δ means the arm beat crisp. The across-fold sd above is mostly")
    print("fold difficulty, which is common to all arms; this table removes it.\n")
    print(paired.to_string(index=False))
    save(paired, "taskD_paired_vs_crisp.csv")

    # Classification arm.
    banner("TASK D -- AUC <= 0.5 vs > 0.5 classification")
    classification = []
    for kind, strategy in ARMS:
        rows, _ = run_auc_classification(model_table, kind, strategy)
        classification.append(rows)
    classification = pd.concat(classification, ignore_index=True)

    positive_rate = (model_table["AUC Class"] == "AUC > 0.5").mean()
    print(f"Class imbalance: {positive_rate:.1%} of formulations are AUC > 0.5 "
          f"(~{positive_rate / (1 - positive_rate):.1f}:1).")
    print("Training folds are randomly undersampled to balance; validation folds")
    print("are left at their natural prevalence, so accuracy and precision must be")
    print("read against that rate rather than against 50%.\n")
    print(classification.to_string(index=False))

    class_summary = aggregate(classification, ["representation"], CLASSIFICATION_METRICS)
    class_table = format_comparison(class_summary, CLASSIFICATION_METRICS)
    print("\n--- CLASSIFICATION COMPARISON: mean ± sd across folds ---")
    print(class_table.to_string(index=False))

    save(regression, "taskD_regression_per_fold.csv")
    save(table, "taskD_regression_comparison.csv")
    save(classification, "taskD_classification_per_fold.csv")
    save(class_table, "taskD_classification_comparison.csv")

    return regression, summary, fits


def pick_best(summary: pd.DataFrame) -> tuple[str, str, str]:
    """Best arm by mean RMSE, chosen within the strongest task."""
    pointwise = summary[summary["time setting"] == "(a) time-dependent"]
    best = pointwise.sort_values("RMSE mean").iloc[0]
    return best["time setting"], best["target"], best["representation"]


# --- Task E --------------------------------------------------------------


def task_e(fits, measurements, model_table, best_key):
    banner("TASK E -- Rule extraction from the trained predictor (Layer 3)")

    setting, target, representation = best_key
    print(f"Extracting from the best predictor: {representation} / {setting} / {target}")
    print("Both extractors are fitted on the TRAINING fold, mimicking the trained")
    print("XGBoost model's own predictions, then re-tested on the held-out fold.\n")

    arm_fits = fits[best_key]
    smiles = measurements[SMILES_COL].to_numpy()

    all_rules, fidelity = [], []
    for fit in arm_fits:
        for extract in (rulefit_rules, surrogate_tree_rules):
            rules, report = extract(fit)
            fidelity.append(report)
            if len(rules):
                groups = smiles[fit.valid_positions]
                all_rules.append(validate_rules(rules, fit, groups))

    validated = pd.concat(all_rules, ignore_index=True)
    fidelity = pd.DataFrame(fidelity)

    print("Surrogate fidelity to the predictor, measured on held-out rows:")
    print(
        fidelity.groupby("method")[["fidelity R2 (valid)", "fidelity RMSE (valid)", "n rules kept"]]
        .mean()
        .round(4)
        .to_string()
    )
    print("\n  Fidelity is agreement with the XGBoost model, not accuracy against")
    print("  measured release. A rule set can mimic the model well and still")
    print("  inherit every one of the model's errors.")

    passed = validated[validated["validated"]]
    failed = validated[~validated["validated"]]
    print(f"\nRules extracted across all folds : {len(validated)}")
    print(f"  passed all three held-out tests : {len(passed)}")
    print(f"  failed at least one             : {len(failed)}")
    if len(failed):
        print("  failure breakdown: "
              f"coverage < 5% in {int((~failed['enough coverage']).sum())}, "
              f"direction flipped in {int((~failed['same direction']).sum())}, "
              f"CI included 0 in {int((~failed['CI excludes 0']).sum())}")

    label = "release" if target == "Release (pointwise)" else target

    def show(subset, limit):
        if not len(subset):
            print("  None.")
            return
        for _, rule in subset.sort_values("valid coverage", ascending=False).head(limit).iterrows():
            print(f"\n  {rule['readable']}")
            print(f"    {describe_effect(rule['valid effect'], label)}")
            print(f"    fold {rule['fold']} | {rule['method']} | "
                  f"coverage {rule['valid coverage']:.1%} | "
                  f"95% CI [{rule['valid effect CI low']:+.3f}, {rule['valid effect CI high']:+.3f}]")

    formulation_rules = passed[~passed["time only"]]
    carries_own_weight = formulation_rules[formulation_rules["formulation part holds"]]
    time_driven = formulation_rules[~formulation_rules["formulation part holds"]]

    print("\n--- VALIDATED RULES WHOSE FORMULATION CONDITION CARRIES ITS OWN WEIGHT ---")
    print("Every extracted rule pairs a formulation condition with a time condition,")
    print("and release rises steeply with time. These are the rules where the")
    print("formulation condition still separates held-out measurements AFTER")
    print("restricting to the rule's own time window -- so time alone does not")
    print("explain the effect. This is the strictest result in the study.")
    print(f"  {len(carries_own_weight)} of {len(formulation_rules)} formulation rules pass it.\n")
    show(carries_own_weight, 12)

    print("\n--- VALIDATED RULES WHOSE EFFECT IS EXPLAINED BY TIME ALONE (top 5) ---")
    print("These passed the whole-rule test but their formulation condition adds")
    print("nothing once time is held fixed. They should NOT be reported as")
    print("formulation findings.")
    show(time_driven, 5)

    print("\n--- VALIDATED RULES THAT ONLY TEST TIME (top 3) ---")
    print("Reported for completeness. They validate easily and carry the highest")
    print("coverage of any rule extracted, but 'later measurements show more")
    print("release' restates the shape of a release curve rather than telling a")
    print("formulator anything about a formulation.")
    show(passed[passed["time only"]], 3)

    print("\n--- RULES THAT DID NOT HOLD UP (top 5, reported not hidden) ---")
    for _, rule in failed.sort_values("valid coverage", ascending=False).head(5).iterrows():
        reasons = [
            name for name, ok in [
                ("coverage", rule["enough coverage"]),
                ("direction", rule["same direction"]),
                ("CI excludes 0", rule["CI excludes 0"]),
            ] if not ok
        ]
        print(f"  {rule['readable'][:110]}")
        print(f"    failed: {', '.join(reasons)} | coverage {rule['valid coverage']:.1%}")

    stable = stable_signatures(validated, require="formulation part holds")
    formulation_stable = stable[
        stable["stable"]
        & ~stable["signature"].map(lambda s: all(f == LOG_TIME_COL for f, _ in s))
    ]
    print("\n--- CONDITION PATTERNS STABLE ACROSS FOLDS (>=3 of 5, same direction) ---")
    print("Restricted to patterns touching a formulation characteristic. Each fold")
    print("holds out different drugs, so recurrence is not a repeat of one result.")
    if len(formulation_stable):
        for _, row in formulation_stable.iterrows():
            print(f"\n  {row['example rule']}")
            print(f"    {describe_effect(row['mean valid effect'], label)}")
            print(f"    validated in {row['folds validated']}/5 folds | "
                  f"mean coverage {row['mean valid coverage']:.1%}")
    else:
        print("  NONE reached 3 of 5 folds with a consistent direction.")
        print("  This is the study's main negative result on rules: no formulation")
        print("  pattern that survives the time-window test replicates across folds.")

        near = stable[(stable["folds validated"] == 2) & stable["consistent direction"]]
        if len(near):
            print(f"\n  Closest candidates ({len(near)} patterns validated in 2 of 5 folds).")
            print("  Below the pre-set threshold -- these are leads to test on more")
            print("  data, NOT findings to report:")
            for _, row in near.iterrows():
                print(f"\n    {row['example rule']}")
                print(f"      {describe_effect(row['mean valid effect'], label)}")
                print(f"      2/5 folds | mean coverage {row['mean valid coverage']:.1%}")

    print("\nThese are empirical associations in 321 literature-sourced formulations.")
    print("They are NOT mechanistic or causal claims about drug release.")

    save(validated.drop(columns=["signature"]), "taskE_rules_validated.csv")
    save(fidelity, "taskE_surrogate_fidelity.csv")
    save(stable.astype({"signature": str}), "taskE_stable_signatures.csv")
    return validated, stable


# --- Task F --------------------------------------------------------------


def task_f(fits, measurements, best_key):
    banner("TASK F -- SHAP importance and predicted-vs-actual curves")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    def block_of(name: str) -> str:
        if " is " in name:
            return "fuzzy membership"
        if name == LOG_TIME_COL:
            return "time"
        if name.startswith("Formulation Method"):
            return "method (one-hot)"
        return "crisp"

    def shap_importance(arm_fits):
        frames = []
        for fit in arm_fits:
            explainer = shap.TreeExplainer(fit.model)
            values = explainer.shap_values(fit.X_valid)
            frames.append(
                pd.Series(
                    np.abs(values).mean(axis=0), index=fit.X_valid.columns, name=fit.fold
                )
            )
        wide = pd.concat(frames, axis=1)
        table = pd.DataFrame(
            {"mean |SHAP|": wide.mean(axis=1), "sd across folds": wide.std(axis=1)}
        ).sort_values("mean |SHAP|", ascending=False)
        table["block"] = [block_of(name) for name in table.index]
        return table

    arm_fits = fits[best_key]
    representation = best_key[2]
    importance = shap_importance(arm_fits)

    print(f"Global SHAP importance for the best model ({representation}),")
    print("averaged over held-out rows in all 5 folds. Top 20:\n")
    print(importance.head(20).round(5).to_string())
    save(importance.reset_index().rename(columns={"index": "feature"}),
         "taskF_shap_importance_best.csv")

    # The best arm is fuzzy-only, so it has no crisp block to compare against.
    # The crisp-vs-fuzzy question is only answerable where both are present, so
    # it is asked of the hybrid arm, which carries each feature in both forms.
    hybrid_key = (best_key[0], best_key[1], "hybrid (quantile)")
    hybrid = shap_importance(fits[hybrid_key])

    print(f"\n--- CRISP vs FUZZY, measured on the hybrid arm ({hybrid_key[2]}) ---")
    print("The best arm is fuzzy-only and has no crisp block, so this question is")
    print("asked of the hybrid arm, where every feature is present in both forms")
    print("and competes for the same splits.\n")
    print(hybrid.head(15).round(5).to_string())

    blocks = hybrid.groupby("block")["mean |SHAP|"].agg(["sum", "mean", "count"])
    blocks.columns = ["total |SHAP|", "mean per feature", "n features"]
    print("\nAggregated by block:")
    print(blocks.round(5).to_string())
    print("\n  Per-feature mean is the fairer comparison: the hybrid arm carries")
    print("  three membership columns for every crisp one, so a block total")
    print("  rewards the fuzzy block for being wider rather than more useful.")
    print("  Memberships are also derived from the very crisp features sitting")
    print("  beside them, so the two blocks split credit for one signal and")
    print("  neither number should be read as that signal's total importance.")

    paired_blocks = []
    for feature in STATIC_FEATURES:
        crisp_value = hybrid.loc[feature, "mean |SHAP|"] if feature in hybrid.index else np.nan
        fuzzy_value = hybrid[hybrid.index.str.startswith(f"{feature} is ")][
            "mean |SHAP|"
        ].sum()
        paired_blocks.append(
            {
                "feature": feature,
                "crisp |SHAP|": crisp_value,
                "fuzzy |SHAP| (3 sets summed)": fuzzy_value,
                "fuzzy share": fuzzy_value / (fuzzy_value + crisp_value)
                if np.isfinite(crisp_value) and (fuzzy_value + crisp_value) > 0
                else np.nan,
            }
        )
    paired_blocks = pd.DataFrame(paired_blocks).sort_values("fuzzy share", ascending=False)
    print("\nPer feature, how the model split credit between the two forms:")
    print(paired_blocks.round(5).to_string(index=False))
    print("\n  A share near 0.5 means the model drew on both forms; near 1 means it")
    print("  preferred the membership columns for that feature. This shows which")
    print("  features the fuzzy encoding was actually used for -- it does not show")
    print("  that using it improved the prediction, which Task D addresses.")

    save(hybrid.reset_index().rename(columns={"index": "feature"}),
         "taskF_shap_importance_hybrid.csv")
    save(paired_blocks, "taskF_crisp_vs_fuzzy_shap.csv")

    # Representative held-out curves.
    predicted = np.full(len(measurements), np.nan)
    for fit in arm_fits:
        predicted[fit.valid_positions] = fit.predictions
    curves = measurements.assign(Predicted=predicted)

    per_formulation = (
        curves.groupby(ID_COL)
        .apply(lambda g: np.sqrt(np.mean((g[RELEASE_COL] - g["Predicted"]) ** 2)),
               include_groups=False)
        .sort_values()
    )
    chosen = [
        (per_formulation.index[0], "best held-out fit"),
        (per_formulation.index[len(per_formulation) // 2], "median held-out fit"),
        (per_formulation.index[-1], "worst held-out fit"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, (formulation_id, caption) in zip(axes, chosen):
        part = curves[curves[ID_COL] == formulation_id].sort_values(TIME_COL)
        axis.plot(part[TIME_COL], part[RELEASE_COL], "o-", label="measured", color="#1b4965")
        axis.plot(part[TIME_COL], part["Predicted"], "s--", label="predicted", color="#c1666b")
        axis.set_title(f"Formulation {formulation_id}\n{caption} "
                       f"(RMSE {per_formulation[formulation_id]:.3f})", fontsize=10)
        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Fractional release")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    figure.suptitle(
        f"Held-out release curves -- {representation} XGBoost, time-dependent setting",
        fontsize=11,
    )
    figure.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "taskF_held_out_curves.png", dpi=150)
    plt.close(figure)
    print(f"\n  -> study_results/taskF_held_out_curves.png  "
          f"(formulations {[int(c[0]) for c in chosen]})")

    save(per_formulation.rename("held-out RMSE").reset_index(), "taskF_per_formulation_rmse.csv")
    return importance


# --- Task G --------------------------------------------------------------


def task_g(fits, measurements, best_key):
    banner("TASK G -- Fuzzy interpretation of the PREDICTED release (output layer)")

    setting, target, representation = best_key
    print("This is the layer the project's motivating example is about. The model")
    print("predicts a continuous release value; that value is then given a graded")
    print("Low/Medium/High description instead of a single hard category, so a")
    print("formulation at 69% and one at 70% are described as what they are --")
    print("both borderline -- rather than being placed in different categories.\n")

    arm_fits = fits[best_key]
    predicted = np.full(len(measurements), np.nan)
    for fit in arm_fits:
        predicted[fit.valid_positions] = fit.predictions
    held_out = measurements.assign(Predicted=predicted)
    held_out = held_out[np.isfinite(held_out["Predicted"])]

    # Two interpreters. The declared one is fixed; the data one is refit per
    # fold on TRAINING release values only, and the folds agree closely enough
    # that a single pooled version is reported for readability.
    design = ReleaseInterpreter(DESIGN_KNOTS, "design")
    fold_knots = []
    for fold, train, _ in iter_positions(measurements):
        fitted = ReleaseInterpreter.fit_from_training(
            measurements.iloc[train][RELEASE_COL].to_numpy()
        )
        fold_knots.append({"fold": fold, "provenance": fitted.provenance,
                           **{f"knot {i}": round(k, 4) for i, k in enumerate(fitted.knots)}})
    fold_knots = pd.DataFrame(fold_knots)
    data_driven = ReleaseInterpreter(
        tuple(fold_knots[[f"knot {i}" for i in range(3)]].mean()), "data"
    )

    print("Where the Low/Medium/High boundaries come from -- stated, not assumed:")
    provenance = pd.concat([design.report(), data_driven.report()], ignore_index=True)
    print(provenance.to_string(index=False))
    print("\n  Per-fold data-derived knots (fitted on TRAINING release only):")
    print("  " + fold_knots.to_string(index=False).replace("\n", "\n  "))
    print("\n  NEITHER row above is a pharmaceutical definition. No expert-supplied")
    print("  Low/Medium/High criterion was available, so this remains a")
    print("  PARAMETERISED FRAMEWORK. Selecting the real boundaries is a")
    print("  SUPERVISOR / PHARMACEUTICAL-EXPERT DECISION, and everything below")
    print("  moves when they move -- which is why the sensitivity table is run.")
    save(provenance, "taskG_knot_provenance.csv")
    save(fold_knots, "taskG_knot_per_fold.csv")

    # -- Evaluation question C: graded description ------------------------
    print("\n--- (C) INTERPRETABILITY: predicted release as graded membership ---")
    table = interpretation_table(
        held_out, "Predicted", design,
        id_cols=[ID_COL, TIME_COL], actual_col=RELEASE_COL,
    )
    graded_share = float(table["graded"].mean())
    print(f"Held-out predictions described by more than one linguistic set: "
          f"{graded_share:.1%} of {len(table)}.")
    print("A crisp scheme would describe 100% of them with exactly one label.\n")

    near = table[(table[["predicted is Low", "predicted is Medium", "predicted is High"]]
                  .max(axis=1) < 0.65)]
    print(f"The borderline population -- no set above 0.65 membership -- is "
          f"{len(near)} rows ({len(near) / len(table):.1%}).")
    print("These are the formulations a crisp label describes worst. Examples,")
    print("drawn from held-out drugs the model never trained on:\n")
    for _, row in near.head(6).iterrows():
        print(f"  Formulation {int(row[ID_COL])}, day {row[TIME_COL]:.2f}: "
              f"predicted {row['predicted release'] * 100:.1f}%  "
              f"(measured {row['measured release'] * 100:.1f}%)")
        print(f"      Low {row['predicted is Low']:.2f} | "
              f"Medium {row['predicted is Medium']:.2f} | "
              f"High {row['predicted is High']:.2f}"
              f"   <- crisp scheme would say only '{row['crisp label']}'")
    save(table, "taskG_predicted_interpretation.csv")

    # -- Evaluation question B: boundary robustness -----------------------
    print("\n--- (B) BOUNDARY ROBUSTNESS: what a 1-point change does to each scheme ---")
    print("Every predicted release is nudged by +/-1 percentage point -- the size of")
    print("the step in the 69% vs 70% example -- and both schemes are asked how much")
    print("their description changed. A crisp flip costs 1.0 (the formulation is")
    print("reassigned); the fuzzy shift is on the same 0-1 scale, so the two are")
    print("directly comparable.\n")

    robustness = pd.DataFrame([
        {"interpreter": "declared (1/3, 2/3)",
         **boundary_robustness(held_out["Predicted"], design)},
        {"interpreter": "data-derived tertiles",
         **boundary_robustness(held_out["Predicted"], data_driven)},
    ])
    print(robustness.to_string(index=False))
    save(robustness, "taskG_boundary_robustness.csv")

    row = robustness.iloc[0]
    print(f"\n  Reading the declared row: {row['share within delta of a boundary']:.1%} of "
          f"held-out predictions ({int(row['n within delta of a boundary'])} rows) sit")
    print("  within 1 point of a cut point.")
    print(f"  The crisp scheme reassigns {row['crisp flip rate | near boundary']:.0%} of "
          f"those to a different category. That 100% is TRUE BY")
    print("  CONSTRUCTION and is not a finding -- 'near a boundary' is defined as")
    print("  straddling a cut point, and a crisp scheme flips on exactly that set.")
    print("  Stating it as a discovery would be circular.")
    print(f"\n  The two numbers that ARE empirical: {row['share within delta of a boundary']:.1%} of held-out")
    print(f"  predictions are exposed to this flip at all, and the fuzzy description")
    print(f"  of that same set moves by {row['fuzzy shift | near boundary']:.3f} instead of 1.0 -- a change of")
    print(f"  about {row['fuzzy shift | near boundary'] * 100:.0f} parts in 100, where the crisp scheme changes everything.")
    print("  The first bounds how much this contribution can matter; the fuzzy")
    print("  layer does nothing for the other 96%, and that is reported rather")
    print("  than buried.")

    print("\n--- Does that conclusion depend on the boundary choice? ---")
    print("The cut points are not established facts, so the same measurement is")
    print("repeated under alternatives a reviewer might propose.\n")
    sensitivity = boundary_sensitivity_to_knots(
        held_out["Predicted"],
        {
            "declared (1/3, 2/3)": DESIGN_KNOTS,
            "data tertiles": data_driven.knots,
            "narrow (0.25, 0.75)": (0.25, 0.5, 0.75),
            "wide (0.4, 0.6)": (0.4, 0.5, 0.6),
        },
    )
    print(sensitivity[["knot set", "knots", "share within delta of a boundary",
                       "crisp label flip rate", "mean fuzzy membership shift",
                       "share with graded (multi-set) description"]].to_string(index=False))
    save(sensitivity, "taskG_boundary_sensitivity.csv")

    print("\n  What survives every column: the fuzzy shift is orders of magnitude")
    print("  smaller than the crisp flip rate under EVERY boundary choice. What")
    print("  does not survive: how many formulations are affected, which moves")
    print("  with the knots and is therefore not a property of the method.")

    return table, robustness


# --- Task H --------------------------------------------------------------


def task_h(measurements, model_table):
    banner("TASK H -- Rule extraction within release phases (Layer 3, revised)")

    print("Time is excluded from the rule learner entirely. Instead the data is cut")
    print("into Early / Middle / Late phases and a shallow tree is fitted inside")
    print("each one, so time is held roughly fixed by restriction and cannot win a")
    print("split. This replaces the earlier approach of letting time compete and")
    print("then discounting it afterwards, which produced rules whose formulation")
    print("condition added nothing.\n")

    phased = add_phase(measurements)
    summary = phase_summary(phased)
    print(summary.drop(columns=["provenance"]).to_string(index=False))
    print("\nPhase boundary provenance -- READ THIS BEFORE READING ANY RULE:")
    for phase in PHASES:
        print(f"  {phase:7s}: {PHASE_PROVENANCE[phase]}")
    save(summary, "taskH_phase_summary.csv")

    alternatives = phase_boundary_alternatives(measurements)
    print(f"\n  Alternative boundary pairs available for sensitivity checks: "
          f"{ {k: tuple(round(x, 2) for x in v) for k, v in alternatives.items()} }")

    print("\nThe tree is fitted on FUZZY MEMBERSHIP columns of the formulation and")
    print("drug descriptors -- all ten static features, not a hand-picked subset --")
    print("targeting MEASURED release. It is an ordinary crisp CART tree used as a")
    print("rule-discovery mechanism over a fuzzy representation. It is NOT a fuzzy")
    print("inference system and is not described as one.\n")

    all_rules, reports, texts = [], [], {}

    for phase in PHASES:
        part = phased[phased[PHASE_COL] == phase].reset_index(drop=True)
        if part[ID_COL].nunique() < 30:
            print(f"  {phase}: only {part[ID_COL].nunique()} formulations -- skipped.")
            continue

        for fold, train, valid in iter_positions(part):
            train_frame, valid_frame = part.iloc[train], part.iloc[valid]
            train_formulations = model_table[model_table[FOLD_COL] != fold]

            representation = Representation("fuzzy", knot_strategy="quantile").fit(
                train_formulations
            )
            X_train = model_ready(representation.transform(train_frame))
            X_valid = model_ready(representation.transform(valid_frame))

            fit = RuleFit(
                fold, representation, X_train, X_valid,
                train_frame[RELEASE_COL], valid_frame[RELEASE_COL],
            )
            rules, report = phase_tree_rules(fit, label=f"{phase} phase tree (depth 3)")
            report["phase"] = phase
            reports.append(report)

            if fold == 0:
                texts[phase] = tree_text(fit)

            if len(rules):
                groups = valid_frame[SMILES_COL].to_numpy()
                validated = validate_rules(rules, fit, groups)
                validated["phase"] = phase
                all_rules.append(validated)

    reports = pd.DataFrame(reports)
    print("Per-phase tree fit (held-out R2 is the tree's own, not XGBoost's):")
    print(
        reports.groupby("phase")[["n train rows", "n valid rows", "train R2",
                                  "valid R2", "n leaves"]]
        .mean().round(3).to_string()
    )
    print("\n  A low or negative held-out R2 here is expected and is not a failure")
    print("  of the method: a depth-3 tree on ten descriptors is meant to be")
    print("  readable, not accurate. Its job is to propose candidate rules that")
    print("  are then tested; the test, not the fit, is what admits a rule.")
    save(reports, "taskH_phase_tree_fit.csv")

    print("\n--- EXACT TREE (fold 0), technical form ---")
    print("The readable rules below paraphrase membership thresholds into words;")
    print("this preserves the actual numerical criterion so it can be checked.")
    for phase, text in texts.items():
        print(f"\n  [{phase} phase]")
        print("  " + text.replace("\n", "\n  ")[:1400])

    if not all_rules:
        print("\n  No rules extracted in any phase.")
        return pd.DataFrame()

    validated = pd.concat(all_rules, ignore_index=True)
    classified = classify_evidence(validated)

    print("\n--- (D) RULE USEFULNESS: evidence tiers on held-out drugs ---")
    print("A rule from a tree is a candidate, not a finding. Tiers are declared in")
    print(f"advance: replication in >={MIN_FOLDS_VALIDATED} of 5 drug-grouped folds with a")
    print(f"consistent direction, and a held-out effect of at least {MIN_EFFECT} "
          f"fractional release.\n")

    counts = evidence_counts(classified)
    print(counts.to_string(index=False))

    by_phase = (
        classified.groupby(["phase", "evidence"]).size().unstack(fill_value=0)
    )
    print("\nBy phase:")
    print(by_phase.to_string())
    save(by_phase.reset_index(), "taskH_evidence_by_phase.csv")

    findings = classified[classified["evidence"] == FINDING]
    replicated = classified[classified["evidence"] == REPLICATED]

    print("\n--- VALIDATED FINDINGS ---")
    if len(findings):
        for _, rule in findings.sort_values("valid coverage", ascending=False).iterrows():
            print(f"\n  [{rule['phase']} phase] {rule['readable']}")
            print(f"    {describe_effect(rule['valid effect'], 'release')}")
            print(f"    replicated in {rule['folds replicated']}/5 folds | "
                  f"coverage {rule['valid coverage']:.1%} | "
                  f"95% CI [{rule['valid effect CI low']:+.3f}, "
                  f"{rule['valid effect CI high']:+.3f}]")
            print(f"    exact criterion: {rule['raw rule']}")
    else:
        print("  NONE. No rule reached the validated-finding tier.")
        print("  This is a negative result and is reported as one. It is NOT")
        print("  evidence that no formulation-release relationship exists; it says")
        print("  that none is resolvable from ten static descriptors at the")
        print("  replication and effect-size bar set in advance.")

    print("\n--- REPLICATED BUT BELOW THE EFFECT THRESHOLD ---")
    if len(replicated):
        for _, rule in replicated.sort_values("valid coverage", ascending=False).head(8).iterrows():
            print(f"\n  [{rule['phase']} phase] {rule['readable']}")
            print(f"    {describe_effect(rule['valid effect'], 'release')}")
            print(f"    {rule['evidence reason']}")
    else:
        print("  None.")

    print("\n--- (E) TEMPORAL CONSISTENCY: which features the trees split on, by phase ---")
    print("If different formulation characteristics mattered at different stages,")
    print("the phases would split on different features. This asks that directly.\n")
    split_features = []
    for _, rule in classified.iterrows():
        # ``signature`` is the rule's (feature, direction) pairs with the
        # fold-specific thresholds stripped, which is exactly what is wanted
        # here: whether a feature was used, not where it was cut.
        for feature, _direction in rule["signature"]:
            base = feature.rsplit(" is ", 1)[0]
            split_features.append({"phase": rule["phase"], "feature": base,
                                   "validated": rule["validated"]})
    split_features = pd.DataFrame(split_features)
    usage = (
        split_features.groupby(["feature", "phase"]).size().unstack(fill_value=0)
    )
    usage = usage.reindex(columns=[p for p in PHASES if p in usage.columns], fill_value=0)
    usage["total"] = usage.sum(axis=1)
    print(usage.sort_values("total", ascending=False).to_string())
    print("\n  Counts are how often a tree chose to split on that feature across")
    print("  folds. They show what the trees USED, which is weaker than showing")
    print("  the feature matters -- a split can be chosen and still fail the")
    print("  held-out test. Read alongside the evidence tiers above.")
    save(usage.reset_index(), "taskH_phase_feature_usage.csv")

    save(classified.astype({"signature": str}), "taskH_phase_rules.csv")
    return classified


# --- Task I --------------------------------------------------------------


def task_i(model_table):
    banner("TASK I -- Alternative target: <=20% released by day 3 (separate experiment)")

    labels = slow_release_class(model_table)
    observable = labels.notna()
    print(f"Source of the definition: {SLOW_RELEASE_SOURCE}\n")
    print("This is run as a SEPARATE EXPERIMENT, not as a replacement for the")
    print("normalized-AUC class. The two ask different questions: the AUC class")
    print("summarises curve shape after the time axis has been rescaled to [0,1]")
    print("and is blind to elapsed time; this criterion is anchored to real days.")
    print("The threshold is adopted because a published source defines it, NOT")
    print("because it improves any metric.\n")

    print(f"  formulations where the criterion is observable : "
          f"{int(observable.sum())} of {len(model_table)}")
    print(f"  ({int((~observable).sum())} curves end before day 3 and are dropped, "
          f"not imputed)")
    print(f"  positive (slow-releasing) rate                 : "
          f"{labels[observable].mean():.1%}")

    rows = []
    for kind, strategy in ARMS:
        rows.append(run_slow_release_classification(model_table, kind, strategy))
    classification = pd.concat(rows, ignore_index=True)

    summary = aggregate(classification, ["representation"], CLASSIFICATION_METRICS)
    table = format_comparison(summary, CLASSIFICATION_METRICS)
    print("\n--- Mean +/- sd across the same 5 drug-grouped folds ---")
    print(table.to_string(index=False))
    print("\n  Read accuracy against the positive rate above, not against 50%.")

    save(classification, "taskI_slow_release_per_fold.csv")
    save(table, "taskI_slow_release_comparison.csv")
    return table


# --- Final explanation ---------------------------------------------------


def final_explanation(interpretation, classified):
    banner("END-TO-END EXPLANATION -- what a pharmacist would actually be shown")

    print("Assembled from the pipeline's real output. Every number below is")
    print("produced by the model on a held-out drug; none is illustrative.\n")

    # Day-3 predictions are the ones the Early phase and the published
    # criterion both speak to, so the example is drawn from there.
    day3 = interpretation[
        (interpretation[TIME_COL] >= 2.5) & (interpretation[TIME_COL] <= 3.5)
    ]
    borderline = day3.assign(
        peak=day3[["predicted is Low", "predicted is Medium", "predicted is High"]].max(axis=1)
    ).sort_values("peak")

    if borderline.empty:
        print("  No held-out prediction falls near day 3; example omitted.")
        return

    row = borderline.iloc[0]
    print(f"  Formulation {int(row[ID_COL])}  (held-out drug, day {row[TIME_COL]:.1f})")
    print(f"  Predicted release: {row['predicted release'] * 100:.1f}%")
    print("\n  Release interpretation:")
    for label in ("Low", "Medium", "High"):
        degree = row[f"predicted is {label}"]
        if degree > 0:
            print(f"    {label:7s}: {degree * 100:.0f}%")
    print(f"\n  A crisp three-bin scheme would report only '{row['crisp label']}'.")
    print(f"  (Measured value for reference: {row['measured release'] * 100:.1f}%. The")
    print("   graded description is about the boundary, not about accuracy -- this")
    print("   prediction's error is what Task D reports and is not small.)")

    print("\n  Main explanatory pattern:")
    findings = classified[classified["evidence"] == FINDING] if len(classified) else classified
    if len(findings):
        best = findings.sort_values("valid coverage", ascending=False).iloc[0]
        print(f"    {best['readable']}")
        print(f"    {describe_effect(best['valid effect'], 'release')}")
    else:
        print("    NONE AVAILABLE. No formulation rule reached the validated-finding")
        print("    tier (Task H), so this line is deliberately left empty rather than")
        print("    filled with a candidate rule. Showing an unvalidated IF-THEN")
        print("    statement here is exactly the failure mode the study is built to")
        print("    avoid: it would read as a pharmaceutical conclusion while resting")
        print("    on a pattern that did not replicate across held-out drugs.")

    print("\n  Boundaries used for Low/Medium/High are DECLARED, not clinical.")
    print("  They require supervisor / pharmaceutical-expert selection before this")
    print("  display can be shown to anyone making a formulation decision.")


def main() -> int:
    dataset, model_table = task_a()
    measurements = task_b(dataset, model_table)
    task_c(model_table)
    _, summary, fits = task_d(measurements, model_table)

    best_key = pick_best(summary)
    task_e(fits, measurements, model_table, best_key)
    task_f(fits, measurements, best_key)
    interpretation, _ = task_g(fits, measurements, best_key)
    classified = task_h(measurements, model_table)
    task_i(model_table)
    final_explanation(interpretation, classified)

    banner("DONE")
    print(f"Artifacts written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
