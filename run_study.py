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
    summarize_smiles_folds,
    verify_no_smiles_leakage,
)
from fuzzypharma.rules import (
    describe_effect,
    rulefit_rules,
    stable_signatures,
    surrogate_tree_rules,
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
    paired_vs_crisp,
    run_auc_classification,
    run_pointwise_regression,
    run_static_regression,
)

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


def main() -> int:
    dataset, model_table = task_a()
    measurements = task_b(dataset, model_table)
    task_c(model_table)
    _, summary, fits = task_d(measurements, model_table)

    best_key = pick_best(summary)
    task_e(fits, measurements, model_table, best_key)
    task_f(fits, measurements, best_key)

    banner("DONE")
    print(f"Artifacts written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
