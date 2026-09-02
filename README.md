# Explainable Fuzzy Machine Learning for Long-Acting Injectables
**Read the Full Report:** [Download Report PDF](./Interpretable_ML_PLGA_Release_Report.pdf)
> **Project status: Under construction.** This repository contains preliminary research materials, exploratory analysis, and an initial fuzzification demonstration. The methodology and modeling pipeline are still being developed and may change.

This research project explores whether fuzzy feature representations can improve the prediction and interpretation of drug-release behavior from polymer-based long-acting injectable formulations. It focuses on drug-loaded PLGA microparticles and builds on an explainable machine-learning benchmark for early release, release-profile classification, and complete release-curve prediction.

## Implemented workflow

The current work in progress provides:

1. **Dataset preparation:** initial and processed workbooks containing 321 formulations and 4,913 release measurements. The processed data combines formulation-level material properties with time-series release values.
2. **Exploratory analysis:** `data_visualization.ipynb` examines release curves, sampling duration, data quality, feature distributions, correlations, candidate release drivers, formulation groups, and drug representation.
3. **Fuzzy feature demonstration:** `fuzzification_polymer_particlesize.ipynb` applies quantile-based triangular membership functions to `Polymer MW` and `Particle Size`, producing Low, Medium, and High membership features.
4. **Rule-validation study:** `validation_of_8_predefined_fuzzy_rules.ipynb` tests whether the eight predefined fuzzy rules generalize to unseen drugs, using 5-fold `GroupKFold` grouped on exact drug SMILES with explicit control for the time confound. See **[`RULE_VALIDATION_STUDY.md`](RULE_VALIDATION_STUDY.md)** for the full write-up.

The broader crisp, fuzzy-only, and hybrid modeling experiments described in `proposal.md` are proposed work; a complete model-training pipeline is not included in this checkout.

### Rule-validation headline

Of the eight predefined rules, **five of the source tree's seven splits are on `Time`**, so only four rules (R3–R6) carry a formulation condition — and those encode just **two distinct claims**, each stated twice from opposite sides.

| Outcome | Rules | Finding |
|---|---|---|
| **Supported** | R5, R6 | Lower-MW PLGA (≤ 21 kDa) releases more within a fixed time window — direction reproduced in **5 of 5** held-out folds (mean +0.215) |
| **Weak / borderline** | R3, R4 | Smaller particles (≤ 9.505 µm) release more — direction held in all **3 evaluable** folds, but 2 folds had too few held-out drugs to test and no fold's CI excludes zero |
| **Unsupported** | R1, R2, R7, R8 | Pure `Time` statements; large uncontrolled effects shrink 53–87% and lose their sign once time is controlled |

Negative results are reported as found. Full method, caveats, and limitations are in [`RULE_VALIDATION_STUDY.md`](RULE_VALIDATION_STUDY.md).

## Repository structure

```text
.
├── data_visualization.ipynb                      # Exploratory dataset analysis
├── fuzzification_polymer_particlesize.ipynb      # Two-feature fuzzification example
├── validation_of_8_predefined_fuzzy_rules.ipynb  # Rule-validation study (self-contained)
├── RULE_VALIDATION_STUDY.md                      # Full write-up of that study
├── rule_validation_summary.csv                   # Its final results table
├── export_evidence.py                            # Exports raw notebook outputs as-is
├── export_figures.py                             # Rebuilds the presentation figures
├── figures/rule_validation/
│   ├── notebook_evidence/                        # Outputs exactly as the notebook produced them
│   └── presentation/                             # Slide-ready versions of the same results
├── mp_dataset_initial.xlsx                       # Source dataset and metadata
├── mp_dataset_processed.xlsx                     # Numeric modeling dataset
├── related_work_table.xlsx                       # Related-literature summary
├── proposal.md                                   # Research context and proposed methodology
└── main.pdf                                      # Benchmark research paper
```
