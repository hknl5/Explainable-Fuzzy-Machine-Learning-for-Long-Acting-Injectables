# Explainable Fuzzy Machine Learning for Long-Acting Injectables

This research project explores whether fuzzy feature representations can improve the prediction and interpretation of drug-release behavior from polymer-based long-acting injectable formulations. It focuses on drug-loaded PLGA microparticles and builds on an explainable machine-learning benchmark for early release, release-profile classification, and complete release-curve prediction.

## Implemented workflow

The repository currently provides:

1. **Dataset preparation:** initial and processed workbooks containing 321 formulations and 4,913 release measurements. The processed data combines formulation-level material properties with time-series release values.
2. **Exploratory analysis:** `data_visualization.ipynb` examines release curves, sampling duration, data quality, feature distributions, correlations, candidate release drivers, formulation groups, and drug representation.
3. **Fuzzy feature demonstration:** `fuzzification_polymer_particlesize.ipynb` applies quantile-based triangular membership functions to `Polymer MW` and `Particle Size`, producing Low, Medium, and High membership features.

The broader crisp, fuzzy-only, and hybrid modeling experiments described in `proposal.md` are proposed work; a complete model-training pipeline is not included in this checkout.

## Repository structure

```text
.
├── data_visualization.ipynb                 # Exploratory dataset analysis
├── fuzzification_polymer_particlesize.ipynb # Two-feature fuzzification example
├── mp_dataset_initial.xlsx                  # Source dataset and metadata
├── mp_dataset_processed.xlsx                # Numeric modeling dataset
├── related_work_table.xlsx                  # Related-literature summary
├── proposal.md                              # Research context and proposed methodology
└── main.pdf                                 # Benchmark research paper
```

## Main libraries

- NumPy
- pandas
- Matplotlib
- openpyxl

No pinned environment or dependency file is provided. The visualization notebook reads the included Excel workbooks directly. The fuzzification notebook additionally imports `fuzzypharma.fuzzy`, which is not included in this repository, so it cannot be executed from a clean checkout without that module.
