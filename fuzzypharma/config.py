"""Shared configuration: paths, column names, and preprocessing constants.

Every other module imports its names from here so that a column rename or a
policy change happens in exactly one place.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "mp_dataset_processed.xlsx"
SHEET_NAME = "Sheet1"
PREPARED_DIR = PROJECT_DIR / "prepared_data"

#: The source dataset from Bao et al. (2025a), Sci. Data 12(1), 364,
#: https://doi.org/10.1038/s41597-025-04621-9, mirrored at
#: https://data.mendeley.com/datasets/zzvtdrcy76/1
#:
#: The processed workbook is a numeric projection of this file that drops drug
#: identity and formulation method. Downloading the original recovers both. Its
#: 12 shared numeric columns are bit-identical to the processed workbook and its
#: ``Formulation Index`` matches row for row, so the join is exact rather than
#: approximate.
METADATA_PATH = PROJECT_DIR / "mp_dataset_initial.xlsx"
METADATA_SHEET = "PLGA_MPs"

# --- Reproducibility -----------------------------------------------------

SEED = 42
N_SPLITS = 5

#: Fold assignment is deterministic (see :func:`fuzzypharma.folds.assign_folds`);
#: ``SEED`` only breaks ties between equal-sized drug groups.
#:
#: It is not delegated to ``StratifiedGroupKFold`` because that produced
#: different folds on different scikit-learn versions for the same seed --
#: 38-101 formulations per fold on 1.2.2 versus 63-65 on 1.8.0.

# --- Columns -------------------------------------------------------------

ID_COL = "Formulation Index"
TIME_COL = "Time"
RELEASE_COL = "Release"
GROUP_COL = "Drug Group"

#: Descriptors that identify the drug molecule. Formulations sharing all three
#: are treated as the same drug (see :func:`fuzzypharma.data.assign_drug_groups`).
DRUG_DESCRIPTOR_COLS = ["Drug MW", "Drug TPSA", "Drug LogP"]

# --- Recovered identity and method columns -------------------------------

DRUG_COL = "Drug"
SMILES_COL = "Drug SMILES"
METHOD_COL = "Formulation Method"
DOI_COL = "DOI"
METADATA_COLS = [DRUG_COL, SMILES_COL, METHOD_COL, DOI_COL]

#: Emulsion methods, declared as a fixed schema rather than read off the data.
#:
#: One-hot encoding against a declared vocabulary keeps the feature width equal
#: in every fold. Fitting the encoder per fold would instead drop a column
#: whenever a method is absent from a training fold, which happens here:
#: ``S/W/O/W`` has exactly one formulation in the whole dataset. No target
#: information enters this list, so declaring it is a schema choice, not a fit.
FORMULATION_METHODS = ["O/W", "S/O/W", "W/O/W", "S/W/O/W"]

#: The ten static inputs present in this workbook, in a fixed order.
#:
#: The benchmark (Robles & Samad, 2026) additionally one-hot encodes a
#: ``Formulation Method`` column that this processed workbook does not contain.
#: Results are therefore not a like-for-like reproduction of the published
#: numbers; see ``PREPROCESSING.md``.
STATIC_FEATURES = [
    "Drug MW",
    "Drug TPSA",
    "Drug LogP",
    "Polymer MW",
    "LA/GA",
    "Initial Drug-to-Polymer Ratio",
    "Particle Size",
    "Drug Loading Capacity",
    "Drug Encapsulation Efficiency",
    "Solubility Enhancer Concentration",
]

EXPECTED_COLUMNS = [ID_COL, *STATIC_FEATURES, TIME_COL, RELEASE_COL]

# --- Time unit -----------------------------------------------------------

#: ``Time`` is recorded in days. Confirmed against Robles & Samad (2026), which
#: reports profile durations of "minimum: 72 h, maximum: 238 days, mean:
#: 30 +/- 25 days"; this workbook reproduces those exactly (3.003 / 237.73 /
#: 30.02 +/- 25.47).
TIME_UNIT = "days"

#: Early-release regression targets, mapped from hours to workbook days.
EARLY_TARGETS = {"Release_24h": 1.0, "Release_48h": 2.0, "Release_72h": 3.0}

# --- Curve / AUC targets -------------------------------------------------

#: Number of points on the common normalized-time grid used for Task 3
#: (complete release-curve prediction).
CURVE_GRID_POINTS = 50

AUC_COL = "AUC"
AUC_CLASS_COL = "AUC Class"

#: Threshold splitting the two mathematical release-profile classes. This is a
#: summary of a normalized curve, NOT a clinical burst-release diagnosis.
AUC_THRESHOLD = 0.5
AUC_CLASS_LOW = "AUC <= 0.5"
AUC_CLASS_HIGH = "AUC > 0.5"

# --- Release-value policy ------------------------------------------------

#: Primary analysis preserves measured release values as recorded, including
#: the 108 rows slightly above 1.0 and the non-monotonic segments. Clipping is
#: available as a documented sensitivity analysis only.
CLIP_RELEASE = False
RELEASE_BOUNDS = (0.0, 1.0)
