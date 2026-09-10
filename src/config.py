"""Single source of truth for paths, seeds and modelling constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
RESULTS_PATH = REPORTS_DIR / "results.json"
EVALUATION_ARTIFACT_PATH = MODELS_DIR / "evaluation_artifacts.joblib"

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW_CSV_PATH = DATA_DIR / "hillstrom.csv"
# SHA-256 of the published CSV used to produce the tracked reports and model artifact. The source
# is HTTP-only; integrity is therefore verified after download rather than assumed from transport.
HILLSTROM_SHA256 = "0e5893329d8b93cef ecc571777672028290ab69865718020c78c7284f291aece".replace(
    " ", ""
)

RANDOM_SEED = 42

ARM_COL = "segment"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
CONTROL_ARM = "No E-Mail"
NOMINAL_PROPENSITIES = {arm: 1 / len(ARMS) for arm in ARMS}
TREATMENT_COL = "treatment"  # 1 if segment != CONTROL_ARM, 0 otherwise — built by data_loader

OUTCOME_COLS = ["visit", "conversion", "spend"]

NUMERIC_COVARIATES = ["recency", "history"]
BINARY_COVARIATES = ["mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel"]
COVARIATE_COLS = NUMERIC_COVARIATES + BINARY_COVARIATES + CATEGORICAL_COVARIATES

# fixed, exhaustive category lists — build_covariate_matrix() uses these (not whatever categories
# happen to appear in a given input) so a single-row inference request produces the same columns
# as a full-dataset fit; without this, get_dummies() on a subset missing a category silently drops
# that dummy column instead of erroring, which only surfaces at serving time (see src/persist.py).
CATEGORICAL_LEVELS = {
    "zip_code": ["Rural", "Surburban", "Urban"],
    "channel": ["Multichannel", "Phone", "Web"],
}

# heterogeneity is only well-powered on visit (~9,000 events across 64,000 rows) — conversion and
# spend have 578 non-zero outcomes total, so CATE work targets visit specifically; see
# reports/data_dictionary.md for the power check this constant is based on.
CATE_OUTCOME = "visit"

# 70/30, stratified on arm and CATE_OUTCOME — every CATE/Qini number reported comes from the 30%
# the forest never saw during fitting.
TRAIN_FRACTION = 0.7
EVAL_FRACTION = 0.3

# per-contact email cost: a stated assumption, not a fitted value, because Hillstrom ships no cost
# data. Configurable per call. The generated report includes reported-spend sensitivity at explicit
# gross-margin scenarios; it does not present spend as profit.
DEFAULT_EMAIL_COST_USD = 0.10
