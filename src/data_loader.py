"""Downloads and validates the Hillstrom MineThatData e-mail experiment CSV.

64,000 customers randomised into three arms (`segment`): no email, a mens-product email, or a
womens-product email. `build_covariate_matrix()` is the single place covariates get encoded into a
numeric matrix, reused by every downstream estimator so the encoding can't drift between modules.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    ARM_COL,
    ARMS,
    BINARY_COVARIATES,
    CATEGORICAL_COVARIATES,
    CONTROL_ARM,
    DATA_DIR,
    HILLSTROM_URL,
    NUMERIC_COVARIATES,
    OUTCOME_COLS,
    RAW_CSV_PATH,
    TREATMENT_COL,
)

EXPECTED_COLUMNS = {
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
}


def download_hillstrom_csv(dest: Path = RAW_CSV_PATH) -> None:
    if dest.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HILLSTROM_URL, dest)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom CSV missing expected columns: {missing}")
    if df.isnull().any().any():
        raise ValueError("Hillstrom CSV has nulls — the published data doesn't; something changed.")
    unknown_arms = set(df[ARM_COL].unique()) - set(ARMS)
    if unknown_arms:
        raise ValueError(f"Unexpected arm labels: {unknown_arms}")
    # conversion is a strict subset of visit in the published data (every buyer visited first);
    # a downstream module (naive.py) relies on this to justify treating visit as the well-powered
    # outcome and conversion/spend as the sparse ones.
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        raise ValueError("Found conversion=1 with visit=0 — conversion no longer implies visit.")


def load_hillstrom() -> pd.DataFrame:
    """Loads, validates and adds a binary `treatment` column (1 if any email, 0 if control)."""
    download_hillstrom_csv()
    df = pd.read_csv(RAW_CSV_PATH)
    _validate_schema(df)
    df[TREATMENT_COL] = (df[ARM_COL] != CONTROL_ARM).astype(int)
    return df


def build_covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric + binary covariates pass through; categoricals become one-hot dummies (first level
    dropped to avoid collinearity in the linear/logit stages — the tree-based estimators don't
    need the drop but tolerate it fine)."""
    numeric = df[NUMERIC_COVARIATES + BINARY_COVARIATES].astype(float)
    dummies = pd.get_dummies(df[CATEGORICAL_COVARIATES], drop_first=True, dtype=float)
    return pd.concat([numeric, dummies], axis=1)


def arm_sizes(df: pd.DataFrame) -> pd.Series:
    return df[ARM_COL].value_counts().reindex(ARMS)


def outcome_rates(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(ARM_COL)[OUTCOME_COLS].mean().reindex(ARMS)


def main() -> None:
    df = load_hillstrom()
    print(f"rows: {len(df)}, columns: {len(df.columns)}")
    print("\narm sizes:")
    print(arm_sizes(df))
    print("\noutcome rates by arm:")
    print(outcome_rates(df))
    print(f"\nnon-zero spend: {(df['spend'] > 0).sum()} rows")
    print(f"spend censored at 499.00: {(df['spend'] == 499.0).sum()} rows")


if __name__ == "__main__":
    main()
