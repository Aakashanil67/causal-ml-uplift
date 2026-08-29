"""Downloads and validates the Hillstrom MineThatData e-mail experiment CSV.

64,000 customers randomised into three arms (`segment`): no email, a mens-product email, or a
womens-product email. `build_covariate_matrix()` is the single place covariates get encoded into a
numeric matrix, reused by every downstream estimator so the encoding can't drift between modules.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    ARM_COL,
    ARMS,
    BINARY_COVARIATES,
    CATEGORICAL_COVARIATES,
    CATEGORICAL_LEVELS,
    CONTROL_ARM,
    HILLSTROM_SHA256,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_hillstrom_csv(dest: Path = RAW_CSV_PATH, timeout: int = 30) -> None:
    """Fetch the raw CSV atomically and verify it against the pinned project checksum."""
    if dest.exists() and _sha256(dest) == HILLSTROM_SHA256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(HILLSTROM_URL, timeout=timeout) as response,
            temporary.open("wb") as file,
        ):
            while block := response.read(1024 * 1024):
                file.write(block)
        actual_hash = _sha256(temporary)
        if actual_hash != HILLSTROM_SHA256:
            raise ValueError(
                f"Hillstrom CSV checksum mismatch: expected {HILLSTROM_SHA256}, got {actual_hash}."
            )
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    need the drop but tolerate it fine).

    Categoricals are cast to `CATEGORICAL_LEVELS`' fixed category lists before dummying, not left
    to whatever values happen to appear in `df`: a single-row prediction request only ever has one
    `zip_code` value, and without a fixed category set `get_dummies` would silently produce a
    different (too-narrow) column set than the one the model was trained on."""
    numeric = df[NUMERIC_COVARIATES + BINARY_COVARIATES].astype(float)
    categorical = df[CATEGORICAL_COVARIATES].copy()
    for col, levels in CATEGORICAL_LEVELS.items():
        categorical[col] = pd.Categorical(categorical[col], categories=levels)
    dummies = pd.get_dummies(categorical, drop_first=True, dtype=float)
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
