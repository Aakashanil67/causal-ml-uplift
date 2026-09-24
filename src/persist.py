"""Serialises the final CausalForestDML for the Streamlit simulator, and the load-and-predict
helper it calls.

Refit on the full 64,000 rows, not the 70% train split `src/cate.py` uses for honest evaluation:
heterogeneity has already been examined on a held-out set in `reports/07_uplift_policy.md`,
so the deployed artifact should use every available row rather than holding 30% back for no benefit
at serving time. Artifact size measured directly rather than assumed: 34.47 MB, comfortably under
both GitHub's 100MB hard limit and the 50MB threshold this project set for needing a distilled
surrogate instead — so the real forest ships, not an approximation of it.
"""

import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML

from src.config import (
    BINARY_COVARIATES,
    CATEGORICAL_LEVELS,
    COVARIATE_COLS,
    MODELS_DIR,
    NUMERIC_COVARIATES,
)
from src.data_loader import build_covariate_matrix
from src.results import build_provenance, validate_provenance

MODEL_PATH = MODELS_DIR / "causal_forest.joblib"


def fit_and_save(
    df: pd.DataFrame, out_path=MODEL_PATH, metadata: dict | None = None
) -> CausalForestDML:
    from src.cate import fit_causal_forest

    cf = fit_causal_forest(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": cf, "metadata": metadata or build_provenance()}, out_path)
    return cf


def load_model(path=MODEL_PATH) -> CausalForestDML:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or set(artifact) != {"model", "metadata"}:
        raise ValueError("Legacy model artifact has no provenance metadata; regenerate it.")
    validate_provenance(artifact["metadata"])
    return artifact["model"]


def predict_cate_for_profile(model: CausalForestDML, profile: dict) -> dict:
    """`profile` must have exactly `src/config.COVARIATE_COLS`' keys — one customer's recency,
    history, mens/womens/newbie flags, zip_code and channel. Returns the point estimate and its
    95% confidence interval on `visit`."""
    missing = set(COVARIATE_COLS) - set(profile)
    if missing:
        raise ValueError(f"profile missing required fields: {missing}")
    for column in NUMERIC_COVARIATES:
        value = profile[column]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{column} must be a finite numeric value.")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{column} must be a finite numeric value.")
        low, high = (1, 12) if column == "recency" else (29.99, 3345.93)
        if not low <= number <= high:
            raise ValueError(f"{column} must be between {low} and {high}.")
    for column in BINARY_COVARIATES:
        if profile[column] not in (0, 1, False, True):
            raise ValueError(f"{column} must be a binary flag equal to 0 or 1.")
    for column, levels in CATEGORICAL_LEVELS.items():
        if profile[column] not in levels:
            raise ValueError(f"{column} has unsupported category {profile[column]!r}.")
    row = pd.DataFrame([profile])[COVARIATE_COLS]
    X = build_covariate_matrix(row).to_numpy()
    effect = float(model.effect(X)[0])
    ci_low, ci_high = model.effect_interval(X)
    return {
        "cate": effect,
        "ci_low": float(ci_low[0]),
        "ci_high": float(ci_high[0]),
        "extrapolative": not (bool(profile["mens"]) or bool(profile["womens"])),
    }


def main() -> None:
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    print("fitting final model on the full dataset...")
    fit_and_save(df)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"saved to {MODEL_PATH}, {size_mb:.2f} MB")

    model = load_model()
    example_profile = {
        "recency": 3,
        "history": 250.0,
        "mens": 1,
        "womens": 0,
        "newbie": 0,
        "zip_code": "Urban",
        "channel": "Web",
    }
    result = predict_cate_for_profile(model, example_profile)
    print(f"example prediction: {result}")


if __name__ == "__main__":
    main()
