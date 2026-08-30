"""Serialises the final CausalForestDML for the Streamlit simulator, and the load-and-predict
helper it calls.

Refit on the full 64,000 rows, not the 70% train split `src/cate.py` uses for honest evaluation:
heterogeneity has already been validated against a held-out set in `reports/07_uplift_policy.md`,
so the deployed artifact should use every available row rather than holding 30% back for no benefit
at serving time. Artifact size measured directly rather than assumed: 34.47 MB, comfortably under
both GitHub's 100MB hard limit and the 50MB threshold this project set for needing a distilled
surrogate instead — so the real forest ships, not an approximation of it.
"""

import joblib
import pandas as pd
from econml.dml import CausalForestDML

from src.config import COVARIATE_COLS, MODELS_DIR
from src.data_loader import build_covariate_matrix
from src.results import build_provenance, validate_provenance

MODEL_PATH = MODELS_DIR / "causal_forest.joblib"


def fit_and_save(df: pd.DataFrame, out_path=MODEL_PATH) -> CausalForestDML:
    from src.cate import fit_causal_forest

    cf = fit_causal_forest(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": cf, "metadata": build_provenance()}, out_path)
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
    row = pd.DataFrame([profile])[COVARIATE_COLS]
    X = build_covariate_matrix(row).to_numpy()
    effect = float(model.effect(X)[0])
    ci_low, ci_high = model.effect_interval(X)
    return {"cate": effect, "ci_low": float(ci_low[0]), "ci_high": float(ci_high[0])}


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
