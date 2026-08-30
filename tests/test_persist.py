import joblib
import numpy as np
import pytest

from src.config import RANDOM_SEED
from src.persist import fit_and_save, load_model, predict_cate_for_profile
from src.results import build_provenance

VALID_PROFILE = {
    "recency": 3,
    "history": 250.0,
    "mens": 1,
    "womens": 0,
    "newbie": 0,
    "zip_code": "Urban",
    "channel": "Web",
}


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    # a full 64k-row refit takes ~23s (see src/persist.py docstring); a 4,000-row subsample is
    # plenty to test the save/load/predict round-trip without paying that cost in every test run.
    from src.data_loader import load_hillstrom

    df = load_hillstrom().sample(n=4000, random_state=RANDOM_SEED).reset_index(drop=True)
    out_path = tmp_path_factory.mktemp("models") / "test_causal_forest.joblib"
    fit_and_save(df, out_path=out_path)
    return load_model(out_path)


def test_round_trip_produces_a_working_model(small_model):
    result = predict_cate_for_profile(small_model, VALID_PROFILE)
    assert np.isfinite(result["cate"])
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["cate"] <= result["ci_high"]


def test_prediction_is_a_plausible_visit_probability_shift(small_model):
    # a CATE on a probability outcome should be a small, bounded shift, not an arbitrary number —
    # catches a real class of bug (wrong outcome scale, wrong effect sign convention).
    result = predict_cate_for_profile(small_model, VALID_PROFILE)
    assert -1.0 < result["cate"] < 1.0


def test_missing_profile_field_raises():
    incomplete = {k: v for k, v in VALID_PROFILE.items() if k != "zip_code"}
    with pytest.raises(ValueError, match="missing required fields"):
        predict_cate_for_profile(object(), incomplete)


def test_predict_ignores_extra_profile_keys(small_model):
    # a profile dict from a UI form might carry extra fields (e.g. a customer id); prediction
    # should use only the known covariates, not choke on the rest.
    extra = {**VALID_PROFILE, "customer_id": 12345}
    result = predict_cate_for_profile(small_model, extra)
    assert np.isfinite(result["cate"])


def test_committed_model_artifact_is_under_the_surrogate_threshold():
    # this project's own decision: ship the real CausalForestDML if it stays under 50MB,
    # distill a LightGBM surrogate instead if it doesn't. If a future change
    # (more estimators, more covariates) pushes the committed artifact over that line, this should
    # fail loudly rather than someone noticing a slow git push or a Streamlit Cloud memory error.
    from src.persist import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("models/causal_forest.joblib not present in this checkout")
    size_mb = MODEL_PATH.stat().st_size / 1e6
    assert size_mb < 50, f"{MODEL_PATH} is {size_mb:.1f}MB, over the 50MB surrogate threshold"


def test_model_loader_rejects_a_tampered_feature_contract(tmp_path):
    metadata = build_provenance()
    metadata["feature_columns"] = ["wrong_column"]
    path = tmp_path / "tampered.joblib"
    joblib.dump({"model": object(), "metadata": metadata}, path)

    with pytest.raises(ValueError, match="feature schema"):
        load_model(path)


def test_provenance_records_reproducibility_inputs():
    metadata = build_provenance()

    assert metadata["random_seed"] == RANDOM_SEED
    assert metadata["data_sha256"]
    assert metadata["feature_columns"] == list(VALID_PROFILE)
    assert set(metadata["packages"]) == {
        "econml",
        "lightgbm",
        "numpy",
        "pandas",
        "scikit-learn",
    }
