import json

import pytest

from src.results import (
    load_evaluation_artifacts,
    load_results,
    save_evaluation_artifacts,
    save_results,
    validate_results,
)


@pytest.fixture
def valid_results():
    return {
        "schema_version": 1,
        "metadata": {
            "data_sha256": "abc123",
            "feature_columns": ["recency", "history"],
            "random_seed": 42,
            "git_sha": "deadbeef",
            "packages": {"scikit-learn": "1.5.2"},
        },
        "headline": {"visit_ate": 0.0601},
        "interactions": {"joint_p_value": 0.001, "terms": []},
        "ranking": {"normalized_qini": 0.02, "repeated_splits": []},
        "policy": {"values": [], "comparisons": []},
    }


def test_results_round_trip_is_deterministic(tmp_path, valid_results):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    save_results(valid_results, first)
    save_results(valid_results, second)

    assert first.read_bytes() == second.read_bytes()
    assert load_results(first) == valid_results
    assert first.read_text(encoding="utf-8").endswith("\n")


def test_results_reject_missing_public_section(valid_results):
    del valid_results["policy"]

    with pytest.raises(ValueError, match="missing sections: policy"):
        validate_results(valid_results)


def test_results_reject_unknown_schema_version(valid_results):
    valid_results["schema_version"] = 99

    with pytest.raises(ValueError, match="schema version"):
        validate_results(valid_results)


def test_results_file_is_plain_json(tmp_path, valid_results):
    path = tmp_path / "results.json"
    save_results(valid_results, path)

    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["random_seed"] == 42


def test_evaluation_artifact_round_trip(tmp_path):
    path = tmp_path / "evaluation.joblib"
    payload = {"eval_rows": 19200, "normalized_qini": 0.0123}

    save_evaluation_artifacts(payload, path)

    assert load_evaluation_artifacts(path) == payload
