import json

import pytest

from src.results import (
    build_provenance,
    load_evaluation_artifacts,
    load_results,
    save_evaluation_artifacts,
    save_results,
    validate_provenance,
    validate_results,
)


@pytest.fixture
def valid_results():
    metadata = build_provenance()
    summary = {
        "confounding_strength": 0.0,
        "estimator": "naive",
        "mean_estimate": 0.2,
        "bias": 0.0,
        "bias_mc_se": 0.01,
        "bias_mc_ci_low": -0.02,
        "bias_mc_ci_high": 0.02,
        "rmse": 0.1,
        "coverage": 0.95,
        "coverage_ci_low": 0.8,
        "coverage_ci_high": 0.99,
        "n_runs": 10,
    }
    return {
        "schema_version": 3,
        "metadata": metadata,
        "headline": {
            "pooled_ate": [{"outcome": "visit", "ate": 0.06, "ci_low": 0.04, "ci_high": 0.08}],
            "per_arm_ate": [
                {
                    "arm": "Mens E-Mail",
                    "outcome": "visit",
                    "ate": 0.06,
                    "ci_low": 0.04,
                    "ci_high": 0.08,
                }
            ],
            "deduplicated_visit_sensitivity": {
                "full_rows": 3,
                "deduplicated_rows": 3,
                "full_visit_difference": 0.06,
                "deduplicated_visit_difference": 0.06,
            },
        },
        "interactions": {
            "analysis": "post-hoc held-out interaction audit",
            "joint_p_value": 0.001,
            "segment_effects": [],
            "contrasts": [],
        },
        "ranking": {
            "cate_summary": {"mean": 0.0, "std": 0.1, "min": -0.2, "max": 0.2},
            "raw_qini": 1.2,
            "normalized_qini": {"value": 0.02, "ci_low": -0.01, "ci_high": 0.05},
            "repeated_summary": {"mean": 0.02, "std": 0.01, "min": -0.01, "max": 0.05},
            "repeated_splits": [],
            "top_k": [],
            "deciles": [],
            "forest_sensitivity": [],
            "gross_spend_top_30": {"value": 0.5, "ci_low": 0.1, "ci_high": 0.9},
        },
        "policy": {
            "values": [{"policy": "learned", "value": 0.18, "ci_low": 0.16, "ci_high": 0.20}],
            "comparisons": [
                {
                    "comparison": "learned - blanket",
                    "difference": 0.01,
                    "ci_low": -0.01,
                    "ci_high": 0.03,
                }
            ],
            "recommendation_counts": {"Mens E-Mail": 1},
            "recommendation_shares": {"Mens E-Mail": 1.0},
            "conclusion": "No win established.",
            "split_sensitivity": [],
        },
        "reported_spend_sensitivity": {
            "values": [],
            "comparisons": [],
            "contact_rate": 0.5,
            "contact_rates": {"learned": 0.5},
            "email_cost_usd": 0.1,
            "break_even_gross_margin": None,
            "margin_sensitivity": [],
        },
        "naive": {
            "estimates": [
                {
                    "outcome": "visit",
                    "treated_mean": 0.2,
                    "control_mean": 0.1,
                    "diff": 0.1,
                    "ci_low": 0.02,
                    "ci_high": 0.18,
                }
            ],
            "balance": [
                {
                    "covariate": "x",
                    "treated_mean": 0.5,
                    "control_mean": 0.5,
                    "standardised_diff": None,
                }
            ],
        },
        "regression": {
            "estimates": [
                {
                    "outcome": "visit",
                    "contrast": "treatment",
                    "kind": "average 0-to-1 change",
                    "effect": 0.1,
                    "ci_low": 0.02,
                    "ci_high": 0.18,
                }
            ],
            "logit_contrasts": {
                "visit": [
                    {
                        "contrast": "treatment",
                        "kind": "average 0-to-1 change",
                        "effect": 0.1,
                        "ci_low": 0.02,
                        "ci_high": 0.18,
                        "se": 0.04,
                    }
                ],
                "conversion": [
                    {
                        "contrast": "treatment",
                        "kind": "average 0-to-1 change",
                        "effect": 0.01,
                        "ci_low": 0.0,
                        "ci_high": 0.02,
                        "se": 0.005,
                    }
                ],
            },
        },
        "identification": {"outcomes": []},
        "confounding": {
            "experimental_reference": {"ate": 0.06, "ci_low": 0.04, "ci_high": 0.08},
            "variants": {
                name: {
                    "n": 100,
                    "naive": 0.1,
                    "ols": 0.06,
                    "dml": {"ate": 0.06, "ci_low": 0.04, "ci_high": 0.08},
                }
                for name in ("observable", "unmeasured")
            },
        },
        "refutations": {
            name: {
                "reference_estimate": 0.06,
                "placebo": {"ate": 0.0, "ci_low": -0.01, "ci_high": 0.01},
                "random_cause": {"ate": 0.06, "ci_low": 0.04, "ci_high": 0.08},
                "subset": [0.05, 0.06],
            }
            for name in ("rct", "selected_sample")
        },
        "figures": {"files": []},
        "simulation": {
            "analysis": "synthetic",
            "rows": [summary],
            "extended_rows": [summary],
            "extended_repetitions": 10,
        },
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


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_results_reject_non_finite_numbers_before_publication(valid_results, tmp_path, bad_value):
    valid_results["ranking"]["normalized_qini"]["value"] = bad_value

    with pytest.raises(ValueError, match="finite"):
        save_results(valid_results, tmp_path / "results.json")


def test_results_allow_null_only_for_an_unavailable_optional_measure(valid_results):
    valid_results["simulation"]["rows"][0]["coverage"] = None
    valid_results["simulation"]["rows"][0]["coverage_ci_low"] = None
    valid_results["simulation"]["rows"][0]["coverage_ci_high"] = None

    validate_results(valid_results)


def test_results_reject_null_for_a_required_numeric_value(valid_results):
    valid_results["simulation"]["rows"][0]["bias"] = None

    with pytest.raises(ValueError, match="simulation.rows.*bias.*cannot be null"):
        validate_results(valid_results)


def test_results_rejects_malformed_nested_policy_and_ranking_fields(valid_results):
    del valid_results["policy"]["comparisons"][0]["ci_high"]

    with pytest.raises(ValueError, match="policy.comparisons.*ci_high"):
        validate_results(valid_results)


def test_results_rejects_non_numeric_nested_ranking_value(valid_results):
    valid_results["ranking"]["normalized_qini"]["value"] = "0.02"

    with pytest.raises(ValueError, match="ranking.normalized_qini.value.*number"):
        validate_results(valid_results)


def test_results_reject_unknown_schema_version(valid_results):
    valid_results["schema_version"] = 99

    with pytest.raises(ValueError, match="schema version"):
        validate_results(valid_results)


def test_results_rejects_schema_one(valid_results):
    valid_results["schema_version"] = 1

    with pytest.raises(ValueError, match="schema version"):
        validate_results(valid_results)


def test_results_rejects_old_interaction_contract(valid_results):
    del valid_results["interactions"]["contrasts"]

    with pytest.raises(ValueError, match="interactions missing"):
        validate_results(valid_results)


def test_results_file_is_plain_json(tmp_path, valid_results):
    path = tmp_path / "results.json"
    save_results(valid_results, path)

    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["random_seed"] == 42


def test_results_loader_rejects_nonstandard_json_nan(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON numeric value NaN"):
        load_results(path)


def test_results_rejects_missing_nested_regression_contrast(valid_results):
    del valid_results["regression"]["logit_contrasts"]["visit"][0]["ci_high"]

    with pytest.raises(ValueError, match="regression.logit_contrasts.visit.*ci_high"):
        validate_results(valid_results)


def test_evaluation_artifact_round_trip(tmp_path):
    path = tmp_path / "evaluation.joblib"
    payload = {"eval_rows": 19200, "normalized_qini": 0.0123}

    save_evaluation_artifacts(payload, path)

    assert load_evaluation_artifacts(path) == payload


def test_provenance_rejects_an_artifact_built_from_different_source():
    metadata = build_provenance()
    metadata["source_sha256"] = "not-the-current-source"

    with pytest.raises(ValueError, match="source fingerprint"):
        validate_provenance(metadata)


def test_load_results_rejects_stale_manifest_provenance(tmp_path, valid_results):
    valid_results["metadata"]["source_sha256"] = "stale"
    path = tmp_path / "results.json"
    save_results(valid_results, path)

    with pytest.raises(ValueError, match="source fingerprint"):
        load_results(path)


def test_provenance_declares_whether_the_worktree_was_dirty():
    metadata = build_provenance()

    assert isinstance(metadata["git_dirty"], bool)
