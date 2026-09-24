"""Versioned contract for numerical results shared by reports and the app."""

import hashlib
import json
import math
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib

from src.config import (
    COVARIATE_COLS,
    EVALUATION_ARTIFACT_PATH,
    HILLSTROM_SHA256,
    RANDOM_SEED,
    RESULTS_PATH,
    ROOT,
)

RESULTS_SCHEMA_VERSION = 3
REQUIRED_SECTIONS = {
    "metadata",
    "headline",
    "interactions",
    "ranking",
    "policy",
    "reported_spend_sensitivity",
    "simulation",
    "naive",
    "regression",
    "identification",
    "confounding",
    "refutations",
    "figures",
}
PROVENANCE_PACKAGES = ("econml", "lightgbm", "numpy", "pandas", "scikit-learn")


def source_sha256() -> str:
    """Fingerprint the code and dependency declarations that define an artifact."""
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        ROOT / "app" / "simulator.py",
        ROOT / "requirements.txt",
        ROOT / "app" / "requirements.txt",
        ROOT / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        # Git checkout line endings vary between Windows and Streamlit's Linux workers.
        # Fingerprint source content, not that platform-specific representation.
        digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_provenance() -> dict:
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_sha = "unknown"
        git_dirty = None
    return {
        "data_sha256": HILLSTROM_SHA256,
        "feature_columns": COVARIATE_COLS,
        "random_seed": RANDOM_SEED,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "source_sha256": source_sha256(),
        "packages": {package: version(package) for package in PROVENANCE_PACKAGES},
    }


def validate_provenance(metadata: dict) -> None:
    if metadata.get("data_sha256") != HILLSTROM_SHA256:
        raise ValueError("Artifact data checksum does not match this project.")
    if metadata.get("feature_columns") != COVARIATE_COLS:
        raise ValueError("Artifact feature schema does not match this project.")
    if metadata.get("source_sha256") != source_sha256():
        raise ValueError("Artifact source fingerprint does not match the current project.")
    runtime = {package: version(package) for package in PROVENANCE_PACKAGES}
    if metadata.get("packages") != runtime:
        raise ValueError("Artifact dependency versions do not match the current runtime.")


def _require_mapping(parent: dict, key: str, path: str) -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Results field {path} must be an object.")
    return value


def _require_string_fields(records, section: str, fields: tuple[str, ...]) -> None:
    if not isinstance(records, list):
        raise ValueError(f"Results field {section} must be an array.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Results field {section}[{index}] must be an object.")
        for field in fields:
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(
                    f"Results field {section}[{index}].{field} must be a non-empty string."
                )


def _require_numeric_mapping(mapping: dict, path: str, fields: tuple[str, ...]) -> None:
    for field in fields:
        if field not in mapping:
            raise ValueError(f"Results field {path}.{field} is required.")
        _require_number(mapping[field], f"{path}.{field}")


def _require_number(value, path: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Results field {path} must be a number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"Results field {path} must be finite.")


def _require_record_fields(records, section: str, fields: dict[str, bool]) -> None:
    if not isinstance(records, list):
        raise ValueError(f"Results field {section} must be an array.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Results field {section}[{index}] must be an object.")
        for field, allow_none in fields.items():
            if field not in record:
                raise ValueError(f"Results field {section}[{index}].{field} is required.")
            _require_number(record[field], f"{section}[{index}].{field}", allow_none)


def _validate_finite_tree(value, path: str = "results") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Results field {path} must be finite.")


def _validate_nulls(value, path: str = "results") -> None:
    allowed = {
        "metadata.git_dirty",
        "naive.balance[].standardised_diff",
        "ranking.forest_sensitivity[].max_depth",
        "reported_spend_sensitivity.break_even_gross_margin",
        "simulation.rows[].coverage",
        "simulation.rows[].coverage_ci_low",
        "simulation.rows[].coverage_ci_high",
        "simulation.extended_rows[].coverage",
        "simulation.extended_rows[].coverage_ci_low",
        "simulation.extended_rows[].coverage_ci_high",
    }
    if value is None:
        normalized = re.sub(r"\[\d+\]", "[]", path)
        normalized = normalized.removeprefix("results.")
        if normalized not in allowed:
            raise ValueError(f"Results field {path} cannot be null.")
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_nulls(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nulls(child, f"{path}[{index}]")


def validate_results(results: dict) -> None:
    if not isinstance(results, dict):
        raise ValueError("Results root must be an object.")
    if results.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported results schema version: {results.get('schema_version')}; "
            f"expected {RESULTS_SCHEMA_VERSION}."
        )
    missing = REQUIRED_SECTIONS - set(results)
    if missing:
        raise ValueError(f"Results missing sections: {', '.join(sorted(missing))}")
    interactions = _require_mapping(results, "interactions", "interactions")
    required_interactions = {"analysis", "joint_p_value", "segment_effects", "contrasts"}
    missing_interactions = required_interactions - set(interactions)
    if missing_interactions:
        raise ValueError("Results interactions missing: " + ", ".join(sorted(missing_interactions)))

    metadata = _require_mapping(results, "metadata", "metadata")
    required_metadata = {
        "data_sha256",
        "feature_columns",
        "random_seed",
        "git_sha",
        "git_dirty",
        "source_sha256",
        "packages",
    }
    missing_metadata = required_metadata - set(metadata)
    if missing_metadata:
        raise ValueError(f"Results metadata missing: {', '.join(sorted(missing_metadata))}")
    if not isinstance(metadata["packages"], dict) or not metadata["packages"]:
        raise ValueError("Results metadata.packages must be a non-empty object.")
    if not all(
        isinstance(name, str) and isinstance(value, str)
        for name, value in metadata["packages"].items()
    ):
        raise ValueError("Results metadata.packages values must be version strings.")
    if not isinstance(metadata["feature_columns"], list) or not metadata["feature_columns"]:
        raise ValueError("Results metadata.feature_columns must be a non-empty array.")
    if isinstance(metadata["random_seed"], bool) or not isinstance(metadata["random_seed"], int):
        raise ValueError("Results metadata.random_seed must be an integer.")
    for field in ("data_sha256", "git_sha", "source_sha256"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"Results metadata.{field} must be a non-empty string.")
    if metadata["git_dirty"] is not None and not isinstance(metadata["git_dirty"], bool):
        raise ValueError("Results metadata.git_dirty must be a boolean or null.")

    _validate_finite_tree(results)
    _validate_nulls(results)

    if not isinstance(interactions["analysis"], str) or not interactions["analysis"]:
        raise ValueError("Results interactions.analysis must be a non-empty string.")
    _require_numeric_mapping(interactions, "interactions", ("joint_p_value",))
    _require_record_fields(
        interactions["segment_effects"],
        "interactions.segment_effects",
        {"estimate": False, "ci_low": False, "ci_high": False},
    )
    _require_string_fields(
        interactions["segment_effects"], "interactions.segment_effects", ("segment",)
    )
    _require_record_fields(
        interactions["contrasts"],
        "interactions.contrasts",
        {"estimate": False, "ci_low": False, "ci_high": False, "p_holm": False},
    )
    _require_string_fields(interactions["contrasts"], "interactions.contrasts", ("contrast",))

    headline = _require_mapping(results, "headline", "headline")
    for key, fields in {
        "pooled_ate": {"ate": False, "ci_low": False, "ci_high": False},
        "per_arm_ate": {"ate": False, "ci_low": False, "ci_high": False},
    }.items():
        records = headline.get(key)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Results field headline.{key} must be a non-empty array.")
        _require_record_fields(records, f"headline.{key}", fields)
    _require_string_fields(headline["pooled_ate"], "headline.pooled_ate", ("outcome",))
    _require_string_fields(headline["per_arm_ate"], "headline.per_arm_ate", ("arm", "outcome"))
    deduplicated = _require_mapping(
        headline, "deduplicated_visit_sensitivity", "headline.deduplicated_visit_sensitivity"
    )
    _require_numeric_mapping(
        deduplicated,
        "headline.deduplicated_visit_sensitivity",
        (
            "full_rows",
            "deduplicated_rows",
            "full_visit_difference",
            "deduplicated_visit_difference",
        ),
    )

    ranking = _require_mapping(results, "ranking", "ranking")
    qini = _require_mapping(ranking, "normalized_qini", "ranking.normalized_qini")
    for field in ("value", "ci_low", "ci_high"):
        if field not in qini:
            raise ValueError(f"Results field ranking.normalized_qini.{field} is required.")
        _require_number(qini[field], f"ranking.normalized_qini.{field}")
    summary = _require_mapping(ranking, "repeated_summary", "ranking.repeated_summary")
    for field in ("mean", "std", "min", "max"):
        if field not in summary:
            raise ValueError(f"Results field ranking.repeated_summary.{field} is required.")
        _require_number(summary[field], f"ranking.repeated_summary.{field}")
    for key, fields in {
        "repeated_splits": {"seed": False, "normalized_qini": False},
        "top_k": {"k": False, "value": False, "ci_low": False, "ci_high": False},
        "deciles": {
            "decile": False,
            "n": False,
            "pred_cate_mean": False,
            "observed_uplift": False,
        },
        "forest_sensitivity": {"normalized_qini": False},
    }.items():
        if key not in ranking:
            raise ValueError(f"Results field ranking.{key} is required.")
        _require_record_fields(ranking[key], f"ranking.{key}", fields)
    _require_record_fields(
        ranking["repeated_splits"],
        "ranking.repeated_splits",
        {"qini_ci_low": False, "qini_ci_high": False, "n_eval": False},
    )
    _require_record_fields(ranking["deciles"], "ranking.deciles", {"n": False})
    _require_record_fields(
        ranking["forest_sensitivity"], "ranking.forest_sensitivity", {"min_samples_leaf": False}
    )
    _require_string_fields(ranking["forest_sensitivity"], "ranking.forest_sensitivity", ("label",))
    _require_numeric_mapping(ranking, "ranking", ("raw_qini",))
    cate_summary = _require_mapping(ranking, "cate_summary", "ranking.cate_summary")
    _require_numeric_mapping(cate_summary, "ranking.cate_summary", ("mean", "std", "min", "max"))

    policy = _require_mapping(results, "policy", "policy")
    for key, fields in {
        "values": {"value": False, "ci_low": False, "ci_high": False},
        "comparisons": {"difference": False, "ci_low": False, "ci_high": False},
    }.items():
        records = policy.get(key)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Results field policy.{key} must be a non-empty array.")
        _require_record_fields(records, f"policy.{key}", fields)
    _require_string_fields(policy["values"], "policy.values", ("policy",))
    _require_string_fields(policy["comparisons"], "policy.comparisons", ("comparison",))
    for key in ("recommendation_counts", "recommendation_shares"):
        mapping = _require_mapping(policy, key, f"policy.{key}")
        if not mapping:
            raise ValueError(f"Results field policy.{key} must not be empty.")
        for arm, value in mapping.items():
            _require_number(value, f"policy.{key}.{arm}")
    if not isinstance(policy.get("conclusion"), str) or not policy["conclusion"]:
        raise ValueError("Results field policy.conclusion must be a non-empty string.")
    split_rows = policy.get("split_sensitivity")
    if not isinstance(split_rows, list):
        raise ValueError("Results field policy.split_sensitivity must be an array.")
    _require_record_fields(
        split_rows,
        "policy.split_sensitivity",
        {
            "seed": False,
            "learned_value": False,
            "blanket_mens_value": False,
            "learned_minus_blanket_mens": False,
        },
    )
    for index, row in enumerate(split_rows):
        shares = _require_mapping(
            row, "recommendation_shares", f"policy.split_sensitivity[{index}].recommendation_shares"
        )
        if not shares:
            raise ValueError(
                f"Results field policy.split_sensitivity[{index}].recommendation_shares must not be empty."
            )
        for arm, value in shares.items():
            _require_number(value, f"policy.split_sensitivity[{index}].recommendation_shares.{arm}")

    for key, fields in {
        "naive.estimates": {"diff": False, "ci_low": False, "ci_high": False},
        "naive.balance": {"standardised_diff": True},
        "regression.estimates": {"effect": False, "ci_low": False, "ci_high": False},
    }.items():
        parent_name, field_name = key.split(".")
        parent = _require_mapping(results, parent_name, parent_name)
        if field_name not in parent:
            raise ValueError(f"Results field {key} is required.")
        _require_record_fields(parent[field_name], key, fields)
    _require_string_fields(results["naive"]["estimates"], "naive.estimates", ("outcome",))
    _require_record_fields(
        results["naive"]["estimates"],
        "naive.estimates",
        {"treated_mean": False, "control_mean": False},
    )
    _require_string_fields(results["naive"]["balance"], "naive.balance", ("covariate",))
    _require_record_fields(
        results["naive"]["balance"], "naive.balance", {"treated_mean": False, "control_mean": False}
    )
    _require_string_fields(
        results["regression"]["estimates"], "regression.estimates", ("outcome", "contrast", "kind")
    )
    regression = _require_mapping(results, "regression", "regression")
    logit_contrasts = _require_mapping(regression, "logit_contrasts", "regression.logit_contrasts")
    for outcome in ("visit", "conversion"):
        if outcome not in logit_contrasts:
            raise ValueError(f"Results field regression.logit_contrasts.{outcome} is required.")
        _require_record_fields(
            logit_contrasts[outcome],
            f"regression.logit_contrasts.{outcome}",
            {"effect": False, "ci_low": False, "ci_high": False, "se": False},
        )
        _require_string_fields(
            logit_contrasts[outcome], f"regression.logit_contrasts.{outcome}", ("contrast", "kind")
        )

    for section in ("identification", "confounding", "refutations", "reported_spend_sensitivity"):
        _require_mapping(results, section, section)

    identification = _require_mapping(results, "identification", "identification")
    if not isinstance(identification.get("outcomes"), list):
        raise ValueError("Results field identification.outcomes must be an array.")
    for index, row in enumerate(identification["outcomes"]):
        if not isinstance(row, dict) or not {"outcome", "estimand_type", "estimand"} <= set(row):
            raise ValueError(f"Results field identification.outcomes[{index}] is malformed.")
        if (
            not isinstance(row["outcome"], str)
            or not isinstance(row["estimand_type"], str)
            or not isinstance(row["estimand"], str)
        ):
            raise ValueError(
                f"Results field identification.outcomes[{index}] has invalid text fields."
            )
        if not isinstance(row.get("backdoor_variables"), list) or not all(
            isinstance(v, str) for v in row["backdoor_variables"]
        ):
            raise ValueError(
                f"Results field identification.outcomes[{index}].backdoor_variables must be a string array."
            )

    confounding = _require_mapping(results, "confounding", "confounding")
    if not {"experimental_reference", "variants"} <= set(confounding):
        raise ValueError(
            "Results confounding section requires experimental_reference and variants."
        )
    experimental_reference = _require_mapping(
        confounding, "experimental_reference", "confounding.experimental_reference"
    )
    _require_numeric_mapping(
        experimental_reference, "confounding.experimental_reference", ("ate", "ci_low", "ci_high")
    )
    variants = _require_mapping(confounding, "variants", "confounding.variants")
    for key in ("observable", "unmeasured"):
        variant = _require_mapping(variants, key, f"confounding.variants.{key}")
        _require_numeric_mapping(variant, f"confounding.variants.{key}", ("n", "naive", "ols"))
        dml = _require_mapping(variant, "dml", f"confounding.variants.{key}.dml")
        _require_numeric_mapping(
            dml, f"confounding.variants.{key}.dml", ("ate", "ci_low", "ci_high")
        )
    refutations = _require_mapping(results, "refutations", "refutations")
    for key in ("rct", "selected_sample"):
        if key not in refutations or not isinstance(refutations[key], dict):
            raise ValueError(f"Results field refutations.{key} must be an object.")
        refutation = refutations[key]
        _require_numeric_mapping(refutation, f"refutations.{key}", ("reference_estimate",))
        for check in ("placebo", "random_cause"):
            item = _require_mapping(refutation, check, f"refutations.{key}.{check}")
            _require_numeric_mapping(
                item, f"refutations.{key}.{check}", ("ate", "ci_low", "ci_high")
            )
        subset = refutation.get("subset")
        if not isinstance(subset, list) or not subset:
            raise ValueError(f"Results field refutations.{key}.subset must be a non-empty array.")
        for i, value in enumerate(subset):
            _require_number(value, f"refutations.{key}.subset[{i}]")

    figures = _require_mapping(results, "figures", "figures")
    _require_record_fields(figures.get("files"), "figures.files", {})
    for index, figure in enumerate(figures["files"]):
        if not isinstance(figure.get("path"), str) or not isinstance(figure.get("sha256"), str):
            raise ValueError(
                f"Results field figures.files[{index}] requires path and sha256 strings."
            )

    spend = _require_mapping(results, "reported_spend_sensitivity", "reported_spend_sensitivity")
    for key in ("values", "comparisons"):
        records = spend.get(key)
        if not isinstance(records, list):
            raise ValueError(f"Results field reported_spend_sensitivity.{key} must be an array.")
        fields = (
            {"value": False, "ci_low": False, "ci_high": False}
            if key == "values"
            else {"difference": False, "ci_low": False, "ci_high": False}
        )
        _require_record_fields(records, f"reported_spend_sensitivity.{key}", fields)
    _require_string_fields(spend["values"], "reported_spend_sensitivity.values", ("policy",))
    _require_string_fields(
        spend["comparisons"], "reported_spend_sensitivity.comparisons", ("comparison",)
    )
    for field in ("contact_rate", "email_cost_usd", "break_even_gross_margin"):
        if field not in spend:
            raise ValueError(f"Results field reported_spend_sensitivity.{field} is required.")
        _require_number(
            spend[field],
            f"reported_spend_sensitivity.{field}",
            allow_none=(field == "break_even_gross_margin"),
        )
    rates = _require_mapping(spend, "contact_rates", "reported_spend_sensitivity.contact_rates")
    if not rates:
        raise ValueError(
            "Results field reported_spend_sensitivity.contact_rates must not be empty."
        )
    for name, value in rates.items():
        _require_number(value, f"reported_spend_sensitivity.contact_rates.{name}")
    _require_record_fields(
        spend.get("margin_sensitivity"),
        "reported_spend_sensitivity.margin_sensitivity",
        {"gross_margin": False, "incremental_net_value": False},
    )

    simulation = _require_mapping(results, "simulation", "simulation")
    for key in ("rows", "extended_rows"):
        if key not in simulation or not isinstance(simulation[key], list):
            raise ValueError(f"Results field simulation.{key} must be an array.")
        _require_record_fields(
            simulation[key],
            f"simulation.{key}",
            {
                "confounding_strength": False,
                "mean_estimate": False,
                "bias": False,
                "bias_mc_se": False,
                "bias_mc_ci_low": False,
                "bias_mc_ci_high": False,
                "rmse": False,
                "coverage": True,
                "coverage_ci_low": True,
                "coverage_ci_high": True,
                "n_runs": False,
            },
        )
        _require_string_fields(simulation[key], f"simulation.{key}", ("estimator",))
    if not isinstance(simulation.get("analysis"), str) or not isinstance(
        simulation.get("extended_repetitions"), int
    ):
        raise ValueError("Results simulation analysis and extended_repetitions are required.")


def save_results(results: dict, path: Path = RESULTS_PATH) -> None:
    validate_results(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    path.write_text(rendered, encoding="utf-8")


def load_results(path: Path = RESULTS_PATH) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON numeric value {value}; use null for unavailable values.")

    results = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    validate_results(results)
    validate_provenance(results["metadata"])
    return results


def save_evaluation_artifacts(
    payload: dict, path: Path = EVALUATION_ARTIFACT_PATH, metadata: dict | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": metadata or build_provenance(), "payload": payload}, path, compress=3)


def load_evaluation_artifacts(path: Path = EVALUATION_ARTIFACT_PATH) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or set(artifact) != {"metadata", "payload"}:
        raise ValueError("Evaluation artifact has an unsupported structure; regenerate it.")
    validate_provenance(artifact["metadata"])
    return artifact["payload"]
