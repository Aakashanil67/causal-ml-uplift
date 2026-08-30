"""Versioned contract for numerical results shared by reports and the app."""

import hashlib
import json
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

RESULTS_SCHEMA_VERSION = 1
REQUIRED_SECTIONS = {"metadata", "headline", "interactions", "ranking", "policy"}
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
        digest.update(path.read_bytes())
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


def validate_results(results: dict) -> None:
    if results.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported results schema version: {results.get('schema_version')}; "
            f"expected {RESULTS_SCHEMA_VERSION}."
        )
    missing = REQUIRED_SECTIONS - set(results)
    if missing:
        raise ValueError(f"Results missing sections: {', '.join(sorted(missing))}")

    metadata = results["metadata"]
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


def save_results(results: dict, path: Path = RESULTS_PATH) -> None:
    validate_results(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(rendered, encoding="utf-8")


def load_results(path: Path = RESULTS_PATH) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))
    validate_results(results)
    return results


def save_evaluation_artifacts(payload: dict, path: Path = EVALUATION_ARTIFACT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"metadata": build_provenance(), "payload": payload}, path, compress=3)


def load_evaluation_artifacts(path: Path = EVALUATION_ARTIFACT_PATH) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or set(artifact) != {"metadata", "payload"}:
        raise ValueError("Evaluation artifact has an unsupported structure; regenerate it.")
    validate_provenance(artifact["metadata"])
    return artifact["payload"]
