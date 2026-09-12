import hashlib
import re

from src.config import ROOT


def test_repository_has_code_license_and_model_documentation():
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "models" / "README.md").is_file()
    assert (ROOT / "scripts" / "verify.ps1").is_file()


def test_verification_script_requires_the_project_virtual_environment():
    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe" in script
    assert '"pip", "check"' in script
    assert "pytest" in script
    assert "LASTEXITCODE" in script


def test_documented_model_hashes_match_committed_artifacts():
    model_dir = ROOT / "models"
    model_readme = (model_dir / "README.md").read_text(encoding="utf-8")

    for filename in ("causal_forest.joblib", "evaluation_artifacts.joblib"):
        documented = re.search(rf"`{re.escape(filename)}`: `([0-9a-f]{{64}})`", model_readme)
        assert documented, f"Missing SHA-256 for {filename}"
        actual = hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        assert documented.group(1) == actual
