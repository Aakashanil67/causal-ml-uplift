from src.config import ROOT


def test_repository_has_code_license_and_model_documentation():
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "models" / "README.md").is_file()
    assert (ROOT / "scripts" / "verify.ps1").is_file()


def test_verification_script_requires_the_project_virtual_environment():
    script = (ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe" in script
    assert "pip check" in script
    assert "pytest" in script
