param(
    [switch]$SkipSlow
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Pinned interpreter not found at $python. Create .venv and install requirements.txt first."
}

$pythonPath = (Resolve-Path -LiteralPath $python).Path
$expectedPrefix = ((Resolve-Path -LiteralPath (Join-Path $projectRoot ".venv\Scripts")).Path)
if (-not $pythonPath.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to run outside the project .venv."
}

Push-Location $projectRoot
try {
    & $python -m pip check
    & $python -m ruff check .
    & $python -m ruff format --check .
    $pytestArgs = @("-m", "pytest", "-v")
    if ($SkipSlow) { $pytestArgs += "-m"; $pytestArgs += "not slow" }
    & $python @pytestArgs
}
finally {
    Pop-Location
}
