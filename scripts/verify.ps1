param(
    [switch]$SkipSlow
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$candidateRoots = @(
    $projectRoot,
    (Split-Path -Parent (Split-Path -Parent $projectRoot))
)
$venvRoot = $candidateRoots |
    ForEach-Object { Join-Path $_ ".venv" } |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ "Scripts\python.exe") } |
    Select-Object -First 1
$python = if ($venvRoot) { Join-Path $venvRoot "Scripts\python.exe" } else { Join-Path $projectRoot ".venv\Scripts\python.exe" }

if (-not (Test-Path -LiteralPath $python)) {
    throw "Pinned interpreter not found at $python. Create .venv and install requirements.txt first."
}

$pythonPath = (Resolve-Path -LiteralPath $python).Path
$expectedPrefix = ((Resolve-Path -LiteralPath (Join-Path $venvRoot "Scripts")).Path)
if (-not $pythonPath.StartsWith($expectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to run outside the project .venv."
}

Push-Location $projectRoot
try {
    function Invoke-Checked {
        param(
            [string]$Executable,
            [string[]]$Arguments
        )
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Verification command failed with exit code $LASTEXITCODE`: $Executable $Arguments"
        }
    }

    Invoke-Checked $python @("-m", "pip", "check")
    Invoke-Checked $python @("-m", "ruff", "check", ".")
    Invoke-Checked $python @("-m", "ruff", "format", "--check", ".")
    $pytestArgs = @("-m", "pytest", "-v")
    if ($SkipSlow) { $pytestArgs += "-m"; $pytestArgs += "not slow" }
    Invoke-Checked $python $pytestArgs
}
finally {
    Pop-Location
}
