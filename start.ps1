$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Check Python 3.11+
$pythonCmd = $null
try {
    $pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($pyVer -and [version]$pyVer -ge [version]"3.11") { $pythonCmd = "python" }
} catch {}
if (-not $pythonCmd) {
    try {
        $pyVer = & python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($pyVer -and [version]$pyVer -ge [version]"3.11") { $pythonCmd = "python3" }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python 3.11+ is required but not found." -ForegroundColor Red
    Write-Host "Install from https://www.python.org/downloads/ or run: winget install Python.Python.3.11" -ForegroundColor Yellow
    exit 1
}

# Check/install uv
$uvInstalled = $null
try { $uvInstalled = Get-Command uv -ErrorAction Stop } catch {}
if (-not $uvInstalled) {
    Write-Host "[SETUP] Installing uv..." -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile $env:TEMP\uv-install.ps1; & $env:TEMP\uv-install.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install uv." -ForegroundColor Red
        exit 1
    }
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
}

# First-time setup: sync dev dependencies
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path -LiteralPath $venvPath)) {
    Write-Host "[SETUP] Installing project dependencies..." -ForegroundColor Cyan
    uv sync --dev
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] uv sync failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "[SETUP] Ready! Running whichllm..." -ForegroundColor Green
} else {
    Write-Host "[INFO] Environment already set up." -ForegroundColor Green
}

# Run whichllm with all passed arguments
uv run whichllm @args
