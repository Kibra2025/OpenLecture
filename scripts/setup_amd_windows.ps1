[CmdletBinding()]
param(
    [string]$VenvPath = ".venv-amd",
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)

    Write-Host "[OpenLecture] $Message" -ForegroundColor Cyan
}

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$Description
    )

    Write-Step $Description
    & $FilePath @ArgumentList

    if ($LASTEXITCODE -ne 0) {
        $renderedArgs = $ArgumentList -join " "
        throw "Command failed: $FilePath $renderedArgs"
    }
}

function Resolve-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($versionArg in @("-3.12", "-3.11", "-3.10")) {
            try {
                & py $versionArg -c "import sys" | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    return @{
                        FilePath = "py"
                        ArgumentPrefix = @($versionArg)
                    }
                }
            }
            catch {
            }
        }

        return @{
            FilePath = "py"
            ArgumentPrefix = @()
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            FilePath = "python"
            ArgumentPrefix = @()
        }
    }

    throw "Python launcher not found. Install Python 3.10+ and try again."
}

function Resolve-AbsolutePath {
    param(
        [string]$BasePath,
        [string]$CandidatePath
    )

    if ([System.IO.Path]::IsPathRooted($CandidatePath)) {
        return [System.IO.Path]::GetFullPath($CandidatePath)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $CandidatePath))
}

$projectRoot = Resolve-AbsolutePath -BasePath $PSScriptRoot -CandidatePath ".."
$resolvedVenvPath = Resolve-AbsolutePath -BasePath $projectRoot -CandidatePath $VenvPath
$pythonCommand = Resolve-PythonCommand
$pythonLauncher = $pythonCommand.FilePath
$pythonLauncherArgs = @($pythonCommand.ArgumentPrefix)
$venvPython = Join-Path $resolvedVenvPath "Scripts\python.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path $resolvedVenvPath)) {
        Invoke-CheckedCommand `
            -FilePath $pythonLauncher `
            -ArgumentList ($pythonLauncherArgs + @("-m", "venv", $resolvedVenvPath)) `
            -Description "Creating virtual environment at $resolvedVenvPath"
    }
    else {
        Write-Step "Using existing virtual environment at $resolvedVenvPath"
    }

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment Python executable not found: $venvPython"
    }

    Invoke-CheckedCommand `
        -FilePath $venvPython `
        -ArgumentList @("-m", "pip", "install", "--upgrade", "pip") `
        -Description "Upgrading pip"

    Invoke-CheckedCommand `
        -FilePath $venvPython `
        -ArgumentList @("-m", "pip", "install", "torch-directml") `
        -Description "Installing torch-directml for DirectML acceleration"

    Invoke-CheckedCommand `
        -FilePath $venvPython `
        -ArgumentList @("-m", "pip", "install", "-e", ".[transformers]") `
        -Description "Installing OpenLecture with transformers support"

    if (-not $SkipSmokeTest) {
        $smokeTestScript = @'
import sys
import torch
from openlecture.transcribe import _resolve_transformers_device

device = _resolve_transformers_device("dml")
x = torch.tensor([1.0]).to(device)
y = torch.tensor([2.0]).to(device)

print(f"python={sys.version.split()[0]}")
print(f"device={device}")
print(f"sum={(x + y).item():.1f}")
'@

        Write-Step "Running DirectML smoke test"
        $smokeTestScript | & $venvPython -

        if ($LASTEXITCODE -ne 0) {
            throw "DirectML smoke test failed."
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate it with:" -ForegroundColor Green
Write-Host "  $resolvedVenvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run OpenLecture on the AMD GPU with:" -ForegroundColor Green
Write-Host "  openlecture .\TestAudio.mp3 --backend transformers --device dml --compute-type auto --model small"
Write-Host ""
Write-Host "Note: ffmpeg still needs to be available on the system." -ForegroundColor Yellow
