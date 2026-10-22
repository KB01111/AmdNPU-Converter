#Requires -Version 5.1
<#
.SYNOPSIS
  Bootstrap Python 3.11 worker venvs with uv.
.DESCRIPTION
  Creates workers/.venv (base), plus optional isolated envs:
    workers/.venv-winml  (requirements-winml.txt)
    workers/.venv-quark  (requirements-quark.txt)
  Prints winml.exe discovery guidance.
#>
param(
    [string]$WorkersDir = $PSScriptRoot,
    [switch]$WinmlOnly,
    [switch]$QuarkOnly,
    [switch]$SkipBase
)

$ErrorActionPreference = "Stop"

function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "uv")) {
    Write-Error "uv not found on PATH. Install from https://docs.astral.sh/uv/ then re-run."
    exit 2
}

function New-WorkerVenv($venvPath, $requirementsFile) {
    Write-Host "==> venv: $venvPath"
    if (-not (Test-Path -LiteralPath $venvPath)) {
        & uv venv --python 3.11 $venvPath
        if (-not $?) { throw "uv venv failed for $venvPath" }
    } else {
        Write-Host "    exists, reusing."
    }
    $pipPython = Join-Path $venvPath "Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $pipPython)) {
        # non-Windows layout fallback
        $pipPython = Join-Path $venvPath "bin/python"
    }
    if ($requirementsFile -and (Test-Path -LiteralPath $requirementsFile)) {
        Write-Host "    installing $requirementsFile ..."
        & uv pip install --python $pipPython -r $requirementsFile
        if (-not $?) { throw "uv pip install failed for $requirementsFile" }
    } else {
        Write-Host "    installing base pyproject ..."
        & uv pip install --python $pipPython -e $WorkersDir
        if (-not $?) { throw "base worker installation failed; run: uv pip install --python $pipPython -e $WorkersDir" }
    }
}

$baseVenv  = Join-Path $WorkersDir ".venv"
$winmlVenv = Join-Path $WorkersDir ".venv-winml"
$quarkVenv = Join-Path $WorkersDir ".venv-quark"

try {
    if ($QuarkOnly) {
        New-WorkerVenv $quarkVenv (Join-Path $WorkersDir "requirements-quark.txt")
    } elseif ($WinmlOnly) {
        New-WorkerVenv $winmlVenv (Join-Path $WorkersDir "requirements-winml.txt")
    } else {
        if (-not $SkipBase) {
            New-WorkerVenv $baseVenv $null
        }
        New-WorkerVenv $winmlVenv (Join-Path $WorkersDir "requirements-winml.txt")
        New-WorkerVenv $quarkVenv (Join-Path $WorkersDir "requirements-quark.txt")
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "=== winml.exe discovery ==="
Write-Host "Lookup order used by workers/winml/run.py:"
Write-Host "  1. env WINML_BIN"
Write-Host "  2. workers/.venv-winml/Scripts/winml.exe"
Write-Host "  3. workers/.venv/Scripts/winml.exe"
Write-Host "  4. PATH (Get-Command winml.exe)"
Write-Host ""
$found = $false
if ($env:WINML_BIN -and (Test-Path -LiteralPath $env:WINML_BIN)) {
    Write-Host "WINML_BIN=$env:WINML_BIN (exists)"
    $found = $true
}
foreach ($c in @(
    (Join-Path $winmlVenv "Scripts/winml.exe"),
    (Join-Path $baseVenv "Scripts/winml.exe")
)) {
    if (Test-Path -LiteralPath $c) { Write-Host "found: $c"; $found = $true }
}
$cmd = Get-Command winml.exe -ErrorAction SilentlyContinue
if ($cmd) { Write-Host "found on PATH: $($cmd.Source)"; $found = $true }
if (-not $found) {
    Write-Warning "winml.exe not found. Install WinML CLI: uv pip install -r requirements-winml.txt (in .venv-winml), or set `$env:WINML_BIN='C:\path\to\winml.exe'."
    Write-Host "Verify with: workers/.venv-winml/Scripts/python.exe workers/winml/run.py sys --output-json"
} else {
    Write-Host "OK: winml.exe discoverable. Try: workers/.venv-winml/Scripts/python.exe workers/winml/run.py sys --output-json"
}
