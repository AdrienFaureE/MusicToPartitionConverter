# Installation locale (Windows) : venv, dépendances, vérifications, poids des modèles.
# Utilisation : powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 [-WithMt3] [-SkipModels]
param(
    [switch]$WithMt3,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label a échoué (code $LASTEXITCODE)." }
}

# Recherche de Python 3.11 ou 3.10
$python = $null
if ($env:PYTHON) { $python = @($env:PYTHON) }
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("3.11", "3.10")) {
        & py "-$version" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = @("py", "-$version"); break }
    }
}
if (-not $python -and (Get-Command python -ErrorAction SilentlyContinue)) { $python = @("python") }
if (-not $python) { throw "Python introuvable. Installez Python 3.10 ou 3.11 (voir docs\prerequisites.md)." }

$exe = $python[0]
$pyArgs = @($python | Select-Object -Skip 1)
& $exe @pyArgs -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 ou 3.11 requis. Indiquez-le avec `$env:PYTHON = 'C:\chemin\python.exe'." }

Write-Host "== Environnement virtuel (.venv) =="
if (-not (Test-Path ".venv")) { Invoke-Checked "La création du venv" { & $exe @pyArgs -m venv .venv } }
$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
Invoke-Checked "La mise à jour de pip" { & $venvPython -m pip install --upgrade pip "setuptools<81" wheel }

Write-Host "== Dépendances communes =="
Invoke-Checked "L'installation de requirements.txt" { & $venvPython -m pip install -r requirements.txt }
Invoke-Checked "L'installation de requirements-dev.txt" { & $venvPython -m pip install -r requirements-dev.txt }
Invoke-Checked "L'installation du projet" { & $venvPython -m pip install --no-deps -e . }

if ($WithMt3) {
    Write-Host "== MT3 (optionnel) =="
    Write-Host "MT3 n'est pas installable nativement sous Windows (tensorflow-text n'y est pas distribué)."
    Write-Host "Utilisez WSL2 et scripts/setup_local.sh --with-mt3 (voir docs\models.md)."
}

if (-not $SkipModels) {
    Write-Host "== Poids des modèles =="
    & $venvPython scripts\download_models.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "Téléchargement incomplet : relancez scripts\download_models.py." }
}

Write-Host "== Vérification des prérequis =="
& $venvPython scripts\check_prerequisites.py

Write-Host ""
Write-Host "Installation terminée. Lancez l'application avec :"
Write-Host "  .venv\Scripts\python.exe -m partition.cli serve"
