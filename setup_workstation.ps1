# PERSUADE/LLM Phase-1 MVP -- Workstation setup script.
#
# Designed for: HP Z6 G5 A Workstation (LAB-LIHID), Windows 64-bit,
# NVIDIA RTX PRO 4000 Blackwell (24 GB VRAM).
#
# What this script does, in order:
#   0. Sanity-check the environment (python, kaggle CLI + creds, nvidia driver)
#   1. Create a Python venv at experiments/persuade_aes/.venv/
#   2. Install PyTorch + CUDA 12.4 wheels (Blackwell-compatible)
#   3. Install the remaining packages from requirements_persuade.txt
#   4. Download spacy model + nltk data
#   5. Download the two Kaggle auxiliary datasets (~2 GB total)
#   6. Verify everything imports and CUDA is visible
#
# Total time estimate: 15-30 minutes (network-bound).
# Run from the project root (NOT from inside experiments/persuade_aes/).

# NOTE: deliberately NOT using 'Stop' here. Under PS 5.1 + ErrorAction
# 'Stop', any stderr line from a native command (pip cache warnings,
# kaggle progress, etc.) is wrapped as a NativeCommandError and halts
# the script even when the command exited 0. We use 'Continue' and
# rely on explicit $LASTEXITCODE checks via Check-Exit.
$ErrorActionPreference = 'Continue'

function Assert-ExitOk {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "[$Step] failed with exit code $LASTEXITCODE"
    }
}

# ============================================================
# 0. Resolve repo root and switch to it
# ============================================================
$repoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
Set-Location $repoRoot
Write-Host "[INFO] Repo root: $repoRoot" -ForegroundColor Cyan

# ============================================================
# 0a. Python check (3.10-3.12 preferred)
# ============================================================
try {
    $pyVer = (python --version 2>&1).ToString()
    Write-Host "[INFO] Python: $pyVer" -ForegroundColor Cyan
} catch {
    throw "python not found on PATH. Install Python 3.10-3.12 first."
}

# ============================================================
# 0b. Kaggle credentials check
# ============================================================
# The `kaggle` CLI itself is installed inside the venv via
# requirements_persuade.txt (see step 3), so we do NOT preflight-check
# it on the system PATH -- that would create a false negative on a clean
# workstation. We only verify that the credentials file exists, since
# the venv-installed kaggle CLI reads from the same standard location.
$kaggleJson = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleJson)) {
    throw "Kaggle credentials missing at $kaggleJson. Copy kaggle.json from your laptop into this path."
}
Write-Host "[INFO] Kaggle credentials: found at $kaggleJson" -ForegroundColor Cyan

# ============================================================
# 0c. NVIDIA driver check
# ============================================================
try {
    Write-Host "[INFO] GPU info:" -ForegroundColor Cyan
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} catch {
    Write-Warning "nvidia-smi not found -- PyTorch will install but CUDA paths will not work."
}

# ============================================================
# 1. Create / activate Python venv
# ============================================================
# We avoid `. Activate.ps1` (dot-source) and instead set the two env
# vars Activate.ps1 itself sets -- PATH and VIRTUAL_ENV. This sidesteps
# PowerShell parser edge cases around dot-source on some hosts.
$venvDir = Join-Path $repoRoot "experiments\persuade_aes\.venv"
$venvScripts = Join-Path $venvDir "Scripts"
$venvPython = Join-Path $venvScripts "python.exe"
Write-Host "[INFO] venvDir = $venvDir" -ForegroundColor Cyan

if (-not (Test-Path $venvDir)) {
    # Use Python 3.12 explicitly via the `py` launcher. System default is
    # 3.14 (too new -- tokenizers/PyO3 symbol mismatches; see commit log
    # 7073e37 for the failure that drove us off 3.14). Python 3.12 is
    # broadly supported by the ML wheel ecosystem.
    Write-Host "[STEP] Creating venv with Python 3.12 ..." -ForegroundColor Yellow
    & py -3.12 -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "py -3.12 -m venv failed (exit $LASTEXITCODE). Ensure Python 3.12 is installed via python.org." }
} else {
    Write-Host "[INFO] Venv directory exists, reusing." -ForegroundColor Cyan
}

if (-not (Test-Path $venvPython)) {
    throw "python.exe not found at $venvPython. The venv is corrupt; remove $venvDir and rerun."
}

# Manual activation -- equivalent to Scripts\Activate.ps1 for this script's scope.
$env:VIRTUAL_ENV = $venvDir
$env:PATH = "$venvScripts;$env:PATH"
Write-Host "[INFO] Activated. VIRTUAL_ENV = $env:VIRTUAL_ENV" -ForegroundColor Cyan
Write-Host "[INFO] python in use: $(& python -c 'import sys; print(sys.executable)')" -ForegroundColor Cyan

# ============================================================
# 1b. PyO3 forward-compatibility flag (Python 3.14 workaround)
# ============================================================
# tokenizers / safetensors / polars use PyO3 0.22.5 which officially
# supports Python <= 3.13. Without this flag they refuse to compile on
# Python 3.14 even with MSVC + Rust toolchain installed. Setting
# PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 tells PyO3 to build against the
# stable ABI (ABI3) and skip the version-too-new check.
# Risk: the resulting wheels were not tested by PyO3 against 3.14 --
# could see subtle runtime issues later. We accept this risk for now
# and pivot to Python 3.13 if it materializes.
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
Write-Host "[INFO] PyO3 forward-compat enabled (PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1)" -ForegroundColor Cyan

# ============================================================
# 2. Pip upgrade + PyTorch (CUDA 12.8 -- supports Blackwell + Python 3.14)
# ============================================================
# We use cu128 (not cu124) because the cu124 wheel index stops at cp313
# while the workstation runs Python 3.14. cu128 publishes cp314 wheels
# (torch 2.11+) and is fully Blackwell-compatible. Verified empirically
# 2026-05-13 against https://download.pytorch.org/whl/cu128/torch/.
Write-Host "[STEP] Upgrading pip + wheel ..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel
Assert-ExitOk "pip upgrade"

Write-Host "[STEP] Installing PyTorch with CUDA 12.8 (~3 GB download) ..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
Assert-ExitOk "PyTorch install"

$cudaCheck = python -c "import torch; ok = torch.cuda.is_available(); dev = torch.cuda.get_device_name(0) if ok else 'CPU only'; print(f'CUDA_OK={ok}; DEVICE={dev}')"
Assert-ExitOk "CUDA visibility check"
Write-Host "[CHECK] $cudaCheck" -ForegroundColor Magenta

# ============================================================
# 3. Install the rest from requirements_persuade.txt
# ============================================================
Write-Host "[STEP] Installing remaining Python packages ..." -ForegroundColor Yellow
pip install -r experiments\persuade_aes\requirements_persuade.txt
Assert-ExitOk "requirements install"

# ============================================================
# 4. spacy model + nltk data
# ============================================================
Write-Host "[STEP] Downloading spacy en_core_web_sm ..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm
Assert-ExitOk "spacy en_core_web_sm download"

Write-Host "[STEP] Downloading nltk data (vader_lexicon, punkt, punkt_tab, averaged_perceptron_tagger) ..." -ForegroundColor Yellow
python -m nltk.downloader -q vader_lexicon punkt punkt_tab averaged_perceptron_tagger
Assert-ExitOk "nltk data download"

# ============================================================
# 5. Kaggle auxiliary datasets
# ============================================================
$auxRoot = "data\persuade_aux"
New-Item -ItemType Directory -Force -Path "$auxRoot\feedback_data" | Out-Null
New-Item -ItemType Directory -Force -Path "$auxRoot\sent_debsmall" | Out-Null

if (-not (Test-Path "$auxRoot\feedback_data\feedback_data.csv")) {
    Write-Host "[STEP] Downloading datafan07/feedback-data (~3 MB) ..." -ForegroundColor Yellow
    kaggle datasets download -d datafan07/feedback-data -p "$auxRoot\feedback_data" --unzip
    Assert-ExitOk "feedback-data download"
} else {
    Write-Host "[INFO] feedback-data already present, skipping" -ForegroundColor Cyan
}

if (-not (Test-Path "$auxRoot\sent_debsmall\deberta_small_trained")) {
    Write-Host "[STEP] Downloading datafan07/sent-debsmall (~2 GB, allow 5-15 min) ..." -ForegroundColor Yellow
    kaggle datasets download -d datafan07/sent-debsmall -p "$auxRoot\sent_debsmall" --unzip
    Assert-ExitOk "sent-debsmall download"
} else {
    Write-Host "[INFO] sent-debsmall already present, skipping" -ForegroundColor Cyan
}

# ============================================================
# 6. Verification
# ============================================================
Write-Host ""
Write-Host "=== Verification ===" -ForegroundColor Green

python -c "import torch, transformers, sentence_transformers, spacy, nltk, textstat, lightgbm, polars; print('All key imports: OK')"
Assert-ExitOk "package import sanity check"

$gpu = python -c "import torch; ok = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if ok else 'CPU'; print(f'CUDA: {ok}; device: {name}; torch: {torch.__version__}')"
Assert-ExitOk "CUDA final check"
Write-Host $gpu -ForegroundColor Magenta

Write-Host ""
Write-Host "Auxiliary data layout:" -ForegroundColor Cyan
Get-ChildItem $auxRoot -Recurse -File |
    Select-Object @{N='Path';E={$_.FullName.Substring($repoRoot.Length+1)}}, @{N='Size_MB';E={[math]::Round($_.Length/1MB, 2)}} |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Venv lives at: $venvDir" -ForegroundColor Yellow
Write-Host "To activate manually in a new shell:  . $venvScripts\Activate.ps1" -ForegroundColor Yellow
