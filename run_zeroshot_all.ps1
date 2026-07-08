# ============================================================================
#  Zero-Shot Baseline (ISOLATED) -- plain Llama-3.2-3B, all splits
# ============================================================================
#  Runs zero_shot_baseline.py: a self-contained baseline decoupled from the
#  CPT decoder (no LoRA wrapper, no contrastive tooling). Reuses only the
#  error-analysis module for the Stage-2/3 report. Writes, per split:
#     zeroshot_<split>_<N>_view.txt          input phonemes | predicted | target
#     metrics_summary.csv                    WER / CER / BLEU-4 / ExactMatch
#     zeroshot_<split>_<N>_error_report.json substitution error patterns
#
#  Plain greedy decoding (apples-to-apples with the existing baseline JSON).
#  Resume-safe: each split streams to a .jsonl side-car; Ctrl+C and relaunch
#  picks up at the next unwritten row.
# ============================================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# --- Config -----------------------------------------------------------------
$env:ZS_MODEL          = "meta-llama/Llama-3.2-3B"   # ungated mirror: unsloth/Llama-3.2-3B
$env:ZS_SPLIT          = "all"                        # val -> test -> train
$env:ZS_OUTPUT_DIR     = "zero-shot/baseline"
$env:ZS_ERROR_ANALYSIS = "1"                          # set to "0" to skip on the 45k train split
# ZS_BATCH_SIZE / ZS_MAX_NEW_TOK left at defaults (8 / 34 on CUDA)

# --- Logging ----------------------------------------------------------------
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir  = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "zeroshot_baseline_$stamp.log"

Write-Host "============================================================"
Write-Host "  Zero-Shot Baseline (ISOLATED, plain base model) -- split=all"
Write-Host "  Output : $($env:ZS_OUTPUT_DIR)"
Write-Host "  Log    : $logFile"
Write-Host "============================================================"

# --- Run --------------------------------------------------------------------
& ".\.venv\Scripts\python.exe" -u zero_shot_baseline.py 2>&1 |
    Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Done. Full log: $logFile"
