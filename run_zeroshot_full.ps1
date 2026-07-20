# Full-corpus zero-shot ablation driver.
# Sequential train (45,839) -> val (1,082) -> test (1,243), each in clean+raw.
# Resume-safe: run_baseline.py appends to preds_<split>_<n>_<mode>.jsonl and
# picks up at the next undecoded row if a previous attempt was interrupted.
# Expected long-run cost ~9x the existing 5k run (5k-clean took ~30 min on
# the GTX 1080; budget ~4-5 hours per mode per split for the full 45k train).

$ErrorActionPreference = 'Stop'
$env:ZS_BATCH_SIZE      = '8'
$env:ZS_LIMIT           = '0'          # 0 = whole split
$env:ZS_MODES           = 'clean,raw'
$env:ZS_ERROR_ANALYSIS  = '1'
$env:ZS_EXTENDED_METRICS = '1'

# Smoke first: re-run val (1,082) to confirm timings and output schema on this
# machine before committing to the ~5h train pass.
Write-Host "===== VAL (1082) smoke =====" -ForegroundColor Cyan
$env:ZS_SPLIT = 'val'
.venv\Scripts\python.exe zero-shot\run_baseline.py
if ($LASTEXITCODE -ne 0) { throw "val smoke failed" }

# Then test (1,243) — small but new split, exercises the same resume code.
Write-Host "===== TEST (1,243) =====" -ForegroundColor Cyan
$env:ZS_SPLIT = 'test'
.venv\Scripts\python.exe zero-shot\run_baseline.py
if ($LASTEXITCODE -ne 0) { throw "test run failed" }

# Then the full train (45,839) — interrupts cleanly mid-run, you can re-launch.
Write-Host "===== TRAIN (45,839) =====" -ForegroundColor Cyan
$env:ZS_SPLIT = 'train'
.venv\Scripts\python.exe zero-shot\run_baseline.py
if ($LASTEXITCODE -ne 0) { throw "train run failed" }

Write-Host "All three splits done. See zero-shot/baseline/metrics_<split>_<n>.csv" -ForegroundColor Green
