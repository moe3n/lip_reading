$ErrorActionPreference = 'Continue'

# Set environment variables for the launch.
$env:CUDA_VISIBLE_DEVICES = '0'   # pin to one GPU; monkeypatch in load_model()
                                       # lets accelerate's .to() proceed for bnb
                                       # (was: '' = both visible = PCIe split)
$env:ZS_SPLIT             = 'train'
$env:ZS_SPLIT_OFFSET      = '0'
$env:ZS_SPLIT_STRIDE      = '1'
$env:ZS_ERROR_ANALYSIS    = '1'
$env:ZS_EXTENDED_METRICS  = '1'
$env:ZS_DEDUP_TRAIN       = '0'
$env:ZS_LIMIT             = ''   # full split, not a smoke-test residue
$env:ZS_MODES             = 'clean,raw'

$ts = Get-Date -Format yyyyMMdd_HHmmss
$logPath = "logs\zeroshot_train_full_ea_$ts.log"
$errPath = "logs\zeroshot_train_full_ea_${ts}_err.log"

Write-Host "[$(Get-Date -Format HH:mm:ss)] launching: log=$logPath"

# Use cmd /c start /B to truly detach the process from this shell session, so
# subsequent monitoring commands in the parent shell don't queue behind the
# long-running python job. stdout/stderr go to separate files via the >>-prefix
# redirection that cmd.exe supports.
$cmd = @"
start "train_ea" /B /MIN cmd /c ".venv\Scripts\python.exe -u zero-shot\run_baseline.py 1>>$logPath 2>>$errPath"
"@
Write-Host "[$(Get-Date -Format HH:mm:ss)] kicking off: $cmd"
cmd /c $cmd

Write-Host "[$(Get-Date -Format HH:mm:ss)] dispatched. Sleep 10s to confirm startup."
Start-Sleep -Seconds 10
if (Test-Path $logPath) {
    $len = (Get-Item $logPath).Length
    Write-Host "[$(Get-Date -Format HH:mm:ss)] log size = $len bytes"
} else {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] log not yet created"
}