$env:PYTHONIOENCODING='utf-8'
Set-Location C:\Projects\lip_reading
Start-Process -FilePath .venv\Scripts\python.exe `
  -ArgumentList '-u','direct_baseline.py','--n','5000','--max_train','4000','--epochs','8' `
  -RedirectStandardOutput (Join-Path $PWD 'direct_baseline_stdout.log') `
  -RedirectStandardError  (Join-Path $PWD 'direct_baseline_stderr.log') `
  -WorkingDirectory $PWD -WindowStyle Hidden
Write-Host 'launched'