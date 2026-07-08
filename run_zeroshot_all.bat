@echo off
REM Double-click launcher for the zero-shot all-split baseline run.
REM Runs the PowerShell script with execution policy bypassed and keeps the
REM window open (-NoExit) so you can watch progress / read errors.
cd /d "%~dp0"
powershell -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_zeroshot_all.ps1"
