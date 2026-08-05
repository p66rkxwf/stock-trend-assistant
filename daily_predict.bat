@echo off
rem Daily online prediction: record pool predictions + refresh report (run after market close).
rem Double-click: results stay on screen. Scheduled mode (arg "scheduled"): append to logs\daily_predict.log.
rem Keep this file ASCII-only: cmd.exe parses batch files with the ANSI codepage, not UTF-8.
cd /d "%~dp0backend"
set PYTHONIOENCODING=utf-8
if "%~1"=="scheduled" (
  if not exist "%~dp0logs" mkdir "%~dp0logs"
  echo ===== %date% %time% ===== >> "%~dp0logs\daily_predict.log"
  .venv\Scripts\python.exe -m stockta.ml.record_predictions >> "%~dp0logs\daily_predict.log" 2>&1
  .venv\Scripts\python.exe -m stockta.ml.report_predictions >> "%~dp0logs\daily_predict.log" 2>&1
  rem Cross-sectional relative-strength online track (Rank IC): record live scores + refresh report.
  .venv\Scripts\python.exe -m stockta.ml.record_rank_predictions >> "%~dp0logs\daily_predict.log" 2>&1
  .venv\Scripts\python.exe -m stockta.ml.report_rank_predictions >> "%~dp0logs\daily_predict.log" 2>&1
) else (
  .venv\Scripts\python.exe -m stockta.ml.record_predictions
  .venv\Scripts\python.exe -m stockta.ml.report_predictions
  .venv\Scripts\python.exe -m stockta.ml.record_rank_predictions
  .venv\Scripts\python.exe -m stockta.ml.report_rank_predictions
  pause
)
