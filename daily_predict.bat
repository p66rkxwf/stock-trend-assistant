@echo off
rem 每日線上預測累積：記錄股票池預測 + 更新實證報告（收盤後執行，約 1-2 分鐘）
rem 手動雙擊：跑完停在畫面上；排程呼叫（帶 scheduled 參數）：結果寫入 logs\daily_predict.log
cd /d "%~dp0backend"
set PYTHONIOENCODING=utf-8
if "%~1"=="scheduled" (
  if not exist "%~dp0logs" mkdir "%~dp0logs"
  echo ===== %date% %time% ===== >> "%~dp0logs\daily_predict.log"
  .venv\Scripts\python.exe -m stockta.ml.record_predictions >> "%~dp0logs\daily_predict.log" 2>&1
  .venv\Scripts\python.exe -m stockta.ml.report_predictions >> "%~dp0logs\daily_predict.log" 2>&1
) else (
  .venv\Scripts\python.exe -m stockta.ml.record_predictions
  .venv\Scripts\python.exe -m stockta.ml.report_predictions
  pause
)
