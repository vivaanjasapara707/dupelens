@echo off
rem Daily DupeLens price check, for Windows Task Scheduler. Log: data\tracker.log
rem Register once:  schtasks /Create /SC DAILY /TN "DupeLens price check" /TR "\"%~f0\"" /ST 09:00
rem Remove:         schtasks /Delete /TN "DupeLens price check" /F
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m src.tracker --update >> "data\tracker.log" 2>&1
