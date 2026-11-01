@echo off
rem Daily DupeLens price check (Windows Task Scheduler). Log: data	racker.log
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m src.tracker --update >> data	racker.log 2>&1
