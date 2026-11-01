@echo off
rem DupeLens online from this laptop: starts the app, then opens a free public https link to it (localhost.run,
rem no account, nothing to install: uses the ssh built into Windows). The link works while this window is open.
rem Double-click this file, or run it from the repo folder:  scripts\go_public.bat
cd /d "%~dp0.."
start "DupeLens app" cmd /k ".venv\Scripts\python.exe -m src.app"
echo Waiting for the app to load its models (about 30 seconds)...
:wait
timeout /t 5 /nobreak >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1 || goto wait
echo.
echo App is running. Your public link is the https://....lhr.life address printed below.
echo Open it in any browser or phone. Close this window to take the site offline.
echo.
ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R 80:127.0.0.1:8000 nokey@localhost.run
