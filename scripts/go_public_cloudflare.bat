@echo off
rem DupeLens online through a Cloudflare Tunnel (free, no account): starts the app, then prints a public
rem https://....trycloudflare.com link. The link works while this window is open.
rem First time only: install cloudflared with   winget install --id Cloudflare.cloudflared
rem Then double-click this file, or run it from the repo folder:  scripts\go_public_cloudflare.bat
cd /d "%~dp0.."
where cloudflared >nul 2>&1 || (echo cloudflared is not installed. Run:  winget install --id Cloudflare.cloudflared   then run this file again. & pause & exit /b 1)
start "DupeLens app" cmd /k ".venv\Scripts\python.exe -m src.app"
echo Waiting for the app to load its models (about 30 seconds)...
:wait
timeout /t 5 /nobreak >nul
curl -s http://127.0.0.1:8000/api/health >nul 2>&1 || goto wait
echo.
echo App is running. Your public link is the https://....trycloudflare.com address printed below.
echo Keep this window open; close it to take the site offline.
echo.
cloudflared tunnel --url http://127.0.0.1:8000
