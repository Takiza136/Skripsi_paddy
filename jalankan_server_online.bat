@echo off
title PadiScan - Server Online Cloudflare Tunnel
echo ========================================================
echo         MENYALAKAN SERVER PADISCAN ONLINE (CLOUDFLARE)
echo ========================================================
echo.
echo 1. Menjalankan backend FastAPI di background...
start "FastAPI Backend" /min cmd /c "cd /d %~dp0web\backend && venv\Scripts\python.exe -m uvicorn main:app --port 8000"
timeout /t 4 /nobreak >nul

echo 2. Menghubungkan ke Cloudflare Tunnel...
echo.
echo ========================================================
echo CATATAN PENTING:
echo - Jangan tutup jendela ini selama web ingin diakses online.
echo - Salin link https://...trycloudflare.com yang muncul di bawah
echo   ke web/frontend/config.js jika link berubah.
echo ========================================================
echo.
%~dp0cloudflared.exe tunnel --url http://127.0.0.1:8000
