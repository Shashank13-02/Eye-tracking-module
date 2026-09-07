@echo off
title 3D Webcam Tracker
cd /d "%~dp0"
echo ========================================================
echo Starting 3D Webcam Eye Tracker...
echo - Press 'c' while looking at the screen to calibrate
echo - Press 'o' to toggle Always-On-Top
echo - Press 'q' in any window to quit
echo ========================================================
python MonitorTracking.py
pause
