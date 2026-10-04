@echo off
title AI Exam Proctoring Surveillance System
echo ==============================================================================
echo  Launching AI Examination Behavior Proctoring System Desktop GUI...
echo ==============================================================================
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo [Notice] Application closed with an exit code.
    pause
)
