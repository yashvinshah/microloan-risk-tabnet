@echo off
REM Batch script for Windows for the micro-loan default risk prediction pipeline

setlocal enabledelayedexpansion

if "%1"=="" (
    echo Micro-Loan Default Risk Prediction Pipeline
    echo.
    echo Available commands:
    echo   run.bat install        - Install dependencies
    echo   run.bat test           - Run component tests
    echo   run.bat quick-start    - Run quick start examples
    echo   run.bat run            - Run full pipeline (no HPO)
    echo   run.bat run-hpo        - Run full pipeline with HPO
    echo   run.bat clean          - Clean outputs and cache
    echo   run.bat docs           - Display documentation
    echo   run.bat help           - Show this help message
    goto :eof
)

if "%1"=="install" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Dependencies installed successfully!
    goto :eof
)

if "%1"=="test" (
    echo Running component tests...
    python test_pipeline.py
    goto :eof
)

if "%1"=="quick-start" (
    echo Running quick start examples...
    python -c "from quick_start import run_all_examples; run_all_examples()"
    goto :eof
)

if "%1"=="run" (
    echo Running full pipeline (no HPO)...
    python main.py
    goto :eof
)

if "%1"=="run-hpo" (
    echo Running full pipeline with HPO...
    python main.py --run_hpo
    goto :eof
)

if "%1"=="clean" (
    echo Cleaning outputs...
    rmdir /s /q outputs 2>nul
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    del /s /q *.pyc >nul 2>&1
    echo Clean complete!
    goto :eof
)

if "%1"=="docs" (
    echo Displaying documentation...
    python DOCUMENTATION.py
    goto :eof
)

if "%1"=="help" (
    echo Micro-Loan Default Risk Prediction Pipeline
    echo.
    echo Available commands:
    echo   run.bat install        - Install dependencies
    echo   run.bat test           - Run component tests
    echo   run.bat quick-start    - Run quick start examples
    echo   run.bat run            - Run full pipeline (no HPO)
    echo   run.bat run-hpo        - Run full pipeline with HPO
    echo   run.bat clean          - Clean outputs and cache
    echo   run.bat docs           - Display documentation
    goto :eof
)

echo Unknown command: %1
goto :eof
