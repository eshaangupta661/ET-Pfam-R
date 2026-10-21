@echo off
REM =========================================================================
REM Commit and Push script for ET-Pfam-R
REM Usage:
REM   commit_and_push.bat [REMOTE_URL]
REM Example:
REM   commit_and_push.bat https://github.com/<username>/<repo-name>.git
REM =========================================================================

echo ===================================================
echo   Committing ET-Pfam-R to GitHub
echo ===================================================

REM Check if git is initialized
if not exist ".git" (
    echo [*] Initializing Git repository...
    git init
    git branch -M main
) else (
    echo [*] Git repository already initialized.
)

REM Add all files
echo [*] Staging all files...
git add .

REM Commit with descriptive title and body
echo [*] Committing changes...
git commit -m "feat(et-pfam-r): add modular pLM backbone, conformal calibration, FAISS retrieval, and explainability pipeline" -m "- Modular pLM abstraction layer supporting ESM-2 and baseline ESM-1b with [1280, L] shape guarantee" -m "- Conformal prediction wrapper (LAC and APS) for 90%% coverage prediction sets" -m "- Random Neighbor Score (RNS) for out-of-distribution (OOD) sequence detection" -m "- FAISS vector similarity index for top-5 nearest-neighbor reference homology retrieval" -m "- Sequence explainability: Input x Grad & Integrated Gradients saliency and Pfam domain mapping" -m "- Unified end-to-end CLI tool predict_et_pfam_r.py with JSON & CSV report export" -m "- Comprehensive test suite (run_all_tests.py) covering all 5 subsystems"

REM Set remote if URL provided as argument
if "%~1" neq "" (
    echo [*] Setting remote origin to %~1...
    git remote remove origin 2>nul
    git remote add origin %~1
    echo [*] Pushing to remote main branch...
    git push -u origin main
) else (
    REM Check if origin is already configured
    git remote get-url origin >nul 2>&1
    if %errorlevel% equ 0 (
        echo [*] Pushing to configured origin...
        git push -u origin main
    ) else (
        echo.
        echo [!] Notice: No remote origin URL was provided.
        echo [!] To push to your GitHub repo, run:
        echo     git remote add origin https://github.com/^<your-username^>/^<your-repo-name^>.git
        echo     git push -u origin main
    )
)

echo.
echo [*] Done!
pause
