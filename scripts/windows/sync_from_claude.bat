@echo off
REM ======================================================================================================
REM MPPP - take a delivery from Claude into git and push it to GitHub (v0p50).
REM
REM Claude copies the changed files into this MPPP folder and the new history into _transfer\mppp_latest.bundle.
REM This adopts that history (only if it continues yours: fast-forward), deletes the files it removed, restores
REM tracked files that are missing, shows what differs, and pushes main and the tags to GitHub.
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || (pause & exit /b 1)
set "BUNDLE=%MPPP_HOME%\_transfer\mppp_latest.bundle"
if not exist ".git" (echo not a git working copy yet: run setup_github.bat first & pause & exit /b 1)
if not exist "%BUNDLE%" (echo %BUNDLE% not found & pause & exit /b 1)
for /f %%H in ('git rev-parse HEAD') do set "OLD=%%H"
git fetch -q "%BUNDLE%" "+refs/heads/main:refs/remotes/claude/main" "+refs/tags/*:refs/tags/*" || (pause & exit /b 1)
git merge-base --is-ancestor HEAD claude/main
if errorlevel 1 (
  echo Claude's history does not continue yours (you committed here since the last delivery^).
  echo Nothing changed. Tell Claude, or merge by hand:  git merge claude/main
  pause & exit /b 1
)
git reset -q --mixed claude/main
for /f "delims=" %%F in ('git diff --name-only --no-renames --diff-filter=D %OLD% HEAD') do if exist "%%F" (echo removed: %%F& del /q "%%F")
for /f "delims=" %%F in ('git ls-files --deleted') do git checkout -q HEAD -- "%%F"
echo.
git log --oneline -1
echo differences from the delivered version (should be empty unless you edited files):
git status --short
echo.
git push -q origin main --tags && echo pushed to GitHub.
pause
