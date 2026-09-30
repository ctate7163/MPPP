@echo off
REM ======================================================================================================
REM MPPP - take a delivery from Claude into git and push it to GitHub (v0p50).
REM
REM Claude copies the changed files into this MPPP folder and the new history into _transfer\mppp_latest.bundle.
REM This adopts that history (only if it continues yours: fast-forward), deletes the files it removed, restores
REM tracked files that are missing, shows what differs, then github_push.bat: pull from GitHub first, then push.
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
if errorlevel 1 goto :merge_claude
git reset -q --mixed claude/main
goto :adopted
:merge_claude
REM this copy has commits Claude's history lacks (e.g. a merge of GitHub): merge Claude's delivery in
echo this copy has its own commits since the last delivery: merging Claude's delivery in
git stash -q -u -- src scripts tests notebooks docs studies README.md CHANGELOG.md pyproject.toml >nul 2>nul
git merge --no-edit claude/main
if errorlevel 1 (git merge --abort & echo merge conflict - nothing changed; tell Claude & pause & exit /b 1)
git stash list | find "stash@{0}" >nul && echo (the files in this folder before the merge are kept in: git stash list)
:adopted
for /f "delims=" %%F in ('git diff --name-only --no-renames --diff-filter=D %OLD% HEAD') do if exist "%%F" (echo removed: %%F& del /q "%%F")
for /f "delims=" %%F in ('git ls-files --deleted') do git checkout -q HEAD -- "%%F"
echo.
git log --oneline -1
echo differences from the delivered version (should be empty unless you edited files):
git status --short
echo.
call "%~dp0github_push.bat"
pause
