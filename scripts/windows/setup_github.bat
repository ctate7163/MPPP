@echo off
REM ======================================================================================================
REM MPPP - one-time set-up of git and the private GitHub repository (v0p50).
REM
REM   1. Makes D:\code\MPPP (this MPPP folder) a git working copy of the history in
REM      _transfer\mppp_latest.bundle (the files already here are kept; nothing is downloaded).
REM   2. Deletes the files that MPPP no longer has (params\, the old test_v0pNN.py, moved scripts), after asking.
REM   3. Creates the private repository github.com/<you>/MPPP and pushes (GitHub CLI "gh" if installed; otherwise
REM      it tells you the two steps).
REM   4. Offers to delete the old mppp_v0p4*.bundle files in the MPPP folder.
REM
REM Needs Git for Windows (https://git-scm.com/download/win). Afterwards: sync_from_claude.bat after each delivery.
REM ======================================================================================================
setlocal EnableExtensions
set "GH_USER=ctate7163"
set "REPO=MPPP"
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || (echo cannot open %MPPP_HOME% & pause & exit /b 1)
set "BUNDLE=%MPPP_HOME%\_transfer\mppp_latest.bundle"

where git >nul 2>nul || (echo Git is not installed: get it from https://git-scm.com/download/win, then run this again. & pause & exit /b 1)
if not exist "%BUNDLE%" (echo %BUNDLE% not found & pause & exit /b 1)

if exist ".git" (
  echo %MPPP_HOME% is already a git working copy - step 1 skipped.
) else (
  echo [1] git history from %BUNDLE%
  git init -q -b main || (pause & exit /b 1)
  git fetch -q "%BUNDLE%" "+refs/heads/main:refs/remotes/claude/main" "+refs/tags/*:refs/tags/*" || (pause & exit /b 1)
  git reset -q --mixed claude/main || (pause & exit /b 1)
)
git config user.name >nul 2>nul || git config user.name "Christian Tate"
git config user.email >nul 2>nul || git config user.email "ctate7163@gmail.com"

echo.
echo [2] files MPPP no longer has (deleted or moved since 0.44.0):
git diff --name-only --no-renames --diff-filter=D v0.44.0 HEAD > "%TEMP%\mppp_gone.txt"
set "N=0"
for /f "usebackq delims=" %%F in ("%TEMP%\mppp_gone.txt") do if exist "%%F" (echo    %%F& set /a N+=1 >nul)
if exist "params" echo    params\  (whole folder; its files are in src\mppp\data now)
choice /M "Delete these"
if errorlevel 2 goto :restore
for /f "usebackq delims=" %%F in ("%TEMP%\mppp_gone.txt") do if exist "%%F" del /q "%%F"
if exist "params" rmdir /s /q "params"
for %%D in (src\mppp\data\m20_cmods src\mppp\data\navcam_consensus) do if exist "%%D" rmdir /s /q "%%D"

:restore
REM tracked files that are missing here (e.g. the unversioned notebooks) come from the history
git ls-files --deleted > "%TEMP%\mppp_missing.txt"
for /f "usebackq delims=" %%F in ("%TEMP%\mppp_missing.txt") do git checkout -q HEAD -- "%%F"
echo.
echo git status (files you changed yourself show as modified; the notebook copies *_v0pNN.ipynb are ignored):
git status --short
echo.

echo [3] GitHub: private repository %GH_USER%/%REPO%
git remote get-url origin >nul 2>nul && goto :push
where gh >nul 2>nul
if errorlevel 1 goto :nogh
gh auth status >nul 2>nul || gh auth login -w
gh repo create %GH_USER%/%REPO% --private --source . --remote origin --description "Mars Photogrammetry Preprocessing Pipeline" || goto :nogh
goto :push
:nogh
echo The GitHub CLI is not installed (or could not create the repository). Two steps instead:
echo   a. In the browser: https://github.com/new  - name %REPO%, Private, no README / .gitignore / license - Create.
echo   b. Then press a key here: the push asks you to sign in to GitHub in the browser once.
pause
git remote add origin https://github.com/%GH_USER%/%REPO%.git
:push
git push -u origin main --tags || (echo push failed - check the repository exists and you are signed in & pause & exit /b 1)
echo Pushed to https://github.com/%GH_USER%/%REPO% (private).

echo.
echo [4] the old bundles (the history is in git and on GitHub now):
dir /b mppp_v0p*.bundle 2>nul
if exist mppp_v0p*.bundle (
  choice /M "Delete them"
  if not errorlevel 2 del /q mppp_v0p*.bundle
)
echo Done. After each delivery from Claude run sync_from_claude.bat.
pause
