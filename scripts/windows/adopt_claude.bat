@echo off
REM ======================================================================================================
REM MPPP - make this copy exactly Claude's latest delivery, then push it to GitHub (v0p61).
REM
REM For when sync_from_claude.bat or github_push.bat cannot merge (e.g. "Your local changes ... would be overwritten").
REM   1. every uncommitted change in this folder (also untracked files) is put aside in a stash - nothing is lost:
REM      "git stash list" shows it, "git stash show -p stash@{0}" what it holds, "git stash pop" brings it back;
REM   2. main becomes the history in _transfer\mppp_latest.bundle and the files are reset to it;
REM   3. github_push.bat: choose K if GitHub has older commits (they are kept on GitHub as a branch).
REM Edit src\mppp\data\sites.json again afterwards if you changed it since the last delivery (it is in the stash).
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || (pause & exit /b 1)
set "BUNDLE=%MPPP_HOME%\_transfer\mppp_latest.bundle"
if not exist ".git" (echo not a git working copy yet: run setup_github.bat first & pause & exit /b 1)
if not exist "%BUNDLE%" (echo %BUNDLE% not found & pause & exit /b 1)
git fetch -q "%BUNDLE%" "+refs/heads/main:refs/remotes/claude/main" "+refs/tags/*:refs/tags/*" || (pause & exit /b 1)
REM v0p72: a camera-model promotion made here (notebook 04 section 11) that Claude's delivery lacks would be lost
set "NPROMO=0"
for /f %%C in ('git rev-list --count claude/main..HEAD -- src/mppp/data/cmods 2^>nul') do set "NPROMO=%%C"
if not "%NPROMO%"=="0" git diff --quiet HEAD claude/main -- src/mppp/data/cmods && set "NPROMO=0"
if not "%NPROMO%"=="0" (
  echo This copy has %NPROMO% commit^(s^) of promoted camera models ^(src\mppp\data\cmods^) that Claude's delivery lacks.
  echo Tell Claude first ^(see _transfer\promoted.json^); Claude takes them into the next delivery. Nothing changed.
  pause & exit /b 1
)
echo Claude's delivery:
git log --oneline -1 claude/main
echo this copy now:
git log --oneline -1 HEAD
git status --short
choice /C YN /M "Put this copy's changes aside (stash) and make main Claude's delivery"
if errorlevel 2 (echo nothing changed & pause & exit /b 1)
git stash push -u -q -m "adopt_claude %DATE% %TIME%: this copy before taking Claude's delivery" 2>nul
git stash list | find "adopt_claude" >nul && echo your changes are in the stash: git stash list
git checkout -q -B main claude/main || (pause & exit /b 1)
git reset -q --hard claude/main || (pause & exit /b 1)
git log --oneline -1
echo.
call "%~dp0github_push.bat"
