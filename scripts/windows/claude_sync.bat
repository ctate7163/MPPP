@echo off
REM ======================================================================================================
REM MPPP 0.95.0 - take Claude's delivery (_transfer\mppp_latest.bundle), fast-forward main, push it to GitHub,
REM write _transfer\mppp_pc.bundle for Claude, then optionally start a batch file of scripts\windows:
REM
REM   claude_sync.bat                              sync and push only
REM   claude_sync.bat run_pipeline.bat --reset     sync, push, then run_pipeline.bat --reset
REM
REM No questions asked: it only fast-forwards main.  It stops, changing nothing, when
REM   - the delivery is already here (the bundle's main is HEAD),
REM   - the bundle's main is not ahead of this copy (a delivery left on a side branch, or this copy has commits
REM     the delivery lacks - then it writes _transfer\mppp_pc.bundle to send to Claude),
REM   - main did not move after the merge.
REM Only refs/heads/main is fetched (no tags: 20-27 Oct, the tag refs broke the pushes).  Uncommitted changes to
REM tracked files go to a stash and come back after the merge (0.95.0: the camera models in use live in MPPP_CMODS,
REM outside git, so there should be none).  Log: _transfer\claude_sync_log.txt (the last runs, newest at the end).
REM Run it from scripts\windows - a copy elsewhere (the old one in _transfer) refuses to run.
REM 0.95.0 (29 Oct): it runs from a copy in %TEMP%.  cmd.exe reads a batch file from disk line by line, and the merge
REM replaces this file: on 29 Oct the running script lost its place after the merge, started over, found "already
REM here" and stopped - the stash was never popped, main never pushed.  The copy is never touched by the merge.
REM ======================================================================================================
if /i not "%~1"=="--from-copy" (
  copy /y "%~f0" "%TEMP%\mppp_claude_sync_run.bat" >nul || ( echo cannot copy %~f0 to %TEMP% & pause & exit /b 1 )
  "%TEMP%\mppp_claude_sync_run.bat" --from-copy "%~dp0" %*
)
setlocal EnableExtensions EnableDelayedExpansion
set "HERE=%~2"
shift
shift
if /i not "%HERE:~-16%"=="scripts\windows\" (
  echo claude_sync.bat must run from scripts\windows ^(this copy: %HERE%^) - delete the old copy in _transfer
  pause
  exit /b 1
)
for %%I in ("%HERE%..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || goto :fail
set "LOG=%MPPP_HOME%\_transfer\claude_sync_log.txt"
set "BUNDLE=%MPPP_HOME%\_transfer\mppp_latest.bundle"
if not exist "%BUNDLE%" (echo %BUNDLE% not found & pause & exit /b 1)
REM the log keeps the last runs: beyond ~3000 lines the old one becomes claude_sync_log_prev.txt
if exist "%LOG%" for /f %%N in ('type "%LOG%" ^| find /c /v ""') do if %%N gtr 3000 move /y "%LOG%" "%MPPP_HOME%\_transfer\claude_sync_log_prev.txt" >nul
echo. >> "%LOG%"
echo ==== claude_sync %DATE% %TIME% >> "%LOG%"
echo before: >> "%LOG%"
git log --oneline -1 >> "%LOG%" 2>&1
for /f "delims=" %%H in ('git rev-parse HEAD') do set "BEFORE=%%H"
echo bundle heads: >> "%LOG%"
git bundle list-heads "%BUNDLE%" >> "%LOG%" 2>&1
git fetch "%BUNDLE%" "+refs/heads/main:refs/remotes/claude/main" >> "%LOG%" 2>&1 || goto :fail
echo newest commit of the delivery: >> "%LOG%"
git log --oneline -1 claude/main >> "%LOG%" 2>&1
for /f "delims=" %%H in ('git rev-parse claude/main') do set "DELIVERY=%%H"
if "%DELIVERY%"=="%BEFORE%" (
  echo the delivery is already here ^(main at %BEFORE:~0,7%^) - nothing to sync >> "%LOG%"
  goto :done
)
git merge-base --is-ancestor claude/main HEAD && (
  echo NOTHING TO SYNC: the bundle's main ^(%DELIVERY:~0,7%^) is behind this copy - the delivery is not on main, or this copy has commits the delivery lacks; send Claude _transfer\mppp_pc.bundle >> "%LOG%"
  git bundle create "_transfer\mppp_pc.bundle" main >> "%LOG%" 2>&1
  goto :fail
)
git merge-base --is-ancestor HEAD claude/main || (
  echo DIVERGED: this copy has commits that the delivery lacks - nothing changed; send Claude _transfer\mppp_pc.bundle >> "%LOG%"
  git bundle create "_transfer\mppp_pc.bundle" main >> "%LOG%" 2>&1
  goto :fail
)
echo uncommitted changes (tracked ones are stashed): >> "%LOG%"
git status --porcelain >> "%LOG%" 2>&1
set "STASHMSG=claude_sync %DATE% %TIME%"
git stash push -m "%STASHMSG%" >> "%LOG%" 2>&1
git merge --ff-only claude/main >> "%LOG%" 2>&1 || goto :fail
REM the stash made by this run (and only that one) comes back right after the merge; on a conflict it stays
git stash list | findstr /c:"%STASHMSG%" >nul 2>&1 && (git stash pop >> "%LOG%" 2>&1 || echo YOUR EDITS ARE STILL IN THE STASH - git stash list, git stash pop >> "%LOG%")
echo after: >> "%LOG%"
git log --oneline -3 >> "%LOG%" 2>&1
for /f "delims=" %%H in ('git rev-parse HEAD') do set "AFTER=%%H"
if "%AFTER%"=="%BEFORE%" (
  echo MAIN DID NOT MOVE ^(%BEFORE:~0,7%^) - the merge took nothing >> "%LOG%"
  goto :fail
)
git push origin main >> "%LOG%" 2>&1 || echo PUSH OF MAIN FAILED >> "%LOG%"
REM Claude's other branches, if the bundle carries any (refs/heads/* only - never the tags)
git fetch "%BUNDLE%" "+refs/heads/*:refs/remotes/claude/*" >> "%LOG%" 2>&1
for /f "tokens=1,2" %%A in ('git bundle list-heads "%BUNDLE%"') do (
  set "REF=%%B"
  if /i "!REF:~0,11!"=="refs/heads/" if /i not "!REF!"=="refs/heads/main" (
    git push origin "refs/remotes/claude/!REF:~11!:!REF!" >> "%LOG%" 2>&1 || echo PUSH OF !REF:~11! FAILED >> "%LOG%"
  )
)
git bundle create "_transfer\mppp_pc.bundle" main >> "%LOG%" 2>&1
echo SYNC DONE: main %BEFORE:~0,7% -^> %AFTER:~0,7% >> "%LOG%"
:done
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 60" 2>nul || type "%LOG%"
if "%~1"=="" (timeout /t 20 & exit /b 0)
set "NEXT=%~1"
shift
set "ARGS="
:args
if "%~1"=="" goto :next
set "ARGS=%ARGS% %1"
shift
goto :args
:next
echo starting %NEXT%%ARGS%
call "%MPPP_HOME%\scripts\windows\%NEXT%"%ARGS%
exit /b %ERRORLEVEL%
:fail
echo SYNC FAILED >> "%LOG%"
powershell -NoProfile -Command "Get-Content -Path '%LOG%' -Tail 60" 2>nul || type "%LOG%"
pause
exit /b 1
