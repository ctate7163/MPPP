@echo off
REM ======================================================================================================
REM MPPP (v0p82) - take Claude's delivery (_transfer\mppp_latest.bundle), push it to GitHub, write
REM _transfer\mppp_pc.bundle for Claude, then optionally start a batch file of scripts\windows:
REM
REM   claude_sync.bat                              sync and push only
REM   claude_sync.bat run_pipeline.bat --reset     sync, push, then run_pipeline.bat --reset
REM
REM No questions asked: it only fast-forwards (stops, changing nothing, if this copy has commits the delivery lacks -
REM then send Claude _transfer\mppp_pc.bundle). Uncommitted changes to tracked files go to a stash first
REM ("git stash list"; "git stash pop" brings them back). Log: _transfer\claude_sync_log.txt
REM Replaces the one-off sync batch files Claude used to put in _transfer.
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if exist "%~dp0..\scripts\windows\mppp_env.bat" for %%I in ("%~dp0..") do set "MPPP_HOME=%%~fI"
REM (a copy in _transfer works too)
cd /d "%MPPP_HOME%" || goto :fail
set "LOG=%MPPP_HOME%\_transfer\claude_sync_log.txt"
set "BUNDLE=%MPPP_HOME%\_transfer\mppp_latest.bundle"
if not exist "%BUNDLE%" (echo %BUNDLE% not found & pause & exit /b 1)
echo ==== claude_sync %DATE% %TIME% > "%LOG%"
echo before: >> "%LOG%"
git log --oneline -1 >> "%LOG%" 2>&1
git fetch "%BUNDLE%" "+refs/heads/main:refs/remotes/claude/main" "+refs/tags/*:refs/tags/*" >> "%LOG%" 2>&1 || goto :fail
git merge-base --is-ancestor HEAD claude/main || (echo this copy has commits that the delivery lacks - nothing changed; send Claude _transfer\mppp_pc.bundle >> "%LOG%" & git bundle create "_transfer\mppp_pc.bundle" main --tags >> "%LOG%" 2>&1 & goto :fail)
echo uncommitted changes (tracked ones are stashed): >> "%LOG%"
git status --porcelain >> "%LOG%" 2>&1
git stash push -m "claude_sync %DATE% %TIME%" >> "%LOG%" 2>&1
git merge --ff-only claude/main >> "%LOG%" 2>&1 || goto :fail
echo after: >> "%LOG%"
git log --oneline -3 >> "%LOG%" 2>&1
git push origin main >> "%LOG%" 2>&1 || echo PUSH OF MAIN FAILED >> "%LOG%"
git push origin --tags >> "%LOG%" 2>&1 || echo some tags were not pushed (see above) >> "%LOG%"
git bundle create "_transfer\mppp_pc.bundle" main --tags >> "%LOG%" 2>&1
echo SYNC DONE >> "%LOG%"
type "%LOG%"
REM v0p86: the JOSS paper (paper\paper.md -> paper\paper.pdf) is rendered after every sync
if exist "%MPPP_HOME%\scripts\windows\render_paper.bat" call "%MPPP_HOME%\scripts\windows\render_paper.bat" --quiet
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
type "%LOG%"
pause
exit /b 1
