@echo off
REM ======================================================================================================
REM MPPP 0.94.4 - run_weekend.bat (Christian, 23 Oct 2026): several days unattended, in a minimised window -
REM   1. wait until nothing else runs on ROOT (a test_lightglue.bat, a run_sites batch)
REM   2. the matcher / pose-guided A/B on Chal Rocks tiny (every variant again) and Seitah Overlook tiny
REM   3. run_pipeline (from the start): the ecam blocks (Navcam + both Hazcam pairs), the Navcam joint and the
REM      Hazcam fit (promoted), the Mastcam-Z blocks, the Mastcam-Z joints (promoted), the error analysis
REM   4. mppp_pipeline\WEEKEND.md: the summary
REM The blocks are matched with SIFT (MATCH_MODE=exhaustive); other settings: run_weekend.bat --set NAME=VALUE.
REM The computer is kept awake while it runs.  If python dies, it starts again after 2 minutes and goes on where
REM it stopped (at most 20 times).  A reboot stops it: double-click this file again.  To stop it: stop_mppp.bat.
REM
REM   log      %ROOT%\mppp_pipeline\weekend_log.txt  (each step: weekend_<step>.log; the pipeline: pipeline_log.txt)
REM   status   run_weekend.bat --status      plan   run_weekend.bat --dry-run
REM Do not sync (claude_sync.bat) while it runs.
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes"
if defined MPPP_SCAPES set "ROOT=%MPPP_SCAPES%"
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if /i "%~1"=="--status" (
  python "%MPPP_HOME%\studies\calibration\run_weekend.py" --root "%ROOT%" --status
  python "%MPPP_HOME%\studies\calibration\run_pipeline.py" --root "%ROOT%" --status
  pause
  exit /b 0
)
if /i "%~1"=="--dry-run" (
  python "%MPPP_HOME%\studies\calibration\run_weekend.py" --root "%ROOT%" %*
  pause
  exit /b 0
)
start "MPPP weekend - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_weekend_loop.bat" "%ROOT%" %*
echo MPPP weekend run started in a minimised window. Log: "%ROOT%\mppp_pipeline\weekend_log.txt"
echo Status: run_weekend.bat --status
timeout /t 15
