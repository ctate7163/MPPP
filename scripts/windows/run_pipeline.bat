@echo off
REM ======================================================================================================
REM MPPP v0p74 - the whole calibration and alignment run in one go, unattended (several days):
REM   wait for a running batch, check/promote the camera models, Navcam consensus (v0p73 form) + rig residuals,
REM   Mastcam-Z start model and low-backlash list from the current blocks, all Navcam + Mastcam-Z 34 mm blocks,
REM   all Navcam blocks, the Mastcam-Z 34 mm calibration from the 3 best blocks, the N+Z blocks again with it,
REM   a point cloud per block, a report.  Runs in a minimised window.  To stop it: stop_mppp.bat (it stops the
REM   pipeline and the alignment it is running; closing the window alone leaves the alignment running).
REM   Double-click again after a stop (or a reboot): it continues after the last finished stage.
REM
REM   log      %ROOT%\mppp_pipeline\pipeline_log.txt  (each stage: mppp_pipeline\<stage>.log)
REM   status   run_pipeline.bat --status          report   %ROOT%\mppp_pipeline\REPORT.md
REM Keep the computer awake (no sleep, no automatic restart for updates) until it has finished.
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\run_pipeline.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if /i "%~1"=="--status" (
  python "%MPPP_HOME%\scripts\run_pipeline.py" --root "%ROOT%" --status
  pause
  exit /b 0
)
if /i "%~1"=="--dry-run" (
  python "%MPPP_HOME%\scripts\run_pipeline.py" --root "%ROOT%" %*
  pause
  exit /b 0
)
start "MPPP pipeline - %ROOT%" /min cmd /c python "%MPPP_HOME%\scripts\run_pipeline.py" --root "%ROOT%" %* ^|^| pause
echo MPPP pipeline started in a minimised window. Log: "%ROOT%\mppp_pipeline\pipeline_log.txt"
echo Status: run_pipeline.bat --status
timeout /t 15
