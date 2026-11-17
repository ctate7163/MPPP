@echo off
REM ======================================================================================================
REM RPP v0p74 - the whole calibration and alignment run in one go, unattended (several days):
REM   wait for a running batch, check/promote the camera models, Navcam consensus (v0p73 form) + rig residuals,
REM   Mastcam-Z start model and low-backlash list from the current blocks, all Navcam + Mastcam-Z 34 mm blocks,
REM   all Navcam blocks, the Mastcam-Z 34 mm calibration from the 3 best blocks, the N+Z blocks again with it,
REM   a point cloud per block, a report.  Runs in a minimized window.  To stop it: stop_rpp.bat (it stops the
REM   pipeline and the alignment it is running; closing the window alone leaves the alignment running).
REM   Double-click again after a stop (or a reboot): it continues after the last finished stage.
REM
REM   log      %ROOT%\mppp_pipeline\pipeline_log.txt  (each stage: mppp_pipeline\<stage>.log)
REM   status   run_pipeline.bat --status          report   %ROOT%\mppp_pipeline\REPORT.md
REM Keep the computer awake (no sleep, no automatic restart for updates) until it has finished.
REM ======================================================================================================
setlocal
REM 2 Nov 2026 (0.95.3 freeze): the blocks live in D:\rover_blocks (was D:\scapes)
set "ROOT=D:\rover_blocks"
if not defined RPP_HOME for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
if not defined RPP_HOME set "RPP_HOME=D:\code\RPP"
set "RPP_WIN=%RPP_HOME%\scripts\windows"
if not exist "%RPP_WIN%\rpp_env.bat" (
  echo RPP not found in "%RPP_HOME%": set RPP_HOME to the RPP folder, e.g.  set RPP_HOME=D:\code\RPP
  pause
  exit /b 1
)
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
if /i "%~1"=="--status" (
  python -m rpp_studies.calibration.run_pipeline --root "%ROOT%" --status
  pause
  exit /b 0
)
if /i "%~1"=="--dry-run" (
  python -m rpp_studies.calibration.run_pipeline --root "%ROOT%" %*
  pause
  exit /b 0
)
start "RPP pipeline - %ROOT%" /min cmd /c python -m rpp_studies.calibration.run_pipeline --root "%ROOT%" %* ^|^| pause
echo RPP pipeline started in a minimized window. Log: "%ROOT%\mppp_pipeline\pipeline_log.txt"
echo Status: run_pipeline.bat --status
timeout /t 15
