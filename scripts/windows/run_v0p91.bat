@echo off
REM ======================================================================================================
REM MPPP 0.91 - the full run (Christian, 20 Oct 2026): the 23 Navcam blocks of navcam_consensus, then the joint as ONE
REM bundle adjustment of every term (mppp.calib.tied), then the 11 Mastcam-Z 34 mm blocks with p1, p2 refined (zcam-v2).
REM Start it after the code has been synced (claude_sync.bat). The current waypoint table is downloaded first.
REM
REM   navcam_runs:        every block aligns again (the run key changed: rig roll held with the yaw, small rig
REM                       epochs joined, the Settings of 0.91)
REM   navcam_consensus_2: the tied joint over the finished blocks - offsets at -20 degC + slopes, per-eye early
REM                       offsets, rig at sol 0 with linear yaw/roll rates and the pitch hinge, one covariance;
REM                       the diagnostic pass (per block / bin / epoch about the solution) is the promotion gate
REM   waypoint_check
REM   zcam_start, zcam_runs, zcam_calibrate, zcam_final_runs: the 11 blocks of zcam34_consensus, zcam-v2
REM   pointclouds, report
REM
REM   run_v0p91.bat --navcam-only   the Navcam part only (stops before zcam_start); then  run_v0p91.bat --continue
REM
REM Same window, log and stop rules as run_pipeline.bat (stop_mppp.bat to stop; double-click this again to go on).
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
python -m mppp doctor
python -m mppp waypoints --refresh
if /i "%~1"=="--continue" (
  call "%~dp0run_pipeline.bat" --no-baseline-study
  exit /b %ERRORLEVEL%
)
if /i "%~1"=="--navcam-only" (
  call "%~dp0run_pipeline.bat" --reset --from navcam_runs --no-baseline-study --stop-before zcam_start
  exit /b %ERRORLEVEL%
)
call "%~dp0run_pipeline.bat" --reset --from navcam_runs --no-baseline-study %*
exit /b %ERRORLEVEL%
