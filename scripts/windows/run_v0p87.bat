@echo off
REM ======================================================================================================
REM MPPP v0p87 (was run_v0p87.bat) - the full run: 16 Navcam blocks, 7 Mastcam-Z 34 mm blocks (Christian, 19 Oct 2026).
REM (0.87 run script)
REM and the code has been synced (git fetch/merge of the bundle). The current waypoint table is downloaded first.
REM
REM   Navcam cameras: the 18 Oct pass-2 consensus of 23 blocks, already promoted in the repo (cmods CHANGES.md)
REM   navcam_runs:     the 16 blocks of the navcam_consensus group (sites.json; Three Forks Large left out in v0p86), rig epochs of at most 25 sols
REM   navcam_consensus_2: the joint over those 16 blocks (NL and NR cx against temperature, early offsets
REM                    started at the pass-2 values, one round), promoted if every check passes; rig residuals
REM   waypoint_check
REM   (baseline study skipped: the v0p82 run's study of 18 Oct answers the baseline question)
REM   zcam_start, zcam_runs, zcam_calibrate, zcam_final_runs: the 7 blocks of zcam34_consensus
REM                    (with Three Forks North since 19 Oct pm)
REM   (sites fixed 19 Oct: Chal Rocks Large from sol 72, Enchanted Lake with sol 421, Bell Island Large from 1431)
REM   pointclouds, report
REM
REM v0p87: Mastcam-Z 34 mm start model = the South Arm calibration of 19 Oct (cmods CHANGES.md).
REM v0p86: SIFT + LightGlue matching where this PC's pycolmap supports it (else SIFT, with a note in the logs).
REM v0p85: the block folders live in D:\scapes (not D:\scapes\colmap) and have no "_colmap" in their names. Before
REM the pipeline starts, scripts\move_scapes.py --apply moves what is still in D:\scapes\colmap (renames, nothing is
REM copied or overwritten; log D:\scapes\mppp_pipeline\move_scapes_v0p85.json). Same as move_scapes.bat.
REM
REM   run_v0p87.bat --navcam-only   the Navcam part only (stops before zcam_start); then  run_v0p87.bat --continue
REM
REM Same window, log and stop rules as run_pipeline.bat (stop_mppp.bat to stop; double-click this again to go on).
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
python -m mppp waypoints --refresh
python "%MPPP_HOME%\studies\calibration\move_scapes.py" --apply || ( echo moving the block folders failed - nothing started & pause & exit /b 1 )
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
