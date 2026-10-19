@echo off
REM ======================================================================================================
REM MPPP v0p85 - the ~25-30 h run (Christian, 19 Oct 2026). Start it only after the v0p82 run has finished
REM and _transfer\claude_sync.bat has been run.
REM
REM   Navcam cameras: the 18 Oct pass-2 consensus of 23 blocks, already promoted in the repo (cmods CHANGES.md)
REM   navcam_runs:     the 12 blocks of the navcam_consensus group (sites.json), rig epochs of at most 25 sols
REM   navcam_consensus_2: the joint over those 12 blocks (NL and NR cx against temperature, early offsets
REM                    started at the pass-2 values, one round), promoted if every check passes; rig residuals
REM   waypoint_check
REM   (baseline study skipped: the v0p82 run's study of 18 Oct answers the baseline question)
REM   zcam_start, zcam_runs, zcam_calibrate, zcam_final_runs: the 6 blocks of zcam34_consensus
REM   pointclouds, report
REM
REM Same window, log and stop rules as run_pipeline.bat (stop_mppp.bat to stop; double-click this again to go on).
REM ======================================================================================================
setlocal
if /i "%~1"=="--continue" (
  call "%~dp0run_pipeline.bat" --no-baseline-study
  exit /b %ERRORLEVEL%
)
call "%~dp0run_pipeline.bat" --reset --from navcam_runs --no-baseline-study %*
exit /b %ERRORLEVEL%
