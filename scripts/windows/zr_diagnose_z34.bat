@echo off
REM ======================================================================================================
REM MPPP v0p85 - Mastcam-Z 34 mm diagnostic run (Christian, 19 Oct 2026): what happens to the right-eye (ZR) frames.
REM
REM   Rochette and South Arm only, Navcam + Mastcam-Z 34 mm, with the Navcam models promoted in the repo (the
REM   18 Oct pass-2 consensus) and the Mastcam-Z start model in use. In the 18-19 Oct run every ZR frame of these
REM   blocks lost its tie points in stage 2 (South Arm at round 3, Rochette at the pose-guided step).
REM
REM   * results in <WORK>\colmap_zrdiag (the blocks' own colmap\ folders are not touched)
REM   * MPPP_ZCAM_TRACE=2: every step of stage 2 writes one record per Mastcam-Z image (registered, observations,
REM     keypoints, frame, rig, camera, pose, pose relative to its left partner against the Mastcam-Z rig) to
REM     colmap_zrdiag\zr_trace.jsonl, and the model after every triangulation to colmap_zrdiag\sparse\_zdiag_<step>
REM   * the v0p85 guard reports and repairs a collapse (its messages are in the run log)
REM   * at the end: D:\scapes\mppp_pipeline\zr_diagnosis.md (step-by-step table, the step where ZR is lost and what
REM     changed there) - send it to Claude, with the two zr_trace.jsonl files
REM
REM About 1 to 1.5 h. Do not run it at the same time as run_v0p85.bat (start it before, or after the Navcam part).
REM Run claude_sync.bat first. This window stays open; closing it stops the run.
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes"
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\move_scapes.py" --apply || ( echo moving the block folders failed - nothing started & pause & exit /b 1 )
set "MPPP_ZCAM_TRACE=2"
set "VARIANT=zrdiag"
echo %DATE% %TIME%  Mastcam-Z diagnostic run: rochette, south_arm (variant %VARIANT%)
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites rochette south_arm --zcam --set ZCAM_ZOOMS=[34] --variant %VARIANT% --force
python "%MPPP_HOME%\scripts\zr_report.py" "%ROOT%\mars2020_sol_0178_rochette_zcam\colmap_%VARIANT%" "%ROOT%\mars2020_sol_1408_south_arm_zcam\colmap_%VARIANT%" --out "%ROOT%\mppp_pipeline\zr_diagnosis.md"
echo.
echo %DATE% %TIME%  done. Report: %ROOT%\mppp_pipeline\zr_diagnosis.md
pause
