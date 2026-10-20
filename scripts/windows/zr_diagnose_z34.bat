@echo off
REM ======================================================================================================
REM MPPP v0p87 - Mastcam-Z 34 mm diagnostic and calibration run (Christian, 19 Oct 2026)
REM
REM   Rochette, South Arm and Marble Mountain (sites.json group zcam34_diag), Navcam + Mastcam-Z 34 mm, with the
REM   Navcam models promoted in the repo (18 Oct pass 2) and the Mastcam-Z start model in use, with the ZR fix
REM   (COLMAP's bogus-camera limit lifted; k2 kept within +-10).
REM
REM   * results in each block's <WORK>\colmap (the 18 Oct first runs there lost their right eye; they are replaced)
REM   * MPPP_ZCAM_TRACE=1: every step of stage 2 writes one record per Mastcam-Z image to colmap\zr_trace.jsonl
REM   * D:\scapes\mppp_pipeline\zr_diagnosis.md: frames with tie points per eye at every step, for each block
REM   * then the Mastcam-Z 34 mm calibration from these three blocks (scripts\zcam_calibration.py calibrate);
REM     it is promoted only if every check passes (report in D:\scapes\camera_analysis\zcam34_calibration_<date>)
REM
REM About 3.5 to 4 h. Do not run it at the same time as run_v0p87.bat. Run claude_sync.bat first.
REM This window stays open; to stop it use stop_mppp.bat.
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes"
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\move_scapes.py" --apply || ( echo moving the block folders failed - nothing started & pause & exit /b 1 )
set "MPPP_ZCAM_TRACE=1"
echo %DATE% %TIME%  Mastcam-Z 34 mm diagnostic run: rochette, south_arm, marble_mountain
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --group zcam34_diag --zcam --set ZCAM_ZOOMS=[34] --force
python "%MPPP_HOME%\scripts\zr_report.py" "%ROOT%\mars2020_sol_0178_rochette_zcam\colmap" "%ROOT%\mars2020_sol_1408_south_arm_zcam\colmap" "%ROOT%\mars2020_sol_1962_marble_mountain_zcam\colmap" --out "%ROOT%\mppp_pipeline\zr_diagnosis.md"
echo.
echo %DATE% %TIME%  Mastcam-Z 34 mm calibration from the three blocks (promoted only if every check passes)
python "%MPPP_HOME%\scripts\zcam_calibration.py" calibrate --root "%ROOT%" --groups zcam34_diag --n 0 --promote
if errorlevel 1 (echo calibration NOT promoted - see %ROOT%\camera_analysis\zcam34_calibration_*\report.md) else (echo calibration PROMOTED to src\mppp\data\cmods)
echo.
echo %DATE% %TIME%  done. Report: %ROOT%\mppp_pipeline\zr_diagnosis.md
pause
