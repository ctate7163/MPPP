@echo off
REM ======================================================================================================
REM MPPP 0.97.0 - zcam_joints_097.bat (Christian, 12 Nov 2026: "I will stop MPPP and start a new joint for z63,
REM z79 and z110").  Stop the freeze first (Ctrl-C in its window, or stop_mppp.bat).  Then this file:
REM   1. aligns the Navcam + Mastcam-Z blocks that are new or changed, with the freeze models (a block already
REM      aligned with the same definition, rules and models is skipped):
REM        wildcat_ridge, skinner_ridge   Rockytop split at sol 500 (they replace rockytop in the joints)
REM        chal_rocks_large, seitah_overlook   26 and 100 mm added; Seitah's two right-eye 79 mm frames back
REM        rose_river_falls (34, 63, 110), knob_mountain_large (34, 63), hippo_pools (34, 48, 63, 110),
REM        overlook_mountain (site 52 drives 2166-2750: 48, 63, 79, 110)
REM   2. the 63, 79 and 110 mm joints over zcam34_joint + zcam_consensus - NOT promoted unless MPPP_PROMOTE=1 is
REM      set before calling (Taylorfjellet back in; Wildcat Ridge not in the 63 mm joint: zcam_joint_exclude)
REM   3. the ratio tests, never promoted: 79 mm with one state for both (1.0) and at the Seitah zoom ladder's
REM      gap (1.006); 63 mm at the ladder's gap (1.0045)
REM   4. validation test 3 on every block (Mastcam-Z pairs alone; reads D:\data\m2020\zcam)
REM   zcam_joints_097.bat             everything
REM   zcam_joints_097.bat joints      steps 2-4 (the blocks are aligned)
REM   zcam_joints_097.bat seitah010   ONLY Seitah Overlook again with a minimum parallax of 0.10 deg (0.25 is the
REM                                   rule): results in ...seitah_overlook_zcam\colmap_tri010, the standard block untouched
REM Models: MPPP_FREEZE_CMODS, else D:\rover_blocks\cmods_freeze2.  Root: MPPP_SCAPES, else D:\rover_blocks.
REM Log: %ROOT%\mppp_pipeline\zcam_joints_097_log.txt
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\rover_blocks"
set "ROOT=%MPPP_SCAPES%"
if defined MPPP_FREEZE_CMODS ( set "MPPP_CMODS=%MPPP_FREEZE_CMODS%" ) else ( set "MPPP_CMODS=%ROOT%\cmods_freeze2" )
if not defined MPPP_ZPDS set "MPPP_ZPDS=D:\data\m2020\zcam"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "CA=%ROOT%\camera_analysis"
set "LOG=%ROOT%\mppp_pipeline\zcam_joints_097_log.txt"
set "GROUP=zcam34_joint,zcam_consensus"
set "PROMOTE=--no-promote"
if "%MPPP_PROMOTE%"=="1" set "PROMOTE=--promote-if-ok --promote-waive "per-block aspect vs the Navcam""
echo %DATE% %TIME% zcam_joints_097: models %MPPP_CMODS%, root %ROOT%, %PROMOTE% >> "%LOG%"
if /i "%~1"=="joints" goto joints
if /i "%~1"=="seitah010" (
  python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites seitah_overlook --jobs 1 --zcam --variant tri010 --set min_tri_angle_deg=0.10
  echo %DATE% %TIME% seitah_overlook tri010: exit !ERRORLEVEL! >> "%LOG%"
  pause
  exit /b
)

echo ==== 1. the new blocks (hours)
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites wildcat_ridge skinner_ridge chal_rocks_large seitah_overlook rose_river_falls knob_mountain_large overlook_mountain hippo_pools --jobs 2 --zcam
echo %DATE% %TIME% blocks: exit %ERRORLEVEL% >> "%LOG%"

:joints
for %%Z in (63 79 110) do (
  echo ==== 2. the %%Z mm joint
  python -m mppp --log "%CA%\zcam%%Z_joint_%TODAY%_097_log.txt" calib zcam --zoom %%Z --group %GROUP% --root "%ROOT%" --out "%CA%\zcam%%Z_joint_%TODAY%_097" --iterations 300 %PROMOTE%
  echo %DATE% %TIME% Z%%Z joint: exit !ERRORLEVEL! >> "%LOG%"
)
echo ==== 3. the ratio tests: 79 mm one state and the ladder's gap, 63 mm the ladder's gap
python -m mppp --log "%CA%\zcam79_one_state_%TODAY%_log.txt" calib zcam --zoom 79 --group %GROUP% --root "%ROOT%" --out "%CA%\zcam79_one_state_%TODAY%" --iterations 300 --zcam-hold-ratio 1.0 --no-promote
python -m mppp --log "%CA%\zcam79_ratio_ladder_%TODAY%_log.txt" calib zcam --zoom 79 --group %GROUP% --root "%ROOT%" --out "%CA%\zcam79_ratio_ladder_%TODAY%" --iterations 300 --zcam-hold-ratio 1.006 --no-promote
python -m mppp --log "%CA%\zcam63_ratio_ladder_%TODAY%_log.txt" calib zcam --zoom 63 --group %GROUP% --root "%ROOT%" --out "%CA%\zcam63_ratio_ladder_%TODAY%" --iterations 300 --zcam-hold-ratio 1.0045 --no-promote
echo %DATE% %TIME% ratio tests: exit %ERRORLEVEL% >> "%LOG%"
echo ==== 4. validation test 3: Mastcam-Z pairs alone
python -m mppp_studies.validation.zpairs --root "%ROOT%" --pds "%MPPP_ZPDS%" --cmods "%MPPP_CMODS%" --zooms 34 48 63 110 --per-block 8 --out "%CA%\validation_zpairs_%TODAY%"
echo %DATE% %TIME% zpairs: exit %ERRORLEVEL% >> "%LOG%"
set "VIC=%MPPP_HOME%\Claude outputs\deck_joint_2026-11-08\vic.jsonl"
if exist "%VIC%" python -m mppp_studies.validation.deck_focal --frames "%VIC%" --cmods "%MPPP_CMODS%" --out "%CA%\validation_deck_focal_%TODAY%"
echo.
echo done: %CA%\zcam63_joint_%TODAY%_097\report.md, zcam79_..., zcam110_..., zcam79_one_state_%TODAY%, validation_zpairs_%TODAY%
echo compare the 79 mm runs: final cost and residuals in ratio_hold_test.json, the sequence offsets in report.md
pause
