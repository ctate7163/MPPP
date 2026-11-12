@echo off
REM ======================================================================================================
REM MPPP 0.97.0 - zcam_joints_097.bat (Christian, 12 Nov 2026: "I will stop MPPP and start a new joint for z63,
REM z79 and z110").  Stop the freeze first (Ctrl-C in its window, or stop_mppp.bat).  Then this file:
REM   1. aligns the new Navcam + Mastcam-Z blocks: rose_river_falls (34, 63, 110 mm), knob_mountain_large (34, 63)
REM      overlook_mountain (site 52 drives 2166-2750: 48, 63, 79, 110) and hippo_pools (34, 48, 63, 110), with the
REM      freeze models
REM      and Rockytop again (63 mm: ZCAM08541 alone; 110 mm with ZCAM08537)
REM   2. the 63, 79 and 110 mm joints over zcam34_joint + zcam_consensus (Taylorfjellet back in: its arm-workspace
REM      station warns; Rockytop at 63 mm with ZCAM08541 only) - NOT promoted unless MPPP_PROMOTE=1 is set before calling
REM   3. the 79 mm test: one state for both (high / low ratio held at 1) and the ratio held at the 34 mm value
REM   4. validation test 3 on every block (Mastcam-Z pairs alone; reads D:\data\m2020\zcam)
REM   zcam_joints_097.bat             everything
REM   zcam_joints_097.bat joints      steps 2-4 (the blocks are aligned)
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

echo ==== 1. the new blocks (hours)
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites rose_river_falls knob_mountain_large overlook_mountain hippo_pools rockytop rockytop_skinner --jobs 2 --zcam
echo %DATE% %TIME% blocks: exit %ERRORLEVEL% >> "%LOG%"

:joints
for %%Z in (63 79 110) do (
  echo ==== 2. the %%Z mm joint
  python -m mppp --log "%CA%\zcam%%Z_joint_%TODAY%_097_log.txt" calib zcam --zoom %%Z --group %GROUP% --root "%ROOT%" --out "%CA%\zcam%%Z_joint_%TODAY%_097" --iterations 300 %PROMOTE%
  echo %DATE% %TIME% Z%%Z joint: exit !ERRORLEVEL! >> "%LOG%"
)
echo ==== 3. 79 mm: one state for both, and the ratio at the 34 mm value
python -m mppp --log "%CA%\zcam79_one_state_%TODAY%_log.txt" calib zcam --zoom 79 --group %GROUP% --root "%ROOT%" --out "%CA%\zcam79_one_state_%TODAY%" --iterations 300 --zcam-hold-ratio 1.0 --no-promote
python -m mppp --log "%CA%\zcam79_ratio_z34_%TODAY%_log.txt" calib zcam --zoom 79 --group %GROUP% --root "%ROOT%" --out "%CA%\zcam79_ratio_z34_%TODAY%" --iterations 300 --zcam-hold-ratio 1.0087 --no-promote
echo %DATE% %TIME% Z79 tests: exit %ERRORLEVEL% >> "%LOG%"
echo ==== 4. validation test 3: Mastcam-Z pairs alone
python -m mppp_studies.validation.zpairs --root "%ROOT%" --pds "%MPPP_ZPDS%" --cmods "%MPPP_CMODS%" --zooms 34 48 63 110 --per-block 8 --out "%CA%\validation_zpairs_%TODAY%"
echo %DATE% %TIME% zpairs: exit %ERRORLEVEL% >> "%LOG%"
echo.
echo done: %CA%\zcam63_joint_%TODAY%_097\report.md, zcam79_..., zcam110_..., zcam79_one_state_%TODAY%, validation_zpairs_%TODAY%
echo compare the 79 mm runs: final cost and residuals in ratio_hold_test.json, the sequence offsets in report.md
pause
