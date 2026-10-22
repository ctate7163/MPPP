@echo off
REM ======================================================================================================
REM MPPP 0.91 - zcam_tied_joint.bat (Christian, 20 Oct 2026): the Mastcam-Z joint - ONE bundle adjustment over the
REM finished Navcam + Mastcam-Z 34 mm blocks of zcam34_consensus (run_z34.bat makes them), with the Navcam held at
REM the models in use (the latest promoted tied joint: cameras on their temperature law, the rig on its sol trend)
REM and the Mastcam-Z focus model fitted (mppp.calib.ztied):
REM   fx, fy = a[state] + b (focus - 600) per eye (one slope for fx and fy), k1, k2, p1, p2 per eye, ZL's principal
REM   point and ZR's cx pinned, ZR's cy with its focus slope, one Mastcam-Z rig (rotation refined, CAHV baseline);
REM   then a diagnostic pass: each block's focus lines and its own rig about the joint (the sol / temperature trends).
REM
REM   zcam_tied_joint.bat                       the joint, the candidate, the check, the figures, and the promotion
REM                                             of the focus model when every check passes (20 Oct: by default)
REM   zcam_tied_joint.bat --no-promote          the same without the promotion
REM   zcam_tied_joint.bat --zpoints 12000       more Mastcam-Z points per block (default 8000; every Navcam-Mastcam-Z
REM                                             point with a third look is kept anyway)
REM
REM A block of the group that is not there or not finished (no colmap\run_done.json) is left out and named in the
REM log ("calib zcam: left out: ..."); the joint needs at least two finished blocks.
REM Writes %MPPP_SCAPES%\camera_analysis\zcam34_joint_<date>[_N]\ (report.md, check.json, the candidate
REM M2020_ZCAM034_focus_model.json, zcam_joint.json, block_lines.json, zcam_rig_trends.png) and the figures into
REM "Claude outputs\figures_<date>". Expect well under an hour for 12 blocks.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_SCAPES%\camera_analysis\zcam34_joint_%TODAY%"
set "N=1"
:next
if exist "%OUT%\zcam_joint.json" ( set /a N+=1 & set "OUT=%MPPP_SCAPES%\camera_analysis\zcam34_joint_%TODAY%_!N!" & goto :next )
echo Mastcam-Z joint: the finished blocks of zcam34_joint ^(health not FAIL^) under %MPPP_SCAPES% -^> %OUT%
python -m mppp cmods list
python -m mppp --log "%OUT%_log.txt" calib zcam --root "%MPPP_SCAPES%" --out "%OUT%" --promote-if-ok %*
if errorlevel 1 ( echo Mastcam-Z joint FAILED - see %OUT%_log.txt & pause & exit /b 1 )
set "MPPP_NOPAUSE=1"
call "%MPPP_HOME%\scripts\windows\make_figures.bat" - "%OUT%"
REM 21 Oct: last, the combined Navcam + Mastcam-Z joint (both laws free) against the Navcam-only law - a check, never
REM promoted (report in nz34_joint_<date>; set MPPP_NZ=none to skip it)
if /i not "%MPPP_NZ%"=="none" (
  echo Navcam + Mastcam-Z combined joint ^(a check, nothing promoted^)
  python -m mppp --log "%MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%_log.txt" calib nz --root "%MPPP_SCAPES%" --out "%MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%"
  if errorlevel 1 echo the combined joint FAILED - see %MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%_log.txt
)
echo done: %OUT%
pause
exit /b 0
