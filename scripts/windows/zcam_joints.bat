@echo off
REM ======================================================================================================
REM MPPP 0.93 - zcam_joints.bat (Christian, 22 Oct 2026): after run_sites_zcam.bat (it calls this file when its blocks
REM are done), the Mastcam-Z joint of EVERY zoom over the good blocks (finished, health not FAIL, the zoom in the
REM site's "zcam" list), each zoom's new focus model checked against the one in use and promoted when every check
REM passes:
REM   34 mm            the blocks of zcam34_joint and zcam_consensus (a calibrated model exists: >= 3 blocks to promote)
REM   48, 63, 79, 110  the blocks of zcam_consensus (no model yet: the joint starts from the blocks' refined focus
REM                    bins; a first calibration needs >= 2 blocks to be promoted)
REM Held at fixed values in every joint (mppp.calib.forms.ZCAM_FIXED_TERMS): 34 mm k2 (ZL +0.5, ZR +0.6); 63, 79,
REM 110 mm k2 = p1 = p2 = 0.  p1, p2 vary at 34 and 48 mm.
REM Last, the Navcam + Mastcam-Z 34 mm combined joint with every finished navcam_consensus block added (a check of
REM the Navcam law against the Mastcam-Z frames, never promoted; about an hour; set MPPP_NZ=none to skip it).
REM
REM   zcam_joints.bat                 the joints and the promotions
REM   zcam_joints.bat --no-promote    the joints only (arguments go to every "mppp calib zcam")
REM   set MPPP_ZOOMS=34 63            only these zooms
REM   set MPPP_HAZ=none               no Hazcam fit (0.93.6: it runs first, over the zcam_consensus blocks)
REM
REM Writes %MPPP_SCAPES%\camera_analysis\zcam<zoom>_joint_<date>[_N]\ (report.md, check.json, the candidate model,
REM zcam_joint.json, block_lines.json, zcam_rig_trends.png) and nz34_joint_<date>\.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_ZOOMS set "MPPP_ZOOMS=34 48 63 79 110"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
python -m mppp cmods list
REM 0.93.6 (Christian, 23 Oct): first the Hazcam law from the same blocks (their Hazcam cameras and rigs), promoted
REM when its checks pass (set MPPP_HAZ=none to skip it)
if /i not "%MPPP_HAZ%"=="none" (
  set "HOUT=%MPPP_SCAPES%\camera_analysis\hazcam_fit_zcam_%TODAY%"
  set "HN=1"
  call :freehaz
  echo.
  echo %DATE% %TIME%  Hazcam fit over the zcam_consensus blocks -^> !HOUT!
  python -m mppp --log "!HOUT!_log.txt" calib hazcam --zcam-blocks --group zcam_consensus --root "%MPPP_SCAPES%" --out "!HOUT!" --promote-if-ok
  if errorlevel 1 echo Hazcam fit: not run or FAILED - see !HOUT!_log.txt
)
for %%Z in (%MPPP_ZOOMS%) do (
  set "GROUP=zcam_consensus"
  if "%%Z"=="34" set "GROUP=zcam34_joint,zcam_consensus"
  set "OUT=%MPPP_SCAPES%\camera_analysis\zcam%%Z_joint_%TODAY%"
  set "N=1"
  call :freeout %%Z
  echo.
  echo %DATE% %TIME%  Mastcam-Z %%Z mm joint over !GROUP! -^> !OUT!
  python -m mppp --log "!OUT!_log.txt" calib zcam --zoom %%Z --group "!GROUP!" --root "%MPPP_SCAPES%" --out "!OUT!" --promote-if-ok %*
  if errorlevel 1 echo Mastcam-Z %%Z mm joint: not run or FAILED - see !OUT!_log.txt
)
set "MPPP_NOPAUSE=1"
if exist "%MPPP_SCAPES%\camera_analysis\zcam34_joint_%TODAY%\zcam_joint.json" call "%MPPP_HOME%\scripts\windows\make_figures.bat" - "%MPPP_SCAPES%\camera_analysis\zcam34_joint_%TODAY%"
if /i not "%MPPP_NZ%"=="none" (
  echo.
  echo %DATE% %TIME%  Navcam + Mastcam-Z 34 mm combined joint with the navcam_consensus blocks ^(a check, nothing promoted^)
  python -m mppp --log "%MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%_log.txt" calib nz --group "zcam34_joint,zcam_consensus" --navcam-group navcam_consensus --root "%MPPP_SCAPES%" --out "%MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%"
  if errorlevel 1 echo the combined joint FAILED - see %MPPP_SCAPES%\camera_analysis\nz34_joint_%TODAY%_log.txt
)
echo %DATE% %TIME%  done
if not defined MPPP_NOPAUSE_END pause
exit /b 0

:freehaz
if exist "!HOUT!\hazcam_fit.json" (
  set /a HN+=1
  set "HOUT=%MPPP_SCAPES%\camera_analysis\hazcam_fit_zcam_%TODAY%_!HN!"
  goto :freehaz
)
exit /b 0

:freeout
if exist "!OUT!\zcam_joint.json" (
  set /a N+=1
  set "OUT=%MPPP_SCAPES%\camera_analysis\zcam%1_joint_%TODAY%_!N!"
  goto :freeout
)
exit /b 0
