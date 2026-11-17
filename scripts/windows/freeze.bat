@echo off
REM ======================================================================================================
REM RPP 0.96.0 - freeze.bat (9 Nov 2026, Christian: "please prepare a freeze.bat that will run on 0.96").
REM The freeze run, unattended and resumable, on its own checkout of the code (RPP_FREEZE_CODE): D:\code\RPP_freeze
REM for RPP (0.99 and later: this tree has src\rpp), D:\code\MPPP_freeze for the old MPPP 0.98 line - cloned from
REM this tree, on its branch, the first time.  See rpp_studies.calibration.freeze for the steps (wait, mask, cmods,
REM zjoints, zfinal, ecam, nz, deck, knee, ztests, studies, ladder, report) and the faults it rides out.
REM
REM   freeze.bat                        start, or continue after a stop (done steps are skipped)
REM   freeze.bat --status               where it is: running / stopped by the user / done / not running; nothing started
REM   freeze.bat --mask-model auto      wait for the mask training under way and its comparison; the new model is used
REM                                     when it beats the blocks' model by 0.01 exclusion IoU (--mask-margin), else
REM                                     the blocks' model stays (default --mask-model keep: no image processed again)
REM   freeze.bat --update-code          the checkout brought up to this tree's branch (git pull --ff-only); nothing started
REM   freeze.bat --fresh --now          0.97.0: from the beginning - the status of the run before is set aside,
REM                                     every step runs again (refused while a freeze is alive)
REM   freeze.bat --now                  start once the pipeline lock is free (do not wait for the first chain)
REM   freeze.bat --redo zjoints         a step again;   freeze.bat --only deck   one step
REM   freeze.bat --zcam-waive-aspect    (default 63 79 110) the zooms whose per-block aspect gate is waived
REM   freeze.bat --fresh --now --cmods D:\rover_blocks\cmods_freeze3 --cmods-from D:\rover_blocks\cmods_freeze2 ^
REM              --candidate <folder with M2020_N_rig.json>     0.98.15: freeze-3 (the folder and candidate are kept
REM                                     in the status; a restart without them goes on with them)
REM Root: RPP_FREEZE_ROOT, else D:\rover_blocks.  Models: the run's own, else RPP_FREEZE_CMODS, else cmods_freeze2.
REM Log: <root>\mppp_pipeline\freeze_log.txt; status freeze_status.json; each step's output freeze_<step>.log.
REM Stop: Ctrl-C in this window, or stop_rpp.bat - both are logged as "stopped by the user"; start again to go on.
REM After a reboot the run starts itself again at your next logon (RunOnce "RPP freeze").
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "RPP_MAIN=%%~fI"
call "%~dp0rpp_env.bat" --legacy-only
REM 0.99.3 / 0.99.5: each line of the code keeps its own checkout, so an MPPP 0.98 checkout is never moved to RPP
REM or back (freeze.py checks the version too): a tree with src\rpp (RPP 0.99 and later - D:\code\RPP, on main
REM after rename_mppp_to_rpp.bat, or on rpp) uses D:\code\RPP_freeze, cloned on the tree's branch; a tree without
REM it (MPPP 0.98) uses D:\code\MPPP_freeze
set "FREEZE_BRANCH="
for /f %%b in ('git -C "%RPP_MAIN%" rev-parse --abbrev-ref HEAD 2^>nul') do set "FREEZE_BRANCH=%%b"
if not defined FREEZE_BRANCH set "FREEZE_BRANCH=main"
if /i "!FREEZE_BRANCH!"=="HEAD" (
  echo freeze: %RPP_MAIN% is not on a branch ^(detached HEAD^) - check out main first
  if not defined RPP_NOPAUSE pause
  exit /b 1
)
set "FREEZE_DEFAULT_CODE=D:\code\MPPP_freeze"
if exist "%RPP_MAIN%\src\rpp\__init__.py" set "FREEZE_DEFAULT_CODE=D:\code\RPP_freeze"
if not defined RPP_FREEZE_ROOT set "RPP_FREEZE_ROOT=D:\rover_blocks"
if not defined RPP_FREEZE_CODE set "RPP_FREEZE_CODE=!FREEZE_DEFAULT_CODE!"
title RPP freeze (!FREEZE_BRANCH!) - %RPP_FREEZE_ROOT%
call "%RPP_MAIN%\scripts\windows\rpp_env.bat"
if errorlevel 1 (
  echo freeze: no Python with rpp found - not started
  if not defined RPP_NOPAUSE pause
  exit /b 1
)
if /i "%RPP_FREEZE_CODE%"=="%RPP_MAIN%" goto :have_code
if not exist "%RPP_FREEZE_CODE%\.git" (
  echo freeze: cloning %RPP_MAIN% ^(branch !FREEZE_BRANCH!^) into %RPP_FREEZE_CODE% ...
  git clone -q -b !FREEZE_BRANCH! "%RPP_MAIN%" "%RPP_FREEZE_CODE%" || ( echo clone failed & if not defined RPP_NOPAUSE pause & exit /b 1 )
)
:have_code
set "RPP_HOME=%RPP_FREEZE_CODE%"
set "PYTHONPATH=%RPP_FREEZE_CODE%\src;%PYTHONPATH%"
set "RPP_HOME_MAIN=%RPP_MAIN%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "RPP_SCAPES=%RPP_FREEZE_ROOT%"
if exist "%RPP_MAIN%\scripts\windows\m20_dem_env.bat" call "%RPP_MAIN%\scripts\windows\m20_dem_env.bat"
set "TPY="
for %%P in ("%RPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, rpp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY set "TPY=python"
for %%A in (%*) do (
  if /i "%%~A"=="--status" goto :quick
  if /i "%%~A"=="--update-code" goto :quick
  if /i "%%~A"=="-h" goto :quick
  if /i "%%~A"=="--help" goto :quick
)
echo %DATE% %TIME%  freeze: root %RPP_FREEZE_ROOT%, code %RPP_FREEZE_CODE%, mask comparison with %TPY%
python -m rpp_studies.calibration.freeze --root "%RPP_FREEZE_ROOT%" --code "%RPP_FREEZE_CODE%" --from-tree "%RPP_MAIN%" --mask-home "%RPP_MAIN%" --ab-python "%TPY%" %*
set "_RC=%ERRORLEVEL%"
echo %DATE% %TIME%  freeze finished (exit %_RC%)
if not defined RPP_NOPAUSE pause
exit /b %_RC%
:quick
python -m rpp_studies.calibration.freeze --root "%RPP_FREEZE_ROOT%" --code "%RPP_FREEZE_CODE%" --from-tree "%RPP_MAIN%" %*
set "_RC=%ERRORLEVEL%"
if not defined RPP_NOPAUSE pause
exit /b %_RC%
