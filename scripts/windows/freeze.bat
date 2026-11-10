@echo off
REM ======================================================================================================
REM MPPP 0.96.0 - freeze.bat (9 Nov 2026, Christian: "please prepare a freeze.bat that will run on 0.96").
REM The freeze run, unattended and resumable, on its own checkout of the code: D:\code\MPPP_freeze (MPPP_FREEZE_CODE),
REM cloned from this tree the first time.  See mppp_studies.calibration.freeze for the steps (wait, mask, cmods,
REM zjoints, zfinal, ecam, nz, deck, ztests, studies, report) and the faults it rides out.
REM
REM   freeze.bat                        start, or continue after a stop (done steps are skipped)
REM   freeze.bat --status               where it is: running / stopped by the user / done / not running; nothing started
REM   freeze.bat --mask-model auto      wait for the mask training under way and its comparison; the new model is used
REM                                     when it beats the blocks' model by 0.01 exclusion IoU (--mask-margin), else
REM                                     the blocks' model stays (default --mask-model keep: no image processed again)
REM   freeze.bat --update-code          the checkout brought up to this tree's main (git pull --ff-only); nothing started
REM   freeze.bat --now                  start once the pipeline lock is free (do not wait for the first chain)
REM   freeze.bat --redo zjoints         a step again;   freeze.bat --only deck   one step
REM   freeze.bat --zcam-waive-aspect    (default 63 79 110) the zooms whose per-block aspect gate is waived
REM Root: MPPP_FREEZE_ROOT, else D:\rover_blocks.  Models: MPPP_FREEZE_CMODS, else D:\rover_blocks\cmods_freeze2.
REM Log: <root>\mppp_pipeline\freeze_log.txt; status freeze_status.json; each step's output freeze_<step>.log.
REM Stop: Ctrl-C in this window, or stop_mppp.bat - both are logged as "stopped by the user"; start again to go on.
REM After a reboot the run starts itself again at your next logon (RunOnce "MPPP freeze").
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_MAIN=%%~fI"
if not defined MPPP_FREEZE_ROOT set "MPPP_FREEZE_ROOT=D:\rover_blocks"
if not defined MPPP_FREEZE_CODE set "MPPP_FREEZE_CODE=D:\code\MPPP_freeze"
title MPPP freeze (0.96) - %MPPP_FREEZE_ROOT%
call "%MPPP_MAIN%\scripts\windows\mppp_env.bat"
if errorlevel 1 (
  echo freeze: no Python with mppp found - not started
  if not defined MPPP_NOPAUSE pause
  exit /b 1
)
if /i "%MPPP_FREEZE_CODE%"=="%MPPP_MAIN%" goto :have_code
if not exist "%MPPP_FREEZE_CODE%\.git" (
  echo freeze: cloning %MPPP_MAIN% into %MPPP_FREEZE_CODE% ...
  git clone -q "%MPPP_MAIN%" "%MPPP_FREEZE_CODE%" || ( echo clone failed & if not defined MPPP_NOPAUSE pause & exit /b 1 )
)
:have_code
set "MPPP_HOME=%MPPP_FREEZE_CODE%"
set "PYTHONPATH=%MPPP_FREEZE_CODE%\src;%PYTHONPATH%"
set "MPPP_HOME_MAIN=%MPPP_MAIN%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "MPPP_SCAPES=%MPPP_FREEZE_ROOT%"
if exist "%MPPP_MAIN%\scripts\windows\m20_dem_env.bat" call "%MPPP_MAIN%\scripts\windows\m20_dem_env.bat"
set "TPY="
for %%P in ("%MPPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, mppp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY set "TPY=python"
for %%A in (%*) do (
  if /i "%%~A"=="--status" goto :quick
  if /i "%%~A"=="--update-code" goto :quick
  if /i "%%~A"=="-h" goto :quick
  if /i "%%~A"=="--help" goto :quick
)
echo %DATE% %TIME%  freeze: root %MPPP_FREEZE_ROOT%, code %MPPP_FREEZE_CODE%, mask comparison with %TPY%
python -m mppp_studies.calibration.freeze --root "%MPPP_FREEZE_ROOT%" --code "%MPPP_FREEZE_CODE%" --from-tree "%MPPP_MAIN%" --ab-python "%TPY%" %*
set "_RC=%ERRORLEVEL%"
echo %DATE% %TIME%  freeze finished (exit %_RC%)
if not defined MPPP_NOPAUSE pause
exit /b %_RC%
:quick
python -m mppp_studies.calibration.freeze --root "%MPPP_FREEZE_ROOT%" --code "%MPPP_FREEZE_CODE%" --from-tree "%MPPP_MAIN%" %*
set "_RC=%ERRORLEVEL%"
if not defined MPPP_NOPAUSE pause
exit /b %_RC%
