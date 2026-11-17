@echo off
REM ======================================================================================================
REM RPP - process and align the Navcam + Mastcam-Z block of every site of the zcam_consensus group of
REM src\rpp\data\sites.json into its own WORK folder mars2020_sol_<sol>_<site>_zcam under ROOT (v0p53;
REM the block has all the Mastcam-Z zooms of the site's "zcam" list),
REM one site after the other, in the background (a minimized window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>_zcam\runs\<date-time>\log.txt and mppp_status.json
REM   status        check_sites.bat
REM
REM 0.93 (Christian, 22 Oct): then zcam_joints.bat - the Mastcam-Z joint of every zoom over the good blocks, a new
REM version of each zoom's focus model, promoted when every check passes ("set RPP_THEN=none" before calling this
REM file aligns only; "set RPP_THEN_ARGS=--no-promote" never promotes).
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
REM which sites: --group zcam_consensus (default) or --sites a b
set "WHICH=--group zcam_consensus"
REM more options, e.g. --jobs 1   (one site at a time; two is the default since 0.94.3)
REM                    --source processed --variant tight --set ATTITUDE_PRIOR_DEG=1.0
set "EXTRA="

REM RPP: this file's own folder if it is RPP's scripts\windows, else RPP_HOME, so this
REM .bat also works when copied elsewhere (e.g. into the scapes root).
if not defined RPP_HOME if exist "%~dp0rpp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
if not defined RPP_HOME set "RPP_HOME=%~dp0..\.."
set "RPP_WIN=%RPP_HOME%\scripts\windows"
if not exist "%RPP_WIN%\rpp_env.bat" (
  echo RPP not found in "%RPP_HOME%": set RPP_HOME to the RPP folder
  pause
  exit /b 1
)
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
REM 20 Oct: RPP_SCAPES after rpp_env.bat (it may come from rpp_scapes.txt), else D:\rover_blocks when it exists
if not defined RPP_SCAPES if exist "D:\rover_blocks" set "RPP_SCAPES=D:\rover_blocks"
if not defined RPP_SCAPES (echo set RPP_SCAPES to the scapes root, e.g. setx RPP_SCAPES D:\rover_blocks & pause & exit /b 1)
set "ROOT=%RPP_SCAPES%"
if not defined RPP_THEN set "RPP_THEN=zcam_joints.bat"
if /i "%RPP_THEN%"=="none" set "RPP_THEN="
start "RPP run_sites zcam - %ROOT%" /min cmd /c call "%RPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% --zcam %EXTRA% %*
echo RPP run_sites started in a minimized window, then %RPP_THEN%. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: check_sites.bat
timeout /t 10
