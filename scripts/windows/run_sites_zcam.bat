@echo off
REM ======================================================================================================
REM MPPP - process and align the Navcam + Mastcam-Z block of every site of the zcam_consensus group of
REM src\mppp\data\sites.json into its own WORK folder mars2020_sol_<sol>_<site>_zcam under ROOT (v0p53;
REM the block has all the Mastcam-Z zooms of the site's "zcam" list),
REM one site after the other, in the background (a minimised window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>_zcam\runs\<date-time>\log.txt and mppp_status.json
REM   status        check_sites.bat
REM
REM 0.93 (Christian, 22 Oct): then zcam_joints.bat - the Mastcam-Z joint of every zoom over the good blocks, a new
REM version of each zoom's focus model, promoted when every check passes ("set MPPP_THEN=none" before calling this
REM file aligns only; "set MPPP_THEN_ARGS=--no-promote" never promotes).
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
REM which sites: --group zcam_consensus (default) or --sites a b
set "WHICH=--group zcam_consensus"
REM more options, e.g. --jobs 1   (one site at a time; two is the default since 0.94.3)
REM                    --source processed --variant tight --set ATTITUDE_PRIOR_DEG=1.0
set "EXTRA="

REM MPPP: this file's own folder if it is MPPP's scripts\windows, else MPPP_HOME, so this
REM .bat also works when copied elsewhere (e.g. into the scapes root).
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=%~dp0..\.."
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
REM 20 Oct: MPPP_SCAPES after mppp_env.bat (it may come from mppp_scapes.txt), else D:\scapes when it exists
if not defined MPPP_SCAPES if exist "D:\scapes" set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & pause & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
if not defined MPPP_THEN set "MPPP_THEN=zcam_joints.bat"
if /i "%MPPP_THEN%"=="none" set "MPPP_THEN="
start "MPPP run_sites zcam - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% --zcam %EXTRA% %*
echo MPPP run_sites started in a minimised window, then %MPPP_THEN%. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: check_sites.bat
timeout /t 10
