@echo off
REM ======================================================================================================
REM RPP - select and process the images of every site (or some of them) into their WORK folders under ROOT,
REM one site after the other, in the background (a minimized window; closing it stops it). No alignment.
REM
REM Uses the newest notebook 03 in RPP_HOME\notebooks with its default settings (sections 1-2: PDS selection
REM and image processing) and the site list in src\rpp\data\sites.json.
REM   result        %ROOT%\mars2020_sol_<sol>_<site>[_zcam]\processed\   (images, masks, manifest, process_done.json)
REM   batch log     %ROOT%\process_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>[_zcam]\runs\<date-time>_process\log.txt and mppp_status.json
REM   status        check_sites.bat
REM Sites already processed with the same settings are skipped, so double-clicking again continues.
REM Then align a site: copy align_here.bat into its WORK folder and double-click it.
REM
REM Edit the settings below (or run scripts\process_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
REM which sites: --all, --group ecam_consensus, or --sites rockytop sid_chal_rocks south_arm
set "WHICH=--all"
REM the Navcam + Mastcam-Z blocks of the sites with a "zcam" list (folders ..._zcam): set ZCAM=--zcam
set "ZCAM="
REM more options, e.g. --force (process again)   --set SKY_ELEVATION_DEG=20   --dry-run
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
start "RPP process_sites - %ROOT%" /min cmd /c call "%RPP_WIN%\_run_process.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo RPP process_sites started in a minimized window. Log: "%ROOT%\process_sites_log.txt"
echo Status of every site: check_sites.bat
timeout /t 10
