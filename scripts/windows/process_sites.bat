@echo off
REM ======================================================================================================
REM MPPP - select and process the images of every site (or some of them) into their WORK folders under ROOT,
REM one site after the other, in the background (a minimised window; closing it stops it). No alignment.
REM
REM Uses the newest notebook 03 in MPPP_HOME\notebooks with its default settings (sections 1-2: PDS selection
REM and image processing) and the site list in src\mppp\data\sites.json.
REM   result        %ROOT%\mars2020_sol_<sol>_<site>[_zcam]\processed\   (images, masks, manifest, process_done.json)
REM   batch log     %ROOT%\process_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>[_zcam]\runs\<date-time>_process\log.txt and mppp_status.json
REM   status        sites_status.bat
REM Sites already processed with the same settings are skipped, so double-clicking again continues.
REM Then align a site: copy align_here.bat into its WORK folder and double-click it.
REM
REM Edit the settings below (or run scripts\process_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
REM which sites: --all, --group navcam_consensus, or --sites rockytop sid_chal_rocks south_arm
set "WHICH=--all"
REM the Navcam + Mastcam-Z blocks of the sites with a "zcam" list (folders ..._zcam): set ZCAM=--zcam
set "ZCAM="
REM more options, e.g. --force (process again)   --set SKY_ELEVATION_DEG=20   --dry-run
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
start "MPPP process_sites - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_process.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo MPPP process_sites started in a minimised window. Log: "%ROOT%\process_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
