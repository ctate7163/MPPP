@echo off
REM ======================================================================================================
REM MPPP - select and process the images of every site (or some of them) into their WORK folders under ROOT,
REM one site after the other, in the background (a minimised window; closing it stops it). No alignment.
REM
REM Uses the newest notebook 03 in MPPP_HOME\notebooks with its default settings (sections 1-2: PDS selection
REM and image processing) and the site list in src\mppp\data\sites.json.
REM   result        %ROOT%\<site>_colmap[_zcam34]\processed\   (images, masks, manifest, process_done.json)
REM   batch log     %ROOT%\process_sites_log.txt
REM   per site      %ROOT%\<site>_colmap[_zcam34]\runs\<date-time>_process\log.txt and mppp_status.json
REM   status        sites_status.bat
REM Sites already processed with the same settings are skipped, so double-clicking again continues.
REM Then align a site: copy align_here.bat into its WORK folder and double-click it.
REM
REM Edit the settings below (or run scripts\process_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
REM which sites: --all, --group navcam_consensus, or --sites rockytop sid_chal_rocks south_arm
set "WHICH=--all"
REM the Navcam + Mastcam-Z 34 mm blocks of the zcam34 sites (folders <site>_colmap_zcam34): set ZCAM=--zcam
set "ZCAM="
REM more options, e.g. --force (process again)   --set SKY_ELEVATION_DEG=20   --dry-run
set "EXTRA="

REM MPPP: this file's own folder if it is MPPP's scripts\windows, else MPPP_HOME (default D:\code\MPPP), so this
REM .bat also works when copied elsewhere (e.g. into D:\scapes\colmap).
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
start "MPPP process_sites - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_process.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo MPPP process_sites started in a minimised window. Log: "%ROOT%\process_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
