@echo off
REM ======================================================================================================
REM MPPP - select and process the images of every site (or some of them) into their WORK folders under ROOT,
REM one site after the other, in the background (a minimised window; closing it stops it). No alignment.
REM
REM Uses the newest notebook 03 in MPPP_HOME\notebooks with its default settings (sections 1-2: PDS selection
REM and image processing) and the site list in src\mppp\data\sites.json.
REM   result        %ROOT%\<site>_colmap[_nav_zcam34]\processed\   (images, masks, manifest, process_done.json)
REM   batch log     %ROOT%\process_sites_log.txt
REM   per site      %ROOT%\<site>_colmap[_nav_zcam34]\runs\<date-time>_process\log.txt and mppp_status.json
REM   status        sites_status.bat
REM Sites already processed with the same settings are skipped, so double-clicking again continues.
REM Then align a site: copy align_here.bat into its WORK folder and double-click it.
REM
REM Edit the settings below (or run scripts\process_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
REM which sites: --all, --group nav_zcam34, --group navcam_consensus, or --sites rockytop sid south_arm
set "WHICH=--all"
REM Mastcam-Z 34 mm too (folders <site>_colmap_nav_zcam34): set ZCAM=--zcam
set "ZCAM="
REM more options, e.g. --force (process again)   --set SKY_ELEVATION_DEG=20   --dry-run
set "EXTRA="

call "%~dp0mppp_env.bat" || ( pause & exit /b 1 )
start "MPPP process_sites - %ROOT%" /min cmd /c call "%~dp0_run_process.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo MPPP process_sites started in a minimised window. Log: "%ROOT%\process_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
