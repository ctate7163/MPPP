@echo off
REM ======================================================================================================
REM MPPP v0p43 - process and align every site of src\mppp\data\sites.json into its own WORK folder under ROOT,
REM one site after the other, in the background (a minimised window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\<site>_colmap[_nav_zcam34]\runs\<date-time>\log.txt and mppp_status.json
REM   status        sites_status.bat
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
REM which sites: --all, --group nav_zcam34, --group navcam_consensus, or --sites rockytop sid
set "WHICH=--all"
REM Mastcam-Z 34 mm too (folders <site>_colmap_nav_zcam34): set ZCAM=--zcam
set "ZCAM="
REM more options, e.g. --then 04 05   (camera models and error analysis at the end)
REM                    --source processed --variant tight --set ATTITUDE_PRIOR_DEG=1.0
set "EXTRA="

call "%~dp0mppp_env.bat" || ( pause & exit /b 1 )
start "MPPP run_sites - %ROOT%" /min cmd /c call "%~dp0_run_sites.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo MPPP run_sites started in a minimised window. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
