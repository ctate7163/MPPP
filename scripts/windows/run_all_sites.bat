@echo off
REM ======================================================================================================
REM MPPP - process and align every site of src\mppp\data\sites.json into its own WORK folder under ROOT,
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
start "MPPP run_sites - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% %ZCAM% %EXTRA% %*
echo MPPP run_sites started in a minimised window. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
