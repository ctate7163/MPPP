@echo off
REM ======================================================================================================
REM MPPP - process and align the Navcam + Mastcam-Z block of every site of the zcam34_consensus group of
REM src\mppp\data\sites.json into its own WORK folder mars2020_sol_<sol>_<site>_zcam under ROOT (v0p53;
REM v0p61: only the 34 mm Mastcam-Z frames (ZCAM_ZOOMS=[34]); run_sites_zcam.bat aligns all the site's zooms),
REM one site after the other, in the background (a minimised window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>_zcam\runs\<date-time>\log.txt and mppp_status.json
REM   status        sites_status.bat
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
REM which sites: --group zcam34_consensus (default) or --sites a b
set "WHICH=--group zcam34_consensus"
REM v0p61: only the 34 mm frames. The results go to <WORK>\colmap like run_sites_zcam.bat (the last run wins);
REM add --variant z34 to keep them apart in <WORK>\colmap_z34 (notebook 04 reads <WORK>\colmap only).
REM more options, e.g. --then 04 05   (camera models and error analysis at the end)
REM                    --source processed --set ATTITUDE_PRIOR_DEG=1.0
set "EXTRA=--set ZCAM_ZOOMS=[34]"

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
start "MPPP run_sites zcam34 - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% --zcam %EXTRA% %*
echo MPPP run_sites started in a minimised window. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
