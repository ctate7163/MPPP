@echo off
REM ======================================================================================================
REM RPP - process and align the Navcam block of every site of the ecam_consensus group of
REM src\rpp\data\sites.json (v0p53) into its own WORK folder mars2020_sol_<sol>_<site> under ROOT
REM (Navcam only; the Mastcam-Z blocks: run_sites_zcam.bat),
REM one site after the other, in the background (a minimized window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>\runs\<date-time>\log.txt and mppp_status.json
REM   status        check_sites.bat
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
REM which sites: --group ecam_consensus (default) or --sites rockytop sid_chal_rocks
set "WHICH=--group ecam_consensus"
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
REM 0.98.19: run_sites.bat --sites hidden_harbor - a --sites or --group given replaces the group above
echo.%* | findstr /i /c:"--sites" /c:"--group" >nul && set "WHICH="
start "RPP run_sites - %ROOT%" /min cmd /c call "%RPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% %EXTRA% %*
echo RPP run_sites started in a minimized window. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: check_sites.bat
timeout /t 10
