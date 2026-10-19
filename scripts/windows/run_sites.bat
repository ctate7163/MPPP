@echo off
REM ======================================================================================================
REM MPPP - process and align the Navcam block of every site of the navcam_consensus group of
REM src\mppp\data\sites.json (v0p53) into its own WORK folder mars2020_sol_<sol>_<site> under ROOT
REM (Navcam only; the Mastcam-Z blocks: run_sites_zcam.bat, run_sites_zcam34.bat ... _zcam110),
REM one site after the other, in the background (a minimised window; closing it stops the batch).
REM Finished sites (same settings) are skipped, so double-clicking again continues where it stopped.
REM
REM   batch log     %ROOT%\run_sites_log.txt
REM   per site      %ROOT%\mars2020_sol_<sol>_<site>\runs\<date-time>\log.txt and mppp_status.json
REM   status        sites_status.bat
REM
REM Edit the settings below (or run scripts\run_sites.py from a prompt; --help lists every option).
REM ======================================================================================================
setlocal
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
REM which sites: --group navcam_consensus (default) or --sites rockytop sid_chal_rocks
set "WHICH=--group navcam_consensus"
REM more options, e.g. --jobs 2   (two sites at a time)
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
start "MPPP run_sites - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% %EXTRA% %*
echo MPPP run_sites started in a minimised window. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
