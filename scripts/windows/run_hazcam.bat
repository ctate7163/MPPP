@echo off
REM ======================================================================================================
REM MPPP 0.93 - run_hazcam.bat (Christian, 22 Oct 2026): align the Navcam blocks of the hazcam_consensus group of
REM src\mppp\data\sites.json (chal_rocks_large, threeforks_north, belva_crater, bell_island, marble_mountain) WITH their
REM front and rear Hazcam frames (FHAZ, RHAZ), the Navcam cameras held at the models in use (NAVCAM_INTRINSICS=hold),
REM into <WORK>\colmap_hazcam (variant "hazcam": the Navcam-only result in colmap\ stays), then the Hazcam fit
REM (hazcam_calibration.bat: the Hazcam law in the Navcam form - fx, fy, k1 linear in temperature, early-mission
REM offsets, each pair's rig with a drift in sol - promoted eye by eye and pair by pair when its checks pass).
REM
REM   run_hazcam.bat                         align, then fit and promote
REM   set MPPP_THEN=none                     align only
REM   set MPPP_THEN_ARGS=--no-promote        fit, never promote
REM
REM   batch log  %ROOT%\run_sites_log.txt;  per site <WORK>\runs\<date-time>\log.txt;  status: sites_status.bat
REM ======================================================================================================
setlocal
set "WHICH=--group hazcam_consensus"
set "EXTRA=--variant hazcam --hazcam both --set NAVCAM_INTRINSICS=hold"
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=%~dp0..\.."
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES if exist "D:\scapes" set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & pause & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
if not defined MPPP_THEN set "MPPP_THEN=hazcam_calibration.bat"
if /i "%MPPP_THEN%"=="none" set "MPPP_THEN="
start "MPPP run_hazcam - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" %WHICH% %EXTRA% %*
echo MPPP run_hazcam started in a minimised window, then %MPPP_THEN%. Log: "%ROOT%\run_sites_log.txt"
echo Status of every site: sites_status.bat
timeout /t 10
