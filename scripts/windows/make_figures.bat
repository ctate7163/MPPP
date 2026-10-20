@echo off
REM ======================================================================================================
REM MPPP - make_figures.bat (0.91, 20 Oct 2026): the standard figure set, like make_maps.bat
REM
REM Double-click it. Writes into "Claude outputs\figures_<yyyymmdd>" (200 dpi):
REM   navcam_rig_trends_<tag>.png                 rig pitch / yaw / roll against sol, stereo cx against temperature
REM   navcam_intrinsics_vs_temperature_<tag>.png  fx, fy, cx, cy of the joint against temperature, the block bins
REM   navcam_block_table_<tag>.md/.json, rig_drift_refit_<tag>.json
REM   lmst_histogram_<tag>.png + _by_site.csv     the blocks' LMST (Navcam, Mastcam-Z 34)
REM   station_maps_<tag>\<block>.png              the blocks' station maps
REM   zcam_rig_trends_<tag>.png                   Mastcam-Z rig rotations and baseline - CAHV against sol and lens T
REM   zcam_focus_lines_vs_temperature_<tag>.png   the Mastcam-Z blocks' focus lines against lens temperature
REM   site_map_<tag>.png                          figure 2 of make_maps (crater | traverse, the official blocks)
REM
REM   make_figures.bat [NAVCAM_JOINT_FOLDER] [ZCAM_CALIBRATION_FOLDER] [DPI]
REM Defaults: the newest %MPPP_SCAPES%\camera_analysis\navcal_tied_* and zcam34_calibration_* folders; 200 dpi.
REM Pass - to skip one (make_figures.bat - D:\scapes\camera_analysis\zcam34_calibration_20261021).
REM Set MPPP_NOPAUSE=1 to run it from another script.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if exist "%MPPP_WIN%\mppp_env.bat" ( call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 ) )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
set "CA=%MPPP_SCAPES%\camera_analysis"

set "STUDY=%~1"
if "%STUDY%"=="" for /f "delims=" %%D in ('dir /b /ad /o:d "%CA%\navcal_tied_*" 2^>nul') do set "STUDY=%CA%\%%D"
set "ZCAM=%~2"
if "%ZCAM%"=="" for /f "delims=" %%D in ('dir /b /ad /o:d "%CA%\zcam34_calibration_*" 2^>nul') do set "ZCAM=%CA%\%%D"
set "DPI=%~3"
if "%DPI%"=="" set "DPI=200"

call "%MPPP_WIN%\m20_dem_env.bat"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_HOME%\Claude outputs\figures_%TODAY%"
if not exist "%OUT%" mkdir "%OUT%"

set "ARGS=--root "%MPPP_SCAPES%" --out "%OUT%" --dpi %DPI%"
if not "%STUDY%"=="-" if not "%STUDY%"=="" (
  set "ARGS=!ARGS! --study "%STUDY%""
  for %%F in ("%STUDY%") do set "TAG=%%~nxF"
)
if not "%ZCAM%"=="-" if not "%ZCAM%"=="" set "ARGS=!ARGS! --zcam "%ZCAM%""
if not defined TAG if not "%ZCAM%"=="-" if not "%ZCAM%"=="" for %%F in ("%ZCAM%") do set "TAG=%%~nxF"
if defined TAG set "ARGS=!ARGS! --tag !TAG!"

echo make_figures: Navcam joint  %STUDY%
echo make_figures: Mastcam-Z     %ZCAM%
echo make_figures: DTMs          %M20_DEMS%
echo make_figures: out           %OUT%  (%DPI% dpi)
python "%MPPP_HOME%\scripts\paper_figures.py" !ARGS!
if errorlevel 1 ( echo make_figures: FAILED ^(see above^) & if not defined MPPP_NOPAUSE pause & exit /b 1 )
echo make_figures: done - %OUT%
if not defined MPPP_NOPAUSE pause
exit /b 0
