@echo off
REM ======================================================================================================
REM MPPP - make_maps.bat (0.90, 20 Oct 2026): the six site-map figures from src\mppp\data\sites.json
REM
REM Double-click it. Writes into "Claude outputs\maps_<yyyymmdd>":
REM   m20_cluster_zooms_<tag>.png                       figure 1   +-30 m panels of every official block
REM   m20_jezero_traverse_clusters_map_<tag>.png        figure 2   crater | traverse, block table
REM   m20_jezero_traverse_sites_regions_<tag>.png       figure 3   + six regional panels
REM   m20_jezero_traverse_sites_regions_wide_<tag>.png  figure 4
REM   m20_mars_to_rover_sites_<tag>.png                 figure 5   Mars -> Isidis -> Jezero -> traverse -> regions
REM   m20_mars_to_rover_sites_no_I_<tag>.png            figure 6   figure 5 with the block list instead of panel I
REM   m20_traverse_banner_<tag>.png                     figure 7   the GitHub banner (the traverse, the title)
REM   m20_cluster_zooms_white.png                       the zooms on white (with dtm / both)
REM One set per background: tag "dtm" (1 m HiRISE DTM where it covers, ~70 m tiles elsewhere) and, when an
REM orthomosaic is found, tag "ortho".  Arguments: make_maps.bat [dtm|ortho|both] [DPI]   (default: both 200).
REM
REM Needs in the MPPP environment: matplotlib scipy pillow adjustText rasterio pyproj  (pip install -e ".[maps]").
REM DTM / orthomosaic paths: M20_DEMS / M20_ORTHOS if set, otherwise the files under D:\scapes\dem (below).
REM One line over the CLI: mppp maps (the figures are the study of studies\m20_sites\m20_sites, one process per step).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if exist "%MPPP_WIN%\mppp_env.bat" ( call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 ) )

set "WHICH=%~1"
if "%WHICH%"=="" set "WHICH=both"
if not "%~2"=="" set "M20_DPI=%~2"
if not defined M20_DPI set "M20_DPI=200"

call "%MPPP_WIN%\m20_dem_env.bat"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_HOME%\Claude outputs\maps_%TODAY%"
set "M20_SITES_FILE=%MPPP_HOME%\src\mppp\data\sites.json"

echo make_maps: sites   %M20_SITES_FILE%
echo make_maps: DTMs    %M20_DEMS%
echo make_maps: orthos  %M20_ORTHOS%
echo make_maps: out     %OUT%  (%M20_DPI% dpi, %WHICH%)
if not exist "%OUT%" mkdir "%OUT%"
if /i not "%WHICH%"=="ortho" (
  python -m mppp maps all --out "%OUT%" --background dtm --tag dtm --dpi %M20_DPI%
  if errorlevel 1 goto :fail
)
if /i not "%WHICH%"=="ortho" (
  python -m mppp maps zooms --out "%OUT%" --background white --tag white --dpi %M20_DPI%
  if errorlevel 1 goto :fail
)
if /i not "%WHICH%"=="dtm" if defined M20_ORTHOS (
  python -m mppp maps all --out "%OUT%" --background ortho --tag ortho --dpi %M20_DPI%
  if errorlevel 1 goto :fail
)
echo make_maps: done - %OUT%
pause
exit /b 0

:fail
echo make_maps: FAILED (see above)
pause
exit /b 1
