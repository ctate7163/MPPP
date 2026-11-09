@echo off
REM ======================================================================================================
REM MPPP - make_maps.bat (0.96, 9 Nov 2026): the site-map figures from src\mppp\data\sites.json
REM
REM 9 Nov 2026: the waypoint table is downloaded first (mppp waypoints --refresh, the current M20_waypoints.json from
REM mars.nasa.gov) and copied into the study folder, so the traverse, the block windows and the N/Z counts are drawn
REM from the latest waypoints; when the download fails the cached copy is used (a warning says so).  The orthomosaic
REM panels carry no contour lines any more (the DTM panels keep their 100 m contours; set M20_CONTOURS=0 to drop them
REM there too).
REM
REM Double-click it. Writes into "Claude outputs\maps_<yyyymmdd>":
REM   m20_cluster_zooms_<tag>.png                       figure 1   +-30 m panels of every official block
REM   m20_jezero_traverse_clusters_map_<tag>.png        figure 2   crater | traverse, block table
REM   m20_jezero_traverse_sites_regions_<tag>.png       figure 3   + six regional panels
REM   m20_jezero_traverse_sites_regions_wide_<tag>.png  figure 4
REM   m20_mars_to_rover_sites_<tag>.png                 figure 5   Mars -> Isidis -> Jezero -> traverse -> regions
REM   m20_mars_to_rover_sites_no_I_<tag>.png            figure 6   figure 5 with the block list instead of panel I
REM   m20_traverse_banner_<tag>.png                     figure 7   the GitHub banner (terrain and the pipeline name)
REM   m20_cluster_zooms_white.png                       the zooms on white (with dtm / both)
REM One set per background: tag "dtm" (1 m HiRISE DTM where it covers, ~70 m tiles elsewhere) and, when an
REM orthomosaic is found, tag "ortho".  Arguments: make_maps.bat [dtm|ortho|both] [DPI] [study]   (default: both 200;
REM "study" skips the waypoint download and draws the study folder's own M20_waypoints.json).
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
set "WPT=refresh"
if /i "%~3"=="study" set "WPT=study"

call "%MPPP_WIN%\m20_dem_env.bat"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_HOME%\Claude outputs\maps_%TODAY%"
set "M20_SITES_FILE=%MPPP_HOME%\src\mppp\data\sites.json"

echo make_maps: sites   %M20_SITES_FILE%
echo make_maps: DTMs    %M20_DEMS%
echo make_maps: orthos  %M20_ORTHOS%
echo make_maps: out     %OUT%  (%M20_DPI% dpi, %WHICH%, waypoints %WPT%)
if not exist "%OUT%" mkdir "%OUT%"
REM the first step downloads the table and copies it into the study folder (a failed download: warning, cached
REM copy); the later steps draw that copy (study)
set "WPT1=%WPT%"
if /i not "%WHICH%"=="ortho" (
  python -m mppp maps all --out "%OUT%" --background dtm --tag dtm --dpi %M20_DPI% --waypoints !WPT1!
  if errorlevel 1 goto :fail
  set "WPT1=study"
)
if /i not "%WHICH%"=="ortho" (
  python -m mppp maps zooms --out "%OUT%" --background white --tag white --dpi %M20_DPI% --waypoints !WPT1!
  if errorlevel 1 goto :fail
  set "WPT1=study"
)
if /i not "%WHICH%"=="dtm" if defined M20_ORTHOS (
  python -m mppp maps all --out "%OUT%" --background ortho --tag ortho --dpi %M20_DPI% --waypoints !WPT1!
  if errorlevel 1 goto :fail
)
echo make_maps: done - %OUT%
if not defined MPPP_NOPAUSE pause
exit /b 0

:fail
echo make_maps: FAILED (see above)
if not defined MPPP_NOPAUSE pause
exit /b 1
