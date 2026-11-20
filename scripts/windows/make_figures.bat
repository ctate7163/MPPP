@echo off
REM ======================================================================================================
REM RPP - make_figures.bat (0.91, 20 Oct 2026): the standard figure set, like make_maps.bat
REM
REM Double-click it. Writes into "Claude outputs\figures_<yyyymmdd>" (200 dpi):
REM   navcam_rig_trends_<tag>.png                 rig pitch / yaw / roll against sol, stereo cx against temperature
REM   navcam_intrinsics_vs_temperature_<tag>.png  fx, fy, cx, cy of the joint against temperature, the block bins
REM   navcam_block_table_<tag>.md/.json, rig_drift_refit_<tag>.json
REM   lmst_histogram_<tag>.png + _by_site.csv     the blocks' LMST (Navcam, Mastcam-Z 34)
REM   station_maps_<tag>\<block>.png              the blocks' station maps
REM   zcam_rig_trends_<tag>.png                   Mastcam-Z rig rotations and baseline - CAHV against sol and lens T
REM   zcam_focus_lines_vs_temperature_<tag>.png   the Mastcam-Z blocks' focus lines against lens temperature
REM   deck_fiducials_label_vs_constrained_<tag>.png  0.97.0: the deck fiducials of the sol 1642 Navcam frame (_early_: sol 448, all nine)
REM   deck_fiducials_and_vfs_<tag>.png            0.98.10: the same with the vicarious fiducials (_early_: sol 448)
REM   navcam_baseline_deck_joint_<tag>.png        0.97.0: the deck joint's Navcam baseline against temperature and sol
REM   site_map_<tag>.png                          figure 2 of make_maps (crater | traverse, the official blocks)
REM   zcam<zoom>_efl_vs_focus_<joint>.png         0.95.2: efl vs focus of the newest 34, 48 and 63 mm joints
REM   hazcam_rig_trends_<F|R>_<fit>.png           0.95.2: the newest Hazcam fit: rig trends of each pair
REM   hazcam_efl_vs_temperature_<fit>.png         0.95.2: and the four eyes' focal length against temperature
REM   joss_figure1..4*.png (+ .json)              1.0: the JOSS paper's figures 1-4, 400 dpi (rpp_studies.paper.joss_figures):
REM                                              pipeline diagram (+ .svg); Navcam pairs sol 9 / 52; blocks Butler Landing,
REM                                              Van Zyl Overlook, Rockytop; stations sol 9 / 461 / 52 - each skipped with one
REM                                              line when its input is missing. Freeze-3 blocks: RPP_FREEZE_ROOT (default
REM                                              D:\rover_blocks_f3). RPP_FIG_ARGS=--joss none leaves them out.
REM Other joints or a Hazcam fit: set RPP_FIG_ARGS=--zcam-joints FOLDER ... --hazcam FOLDER (or - for none)
REM
REM   make_figures.bat [NAVCAM_JOINT_FOLDER] [ZCAM_CALIBRATION_FOLDER] [DPI]
REM Defaults: the newest %RPP_SCAPES%\camera_analysis\navcal_tied_* and zcam34_calibration_* folders; 200 dpi.
REM Pass - to skip one (make_figures.bat - D:\scapes\camera_analysis\zcam34_calibration_20261021).
REM Set RPP_NOPAUSE=1 to run it from another script.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
set "RPP_WIN=%RPP_HOME%\scripts\windows"
if exist "%RPP_WIN%\rpp_env.bat" ( call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 ) )
if not defined RPP_SCAPES set "RPP_SCAPES=D:\rover_blocks"
set "CA=%RPP_SCAPES%\camera_analysis"

set "STUDY=%~1"
if "%STUDY%"=="" for /f "delims=" %%D in ('dir /b /ad /o:d "%CA%\navcal_tied_*" 2^>nul') do set "STUDY=%CA%\%%D"
set "ZCAM=%~2"
if "%ZCAM%"=="" for /f "delims=" %%D in ('dir /b /ad /o:d "%CA%\zcam34_calibration_*" 2^>nul') do set "ZCAM=%CA%\%%D"
set "DPI=%~3"
if "%DPI%"=="" set "DPI=200"

call "%RPP_WIN%\m20_dem_env.bat"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%RPP_HOME%\Claude outputs\figures_%TODAY%"
if not exist "%OUT%" mkdir "%OUT%"

set "ARGS=--root "%RPP_SCAPES%" --out "%OUT%" --dpi %DPI%"
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
python -m rpp_studies.paper.paper_figures !ARGS! %RPP_FIG_ARGS%
if errorlevel 1 ( echo make_figures: FAILED ^(see above^) & if not defined RPP_NOPAUSE pause & exit /b 1 )
echo make_figures: done - %OUT%
if not defined RPP_NOPAUSE pause
exit /b 0
