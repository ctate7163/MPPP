@echo off
REM ======================================================================================================
REM RPP 0.94.4 - run_zcam_states.bat (Christian, 25 Oct 2026): which Mastcam-Z frames are in the low backlash state?
REM
REM Aligns the Navcam + Mastcam-Z block of each site again as the variant "states" (WORK\colmap_states; the
REM blocks' own colmap\ is not touched, and its features and matches are reused) with every Mastcam-Z image as a
REM camera of its own, its focal length free (ZCAM_BIN_BY=image, ZCAM_HOLD_F_IMAGES=0, ZCAM_BIN_REFINE=focal_pp - the right eye's center free too) and
REM no focus lines (ZCAM_FOCUS_LINE=false), so that each image's f / label f is its own.  The low-state list and the
REM found list (%RPP_SCAPES%\zcam_low_backlash_found.csv) give only the starting focal lengths.
REM
REM Then "run_zcam_states.bat report": "rpp zcam-states --variant states" - per block, eye and zoom the high-state line
REM in focus count and the state gap, each image low / high / undecided, each sequence's pattern in clock order
REM (sol 121 ZCAM08114: LHLHL), and the proposal for the found list (additions; "-" lines for listed frames found
REM high; the cases to review) in %RPP_SCAPES%\camera_analysis\zcam_states_<date>\.  Nothing is written to the
REM found list until you run:  python -m rpp zcam-states --root %RPP_SCAPES% --variant states --apply
REM
REM   run_zcam_states.bat                      the ten weekend Mastcam-Z sites (below)
REM   set RPP_STATES_SITES=chal_rocks rockytop   other sites (any with a "zcam" list in sites.json)
REM   run_zcam_states.bat report [--apply]     only the report (after the run; --apply writes the found list)
REM   set RPP_THEN=none                        align only, no report
REM Do not start it while another run_sites batch is running (run_sites refuses: one batch at a time).
REM ======================================================================================================
setlocal
call "%~dp0rpp_env.bat" --legacy-only
if not defined RPP_STATES_SITES set "RPP_STATES_SITES=seitah_overlook chal_rocks threeforks_north belva_crater south_arm marble_mountain taylorfjellet rochette enchanted_lake olifants"
set "SETS=--set ZCAM_BIN_BY=image --set ZCAM_HOLD_F_IMAGES=0 --set ZCAM_BIN_REFINE=focal_pp --set ZCAM_FOCUS_LINE=false"

if not defined RPP_HOME if exist "%~dp0rpp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
if not defined RPP_HOME set "RPP_HOME=%~dp0..\.."
set "RPP_WIN=%RPP_HOME%\scripts\windows"
if not exist "%RPP_WIN%\rpp_env.bat" (
  echo RPP not found in "%RPP_HOME%": set RPP_HOME to the RPP folder
  pause
  exit /b 1
)
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
if not defined RPP_SCAPES if exist "D:\rover_blocks" set "RPP_SCAPES=D:\rover_blocks"
if not defined RPP_SCAPES (echo set RPP_SCAPES to the scapes root, e.g. setx RPP_SCAPES D:\rover_blocks & pause & exit /b 1)
set "ROOT=%RPP_SCAPES%"
if /i "%~1"=="report" (
  shift
  python -m rpp zcam-states --root "%ROOT%" --variant states %2 %3 %4
  if not defined RPP_NOPAUSE_END pause
  exit /b
)
if not defined RPP_THEN (
  set "RPP_THEN=run_zcam_states.bat"
  set "RPP_THEN_ARGS=report"
)
if /i "%RPP_THEN%"=="none" set "RPP_THEN="
start "RPP Mastcam-Z states - %ROOT%" /min cmd /c call "%RPP_WIN%\_run_sites.bat" --root "%ROOT%" --sites %RPP_STATES_SITES% --zcam --variant states --jobs 2 %SETS% %*
echo RPP Mastcam-Z states run started in a minimized window (variant "states"), then %RPP_THEN%.
echo Log: "%ROOT%\run_sites_log.txt"; status: check_sites.bat
timeout /t 10
