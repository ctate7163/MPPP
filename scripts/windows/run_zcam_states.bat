@echo off
REM ======================================================================================================
REM MPPP 0.94.4 - run_zcam_states.bat (Christian, 25 Oct 2026): which Mastcam-Z frames are in the low backlash state?
REM
REM Aligns the Navcam + Mastcam-Z block of each site again as the variant "states" (WORK\colmap_states; the
REM blocks' own colmap\ is not touched, and its features and matches are reused) with every Mastcam-Z image as a
REM camera of its own, its focal length free (ZCAM_BIN_BY=image, ZCAM_HOLD_F_IMAGES=0, ZCAM_BIN_REFINE=focal) and
REM no focus lines (ZCAM_FOCUS_LINE=false), so that each image's f / label f is its own.  The low-state list and the
REM found list (%MPPP_SCAPES%\zcam_low_backlash_found.csv) give only the starting focal lengths.
REM
REM Then "run_zcam_states.bat report": "mppp zcam-states --variant states" - per block, eye and zoom the high-state line
REM in focus count and the state gap, each image low / high / undecided, each sequence's pattern in clock order
REM (sol 121 ZCAM08114: LHLHL), and the proposal for the found list (additions; "-" lines for listed frames found
REM high; the cases to review) in %MPPP_SCAPES%\camera_analysis\zcam_states_<date>\.  Nothing is written to the
REM found list until you run:  python -m mppp zcam-states --root %MPPP_SCAPES% --variant states --apply
REM
REM   run_zcam_states.bat                      the ten weekend Mastcam-Z sites (below)
REM   set MPPP_STATES_SITES=chal_rocks rockytop   other sites (any with a "zcam" list in sites.json)
REM   run_zcam_states.bat report [--apply]     only the report (after the run; --apply writes the found list)
REM   set MPPP_THEN=none                        align only, no report
REM Do not start it while another run_sites batch is running (run_sites refuses: one batch at a time).
REM ======================================================================================================
setlocal
if not defined MPPP_STATES_SITES set "MPPP_STATES_SITES=seitah_overlook chal_rocks threeforks_north belva_crater south_arm marble_mountain taylorfjellet rochette enchanted_lake olifants"
set "SETS=--set ZCAM_BIN_BY=image --set ZCAM_HOLD_F_IMAGES=0 --set ZCAM_BIN_REFINE=focal --set ZCAM_FOCUS_LINE=false"

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
if /i "%~1"=="report" (
  shift
  python -m mppp zcam-states --root "%ROOT%" --variant states %2 %3 %4
  if not defined MPPP_NOPAUSE_END pause
  exit /b
)
if not defined MPPP_THEN (
  set "MPPP_THEN=run_zcam_states.bat"
  set "MPPP_THEN_ARGS=report"
)
if /i "%MPPP_THEN%"=="none" set "MPPP_THEN="
start "MPPP Mastcam-Z states - %ROOT%" /min cmd /c call "%MPPP_WIN%\_run_sites.bat" --root "%ROOT%" --sites %MPPP_STATES_SITES% --zcam --variant states --jobs 2 %SETS% %*
echo MPPP Mastcam-Z states run started in a minimised window (variant "states"), then %MPPP_THEN%.
echo Log: "%ROOT%\run_sites_log.txt"; status: check_sites.bat
timeout /t 10
