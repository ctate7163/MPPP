@echo off
REM ======================================================================================================
REM MPPP 0.95.3 - zcam_backlash_rerun.bat (Christian, 31 Oct 2026: "Before the final run, I need one good zcam
REM joint to capture all of the problematic backlash anomalies. Prepare the rerun of these three blocks with all
REM zooms including z110. Allow more joint iterations for convergence.")
REM
REM In one minimised window, one step after the other (about a day):
REM   1. the backlash states run of the three blocks (variant "states": every Mastcam-Z image its own focal
REM      length; run_zcam_states.bat's settings; variant states_1031) - Chal Rocks, Seitah Overlook, Rockytop (never had one);
REM   2. the states report of those blocks and the found list updated from it (--apply: a dated backup first;
REM      set MPPP_STATES_APPLY=0 to stop after the report and look first);
REM   3. the three blocks aligned again (--force: the found list is not part of the run key) with every zoom of
REM      their "zcam" lists - 34, 48, 63, 79 and 110 mm (Rockytop: 110 mm only its zcam_include sequences);
REM      sites.json leaves out Seitah Overlook's two mis-tied ZR 79 mm frames of ZCAM08114 and Rockytop's
REM      ZCAM03391 at 79 mm;
REM   4. zcam_joints.bat for 34 48 63 79 110 mm over every zcam_consensus block, --no-promote, with
REM      --iterations %MPPP_JOINT_ITERATIONS% (default 300; the Z48 joint of 31 Oct stopped at 101 twice), no
REM      Hazcam fit (MPPP_HAZ=none), then the Navcam + Mastcam-Z 34 mm check.
REM Nothing is promoted: look at camera_analysis\zcam*_joint_<date>\report.md, then promote_models.bat.
REM
REM   set MPPP_RERUN_SITES=chal_rocks rockytop    other sites
REM   set MPPP_RERUN_FROM=blocks                  start at step 3 (states | report | blocks | joints)
REM   stop_mppp.bat                               stops it (no later step starts)
REM Log: %MPPP_SCAPES%\zcam_backlash_rerun_log.txt (and run_sites_log.txt, the joints' own logs)
REM ======================================================================================================
setlocal EnableExtensions
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=%~dp0..\.."
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES if exist "D:\rover_blocks" set "MPPP_SCAPES=D:\rover_blocks"
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\rover_blocks & pause & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
if not defined MPPP_RERUN_SITES set "MPPP_RERUN_SITES=chal_rocks seitah_overlook rockytop"
if not defined MPPP_JOINT_ITERATIONS set "MPPP_JOINT_ITERATIONS=300"
if not defined MPPP_RERUN_FROM set "MPPP_RERUN_FROM=states"
REM a new variant (colmap_states_1031): seeded from the blocks' current features and matches, not from the 25 Oct
REM states run (before the 0.95.2 feature counts and the 79 / 110 mm frames)
if not defined MPPP_STATES_VARIANT set "MPPP_STATES_VARIANT=states_1031"
set "LOG=%ROOT%\zcam_backlash_rerun_log.txt"
if /i "%~1"=="--chain" goto :chain

if exist "%ROOT%\camera_analysis\zcam_joints.lock" (
  echo zcam_joints.bat is running ^(%ROOT%\camera_analysis\zcam_joints.lock^): start this after it has finished.
  pause
  exit /b 1
)
start "MPPP Mastcam-Z backlash rerun - %ROOT%" /min cmd /c call "%~f0" --chain
echo MPPP Mastcam-Z backlash rerun started in a minimised window: %MPPP_RERUN_SITES% (from step %MPPP_RERUN_FROM%).
echo Log: "%LOG%"; status: check_sites.bat; stop: stop_mppp.bat
timeout /t 10
exit /b 0

:chain
title MPPP Mastcam-Z backlash rerun
del "%ROOT%\mppp_stopped.txt" 2>nul
set "SETS=--set ZCAM_BIN_BY=image --set ZCAM_HOLD_F_IMAGES=0 --set ZCAM_BIN_REFINE=focal_pp --set ZCAM_FOCUS_LINE=false"
echo %DATE% %TIME%  zcam_backlash_rerun: %MPPP_RERUN_SITES%, from %MPPP_RERUN_FROM%, joint iterations %MPPP_JOINT_ITERATIONS% >> "%LOG%"
if /i "%MPPP_RERUN_FROM%"=="report" goto :report
if /i "%MPPP_RERUN_FROM%"=="blocks" goto :blocks
if /i "%MPPP_RERUN_FROM%"=="joints" goto :joints

echo %DATE% %TIME%  1. backlash states run (variant states) >> "%LOG%"
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites %MPPP_RERUN_SITES% --zcam --variant %MPPP_STATES_VARIANT% --jobs 2 --force %SETS%
if errorlevel 2 goto :failed
if exist "%ROOT%\mppp_stopped.txt" goto :stopped

:report
echo %DATE% %TIME%  2. states report and found list >> "%LOG%"
if "%MPPP_STATES_APPLY%"=="0" (
  python -m mppp --log "%ROOT%\camera_analysis\zcam_states_rerun_log.txt" zcam-states --root "%ROOT%" --variant %MPPP_STATES_VARIANT% --sites %MPPP_RERUN_SITES%
  echo %DATE% %TIME%  stopped after the report ^(MPPP_STATES_APPLY=0^): look at camera_analysis\zcam_states_^<date^>, apply, then set MPPP_RERUN_FROM=blocks >> "%LOG%"
  goto :end
)
python -m mppp --log "%ROOT%\camera_analysis\zcam_states_rerun_log.txt" zcam-states --root "%ROOT%" --variant %MPPP_STATES_VARIANT% --sites %MPPP_RERUN_SITES% --apply
if errorlevel 1 goto :failed
if exist "%ROOT%\mppp_stopped.txt" goto :stopped

:blocks
echo %DATE% %TIME%  3. the blocks again, every zoom of their zcam lists >> "%LOG%"
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --sites %MPPP_RERUN_SITES% --zcam --jobs 2 --force
if errorlevel 2 goto :failed
if exist "%ROOT%\mppp_stopped.txt" goto :stopped

:joints
echo %DATE% %TIME%  4. Mastcam-Z joints 34 48 63 79 110 mm, --iterations %MPPP_JOINT_ITERATIONS%, not promoted >> "%LOG%"
set "MPPP_ZOOMS=34 48 63 79 110"
set "MPPP_HAZ=none"
set "MPPP_NOPAUSE_END=1"
call "%MPPP_WIN%\zcam_joints.bat" --no-promote --iterations %MPPP_JOINT_ITERATIONS%
echo %DATE% %TIME%  done - camera_analysis\zcam*_joint_^<date^>\report.md >> "%LOG%"
goto :end

:stopped
echo %DATE% %TIME%  stopped by stop_mppp.bat - no later step >> "%LOG%"
goto :end
:failed
echo %DATE% %TIME%  a step FAILED or could not start (another batch running?) - see run_sites_log.txt >> "%LOG%"
pause
:end
exit /b 0
