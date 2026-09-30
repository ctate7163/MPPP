@echo off
REM ======================================================================================================
REM MPPP - align the images already processed in THIS folder, with notebook 03's default settings.
REM It always runs the current MPPP in MPPP_HOME: the newest notebook 03 there and its default settings, so
REM this file never needs updating when MPPP changes. Process the images first with process_sites.bat.
REM
REM Copy this file into a site's WORK folder (the folder that holds processed\, e.g.
REM D:\scapes\colmap\south_arm_colmap_zcam34) and double-click it. The alignment runs in the background
REM (a minimised window; closing that window stops it). Nothing needs Jupyter.
REM
REM   progress   runs\<date-time>\log.txt   (every notebook cell and every [sfm] line as it happens)
REM   status     mppp_status.json           (double-click this file again while it runs: it shows the status)
REM   results    colmap\                    (sparse\cahv_ba, health\, error_input\, run_done.json)
REM
REM Reruns reuse the processed images, the features and the matches, so they only redo the alignment:
REM   - leave images out: list them in exclude_images.txt here (S032D1184 / sol:658 / seq:NCAM08111 / ZR0_0690_*)
REM   - other settings:   mppp_settings.json here, e.g. {"ATTITUDE_PRIOR_DEG": 1.0, "THERMAL_BINS_DEG": null}
REM   - side by side:     from a prompt, align_here.bat --variant NAME --set NAME=VALUE   (-> colmap_NAME\)
REM   - status only:      align_here.bat status
REM MPPP is looked for in MPPP_HOME (default D:\code\MPPP).
REM ======================================================================================================
setlocal
set "WORK=%~dp0"
set "WORK=%WORK:~0,-1%"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
if not exist "%MPPP_HOME%\scripts\align_scape.py" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
if not exist "%WORK%\processed\" (
  echo "%WORK%" has no processed\ folder: copy align_here.bat into a WORK folder next to processed\.
  pause
  exit /b 1
)
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if /i "%~1"=="status" (
  python "%MPPP_HOME%\scripts\align_scape.py" "%WORK%" --status
  pause
  exit /b 0
)
python "%MPPP_HOME%\scripts\align_scape.py" "%WORK%" --is-running >nul 2>nul
if not errorlevel 1 (
  echo An alignment of this folder is already running:
  python "%MPPP_HOME%\scripts\align_scape.py" "%WORK%" --status
  pause
  exit /b 0
)
start "MPPP align - %WORK%" /min cmd /c call "%MPPP_HOME%\scripts\windows\_run_align.bat" "%WORK%" %*
echo MPPP alignment of "%WORK%" started in a minimised window.
echo Progress: "%WORK%\runs\" (newest folder, log.txt). Double-click align_here.bat again for the status.
timeout /t 10
