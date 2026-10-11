@echo off
REM ======================================================================================================
REM MPPP v0p75 - watch D:\scapes\colmap\_jobs and run the jobs dropped there (scripts\run_jobs.py).
REM   A job is a text file <name>.job whose line is an MPPP script and its arguments, e.g.
REM     run_sites.py --root D:\scapes\colmap --sites chal_rocks_sid --zcam --set ZCAM_ZOOMS=[34] --force
REM   Only scripts in MPPP's scripts folder are run.  Output: <name>.log (live) and <name>.done (exit code).
REM   Claude's cloud session writes jobs there through the desktop link and reads the logs back.
REM   To end the watcher: create a file named "stop" in _jobs, or close this window (a running job goes on;
REM   stop_mppp.bat stops it).
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\run_jobs.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if /i "%~1"=="--here" (
  python "%MPPP_HOME%\scripts\run_jobs.py" --root "%ROOT%"
  exit /b %errorlevel%
)
start "MPPP jobs" /min cmd /c ""%~f0" --here"
echo MPPP job watcher started (minimised window "MPPP jobs"); jobs go to %ROOT%\_jobs
