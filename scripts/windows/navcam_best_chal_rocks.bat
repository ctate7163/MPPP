@echo off
REM ======================================================================================================
REM MPPP v0p80 - promote the best Navcam models available now, then align Chal Rocks with the v0p80 block recipe
REM and diagnose it.  Double-click it (after adopt_claude.bat has taken in the v0p80 bundle).
REM
REM   1. the 4 Oct consensus with the residual rig yaw(T) moved into NR cx(T) (-0.054 -> -0.071 px/degC), promoted
REM      (cmods\history keeps the old files; a local commit; _transfer\promoted.json)
REM   2. Chal Rocks Sid into  <WORK>\colmap_v0p80  (the images already processed there; the 5 Oct run stays in colmap\)
REM   3. the diagnosis against the earlier run:  <WORK>\colmap_v0p80\diagnose\diagnose.md / .png
REM   4. then Chal Rocks Large the same way
REM   All diagnoses are copied to  %ROOT%\error_analysis\chal_rocks_v0p80\
REM
REM   log  %ROOT%\mppp_pipeline\chal_rocks_v0p80_log.txt     (each alignment: <WORK>\runs\)
REM   Options are passed on, e.g.  navcam_best_chal_rocks.bat --no-promote   or   --source pds
REM Runs in a minimised window; stop_mppp.bat stops it.  Keep the computer awake until it has finished (some hours).
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes"
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\navcam_best_chal_rocks.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
start "MPPP v0p80 - best Navcam + Chal Rocks" /min cmd /c python "%MPPP_HOME%\scripts\navcam_best_chal_rocks.py" --root "%ROOT%" --large %* ^|^| pause
echo Started in a minimised window. Log: "%ROOT%\mppp_pipeline\chal_rocks_v0p80_log.txt"
echo Results: "%ROOT%\error_analysis\chal_rocks_v0p80\"
timeout /t 15
