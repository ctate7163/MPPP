@echo off
REM ======================================================================================================
REM RPP - stop every RPP run on this computer: process_sites.bat, run_sites.bat, align_here.bat (their
REM minimized windows, notebook kernels and image workers). Notebooks open in Jupyter are not touched.
REM It lists the runs and asks before stopping them. A site stopped half-way is completed by the next run.
REM ======================================================================================================
setlocal
REM RPP: this file's own folder if it is RPP's scripts\windows, else RPP_HOME, so this
REM .bat also works when copied elsewhere (e.g. into the scapes root).
if not defined RPP_HOME if exist "%~dp0rpp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
if not defined RPP_HOME set "RPP_HOME=%~dp0..\.."
set "RPP_WIN=%RPP_HOME%\scripts\windows"
if not exist "%RPP_WIN%\rpp_env.bat" (
  echo RPP not found in "%RPP_HOME%": set RPP_HOME to the RPP folder
  pause
  exit /b 1
)
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
REM 20 Oct: RPP_SCAPES after rpp_env.bat (it may come from rpp_scapes.txt), else D:\rover_blocks when it exists
if not defined RPP_SCAPES if exist "D:\rover_blocks" set "RPP_SCAPES=D:\rover_blocks"
if not defined RPP_SCAPES (echo set RPP_SCAPES to the scapes root, e.g. setx RPP_SCAPES D:\rover_blocks & pause & exit /b 1)
set "ROOT=%RPP_SCAPES%"
python "%RPP_HOME%\scripts\stop_runs.py" --root "%ROOT%" %*
pause
