@echo off
REM ======================================================================================================
REM MPPP - stop every MPPP run on this computer: process_sites.bat, run_sites.bat, align_here.bat (their
REM minimised windows, notebook kernels and image workers). Notebooks open in Jupyter are not touched.
REM It lists the runs and asks before stopping them. A site stopped half-way is completed by the next run.
REM ======================================================================================================
setlocal
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
REM MPPP: this file's own folder if it is MPPP's scripts\windows, else MPPP_HOME, so this
REM .bat also works when copied elsewhere (e.g. into the scapes root).
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=%~dp0..\.."
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\stop_runs.py" --root "%ROOT%" %*
pause
