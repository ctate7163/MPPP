@echo off
REM ======================================================================================================
REM MPPP - stop every MPPP run on this computer: process_sites.bat, run_sites.bat, align_here.bat (their
REM minimised windows, notebook kernels and image workers). Notebooks open in Jupyter are not touched.
REM It lists the runs and asks before stopping them. A site stopped half-way is completed by the next run.
REM ======================================================================================================
setlocal
set "ROOT=D:\scapes\colmap"
REM MPPP: this file's own folder if it is MPPP's scripts\windows, else MPPP_HOME (default D:\code\MPPP), so this
REM .bat also works when copied elsewhere (e.g. into D:\scapes\colmap).
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\process_sites.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\stop_runs.py" --root "%ROOT%" %*
pause
