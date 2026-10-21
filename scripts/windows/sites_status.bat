@echo off
REM MPPP: the run status of every WORK folder under ROOT (running / finished / failed / stopped).
setlocal
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
REM 20 Oct: MPPP_SCAPES after mppp_env.bat (it may come from mppp_scapes.txt), else D:\scapes when it exists
if not defined MPPP_SCAPES if exist "D:\scapes" set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & pause & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
python "%MPPP_HOME%\scripts\run_sites.py" --status --root "%ROOT%"
pause
