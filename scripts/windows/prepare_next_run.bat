@echo off
REM MPPP v0p73 - before the next batch: check that the right camera models are in use (the v0p72 consensus of
REM 2 Oct 2026 in the v0p73 NL-reference form), promote it if not, check the form, and list what run_sites will run.
REM Double-click it. Add --check-only to change nothing.
setlocal
if not defined MPPP_HOME if exist "%~dp0mppp_env.bat" if exist "%~dp0..\prepare_next_run.py" for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_HOME set "MPPP_HOME=D:\code\MPPP"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
if not exist "%MPPP_WIN%\mppp_env.bat" (
  echo MPPP not found in "%MPPP_HOME%": set MPPP_HOME to the MPPP folder, e.g.  set MPPP_HOME=D:\code\MPPP
  pause
  exit /b 1
)
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\prepare_next_run.py" %*
pause
