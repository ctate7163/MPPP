@echo off
REM MPPP: the run status of every WORK folder under ROOT (running / finished / failed / stopped).
setlocal
set "ROOT=D:\scapes\colmap"
call "%~dp0mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\run_sites.py" --status --root "%ROOT%"
pause
