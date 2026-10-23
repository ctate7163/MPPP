@echo off
REM run_weekend.bat's loop: run_weekend.py again after a crash (at most 20 times), never after a finish or a stop.
setlocal enabledelayedexpansion
set "ROOT=%~1"
shift
set "ARGS="
:args
if "%~1"=="" goto run
set "ARGS=!ARGS! %1"
shift
goto args
:run
set /a N=0
:again
python "%MPPP_HOME%\studies\calibration\run_weekend.py" --root "%ROOT%" !ARGS!
set "RC=!errorlevel!"
if "!RC!"=="0" goto done
python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('state') == 'stopped' else 1)" "%ROOT%\mppp_pipeline\weekend_status.json" && goto stopped
set /a N+=1
if !N! gtr 20 goto gaveup
echo %DATE% %TIME%  run_weekend.py exit !RC!: again in 2 min (restart !N! of 20)>> "%ROOT%\mppp_pipeline\weekend_log.txt"
timeout /t 120 /nobreak >nul
goto again
:stopped
echo %DATE% %TIME%  run_weekend: stopped (stop_mppp)>> "%ROOT%\mppp_pipeline\weekend_log.txt"
exit /b 0
:gaveup
echo %DATE% %TIME%  run_weekend: 20 restarts - given up>> "%ROOT%\mppp_pipeline\weekend_log.txt"
pause
exit /b 1
:done
exit /b 0
