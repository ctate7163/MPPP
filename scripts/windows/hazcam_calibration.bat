@echo off
REM ======================================================================================================
REM MPPP 0.93 - hazcam_calibration.bat: the Hazcam fit (mppp calib hazcam) over the finished blocks of
REM hazcam_consensus aligned by run_hazcam.bat (<WORK>\colmap_hazcam), the candidate files of every eye and pair that
REM passes its checks, and their promotion (arguments go to "mppp calib hazcam", e.g. --no-promote, --group,
REM --variant "" for the blocks' colmap\).  Writes %MPPP_SCAPES%\camera_analysis\hazcam_fit_<date>[_N]\
REM (report.md, check.json, hazcam_fit.json, M2020_FL.json ... M2020_F_rig.json, candidate\).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_SCAPES%\camera_analysis\hazcam_fit_%TODAY%"
set "N=1"
:next
if exist "%OUT%\hazcam_fit.json" ( set /a N+=1 & set "OUT=%MPPP_SCAPES%\camera_analysis\hazcam_fit_%TODAY%_!N!" & goto :next )
echo Hazcam fit: hazcam_consensus under %MPPP_SCAPES% -^> %OUT%
python -m mppp --log "%OUT%_log.txt" calib hazcam --group hazcam_consensus --variant hazcam --root "%MPPP_SCAPES%" --out "%OUT%" --promote-if-ok %*
if errorlevel 1 ( echo Hazcam fit FAILED - see %OUT%_log.txt & pause & exit /b 1 )
python -m mppp cmods list
echo done: %OUT%
pause
exit /b 0
