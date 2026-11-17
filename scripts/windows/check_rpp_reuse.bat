@echo off
REM ======================================================================================================
REM RPP 0.99.0 - check_rpp_reuse.bat: does the rename change what would run?  A dry run of every site (Navcam and
REM Mastcam-Z) on the scapes root with this tree (v0p99) and with D:\code\MPPP (v0p98), side by side
REM (scripts\check_reuse.py).  Nothing is processed or aligned and no WORK folder is written, so it is safe while
REM freeze-3 runs.  Every finished block that v0p98 skips must be skipped by v0p99 too.
REM
REM   check_rpp_reuse.bat [--root D:\rover_blocks] [--old D:\code\MPPP]
REM ======================================================================================================
setlocal
set "RPP_NOPAUSE=1"
call "%~dp0rpp_env.bat" || ( pause & exit /b 1 )
python "%~dp0..\check_reuse.py" %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" ( echo OK: the same blocks are finished for v0p99 and v0p98 ) else ( echo DIFFERENT - send Claude the list above and %TEMP%\rpp_check_reuse.txt )
pause
exit /b %RC%
