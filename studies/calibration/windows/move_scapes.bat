@echo off
REM MPPP v0p85 - move the block folders from D:\scapes\colmap to D:\scapes and drop "_colmap" from their names
REM (scripts\move_scapes.py). Shows what it would move first, then asks. Renames only: nothing is copied or overwritten.
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
python "%MPPP_HOME%\scripts\move_scapes.py"
choice /m "Move them now"
if errorlevel 2 exit /b 0
python "%MPPP_HOME%\scripts\move_scapes.py" --apply
pause
