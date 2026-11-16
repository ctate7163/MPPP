@echo off
REM RPP: the minimized window of align_here.bat (environment already set up by it).
title RPP align - %~1
python "%RPP_HOME%\scripts\align_scape.py" %*
if errorlevel 1 (
  echo.
  echo The alignment stopped with an error - see the newest runs\*\log.txt in %~1
  pause
)
