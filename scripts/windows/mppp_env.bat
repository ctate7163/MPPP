@echo off
REM mppp_env.bat: the name of rpp_env.bat before 0.99 (the rename MPPP -> RPP).  Batch files outside git that still
REM call it get rpp_env.bat; it changes the caller's environment the same way (no setlocal here).
call "%~dp0rpp_env.bat" %*
exit /b %ERRORLEVEL%
