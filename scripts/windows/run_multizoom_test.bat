@echo off
REM ======================================================================================================
REM MPPP 0.92 - the multi-zoom test block: Taylorfjellet (sols 1601-1645, the widest LMST range) with every Mastcam-Z
REM zoom it has (34, 48, 63, 79, 110 mm) in one alignment, kept apart from the Z34 run in <WORK>\colmap_multizoom
REM (variant "multizoom").  34 and 48 mm frames need 10 tie points with Navcam; 63-110 mm frames 10 with Navcam or with
REM a wider zoom.  The zooms other than 34 mm start from their label CAHVOR models (no focus model yet).
REM Log: <WORK>\runs\<date-time>\log.txt; batch log %MPPP_SCAPES%\run_sites_log.txt.  Extra arguments go to run_sites.py.
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES if exist "D:\scapes" set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root & pause & exit /b 1)
if not defined MPPP_SITE set "MPPP_SITE=taylorfjellet"
echo %DATE% %TIME%  multi-zoom test block: %MPPP_SITE% (Mastcam-Z 34, 48, 63, 79, 110 mm; variant multizoom)
python "%MPPP_HOME%\scripts\run_sites.py" --root "%MPPP_SCAPES%" --sites %MPPP_SITE% --zcam --variant multizoom --set "ZCAM_ZOOMS=[34,48,63,79,110]" %*
if errorlevel 1 echo the multi-zoom block did not finish - see its log
echo %DATE% %TIME%  done
pause
