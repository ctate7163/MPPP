@echo off
REM ======================================================================================================
REM MPPP 0.91 - build_example_data.bat (Christian, 20 Oct 2026): the Mars 2020 example data set for the GitHub release
REM   Rockytop, sols 461-464 (the list in src\mppp\example.py, EXAMPLE_FILES): the 32 main Navcam images (the 360
REM   sequences NCAM02/03/13xxx), one front Hazcam pair on sols 461 and 464, the eight Mastcam-Z 34 mm frames of
REM   ZCAM03381 (sol 464), zipped in the archive layout
REM   (datadrive\<sol>\ids\rdr\ncam|fcam, zcam\) with example_manifest.json (sizes, SHA-256).
REM
REM Writes _transfer\mppp_example_m2020_rockytop.zip and .sha256. Then on GitHub: Releases - Draft a new release -
REM tag example-data-v1 - attach both files - Publish. Users get it with:  mppp example fetch
REM ======================================================================================================
setlocal
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_PDS set "MPPP_PDS=D:\data\m2020\datadrive"
for %%I in ("%MPPP_PDS%\..") do set "ZCAMDIR=%%~fI\zcam"
echo example data from %MPPP_PDS% and %ZCAMDIR%
python -m mppp example build --pds "%MPPP_PDS%" --zcam "%ZCAMDIR%" --out "%MPPP_HOME%\_transfer"
if errorlevel 1 ( echo FAILED & pause & exit /b 1 )
pause
exit /b 0
