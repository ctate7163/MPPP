@echo off
REM ======================================================================================================
REM MPPP - m20_dem_env.bat: sets M20_DEMS and M20_ORTHOS (os.pathsep lists) for the site maps from the files under
REM DEMROOT (default %MPPP_SCAPES%\dem, else D:\scapes\dem). Called by make_maps.bat and make_figures.bat:
REM   call m20_dem_env.bat          (the caller has setlocal EnableDelayedExpansion)
REM The files under DEMROOT win over an M20_DEMS / M20_ORTHOS left in the environment (a placeholder broke the
REM 20 Oct run). Orthomosaics: the Mars 2020 TRN mosaic (m2020_trn, the whole traverse, one radiometry) first,
REM then the MSR mosaics (msr_trn).
REM ======================================================================================================
set "DEMROOT=D:\scapes\dem"
if defined MPPP_SCAPES if exist "%MPPP_SCAPES%\dem" set "DEMROOT=%MPPP_SCAPES%\dem"
set "_HAVE_D="
set "_HAVE_O="
if exist "%DEMROOT%\msr_trn\*DTM*.tif" set "_HAVE_D=1"
if exist "%DEMROOT%\m2020_trn\*DTM*.tif" set "_HAVE_D=1"
if exist "%DEMROOT%\msr_trn\*rtho*.tif" set "_HAVE_O=1"
if exist "%DEMROOT%\m2020_trn\*rtho*.tif" set "_HAVE_O=1"
if defined _HAVE_D set "M20_DEMS="
if defined _HAVE_O set "M20_ORTHOS="
if not defined M20_DEMS (
  for %%F in ("%DEMROOT%\msr_trn\*DTM*.tif" "%DEMROOT%\m2020_trn\*DTM*.tif") do (
    if defined M20_DEMS ( set "M20_DEMS=!M20_DEMS!;%%~fF" ) else ( set "M20_DEMS=%%~fF" )
  )
)
if not defined M20_ORTHOS (
  for %%F in ("%DEMROOT%\m2020_trn\*rtho*.tif" "%DEMROOT%\msr_trn\*rtho*.tif") do (
    if defined M20_ORTHOS ( set "M20_ORTHOS=!M20_ORTHOS!;%%~fF" ) else ( set "M20_ORTHOS=%%~fF" )
  )
)
set "_HAVE_D="
set "_HAVE_O="
exit /b 0
