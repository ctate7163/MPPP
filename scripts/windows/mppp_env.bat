@echo off
REM MPPP (v0p43): make "python" the environment that runs the MPPP notebooks (pycolmap 4.2, pyceres, nbclient).
REM Called by the other .bat files; it changes the caller's environment (no setlocal here).
REM Optional environment variables: MPPP_HOME (the MPPP folder; default: two levels above this file),
REM MPPP_CONDA (the miniconda / anaconda folder), MPPP_ENV (the conda environment; default base).
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_ENV set "MPPP_ENV=base"
python -c "import pycolmap, pyceres, nbclient" >nul 2>nul && goto :ok
for %%C in ("%MPPP_CONDA%" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
  if exist "%%~C\Scripts\activate.bat" (
    call "%%~C\Scripts\activate.bat" %MPPP_ENV%
    goto :check
  )
)
:check
python -c "import pycolmap, pyceres, nbclient" >nul 2>nul && goto :ok
echo.
echo MPPP: no Python with pycolmap, pyceres and nbclient was found.
echo Open an Anaconda Prompt in the environment that runs the MPPP notebooks and start this .bat from there,
echo or set MPPP_CONDA / MPPP_ENV (e.g. set MPPP_ENV=base).
exit /b 1
:ok
exit /b 0
