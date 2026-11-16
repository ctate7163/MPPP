@echo off
REM RPP: make "python" a Python that runs the RPP pipeline (rpp, pycolmap, pyceres).
REM Called by the other .bat files; it changes the caller's environment (no setlocal here).
REM Looked for, in this order (the first that passes check_env.py is used):
REM   1. RPP_PYTHON, or the path in rpp_python.txt next to this file (one line: ...\python.exe).
REM      To find the notebooks' own python, run   import sys; print(sys.executable)   in a notebook.
REM   2. "python" as it is (e.g. when started from an Anaconda Prompt in the right environment).
REM   3. every conda environment listed in %USERPROFILE%\.conda\environments.txt
REM   4. RPP_CONDA and the usual miniconda / anaconda / miniforge folders: RPP_ENV (default base) first,
REM      then every environment in their envs\ folder.
REM 0.99.0 (the rename MPPP -> RPP): each MPPP_X set in the environment (setx, or a batch file outside git) is read as
REM RPP_X unless RPP_X is set - except MPPP_HOME: the tree that runs is RPP_HOME.  "rpp_env.bat --legacy-only" does only
REM this (for a batch file that reads a variable before it calls this file).  The one-line files below may still be
REM called mppp_*.txt (a tree set up before 0.99).
for /f "tokens=1 delims==" %%A in ('set MPPP_ 2^>nul') do call :_rpp_legacy %%A
if /i "%~1"=="--legacy-only" exit /b 0
if not defined RPP_HOME for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
if not defined RPP_ENV set "RPP_ENV=base"
set "KMP_DUPLICATE_LIB_OK=TRUE"
REM 0.98.26: the Intel Fortran runtime of numpy/scipy catches Ctrl-C and ends the process with exit code 0 (16 Nov: the
REM pre-freeze blocks killed at 11:20, "blocks done (exit 0)", the joints run on).  Off: a Ctrl-C is a Python
REM KeyboardInterrupt and the exit code says the run did not finish.
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
set "_RPP_CHECK=%~dp0..\check_env.py"
for %%N in (rpp mppp) do if not defined RPP_PYTHON if exist "%~dp0%%N_python.txt" for /f "usebackq eol=# delims=" %%L in ("%~dp0%%N_python.txt") do set "RPP_PYTHON=%%~L"
REM 0.90: the scapes root and the PDS folder may also come from one-line files next to this script (git-ignored),
REM when RPP_SCAPES / RPP_PDS are not set in the environment (setx RPP_SCAPES D:\rover_blocks sets them for good)
for %%N in (rpp mppp) do if not defined RPP_SCAPES if exist "%~dp0%%N_scapes.txt" for /f "usebackq eol=# delims=" %%L in ("%~dp0%%N_scapes.txt") do set "RPP_SCAPES=%%~L"
for %%N in (rpp mppp) do if not defined RPP_PDS if exist "%~dp0%%N_pds.txt" for /f "usebackq eol=# delims=" %%L in ("%~dp0%%N_pds.txt") do set "RPP_PDS=%%~L"
REM 0.95.0: the camera models in use live outside git (rpp cmods init D:\scapes\cmods writes rpp_cmods.txt)
for %%N in (rpp mppp) do if not defined RPP_CMODS if exist "%~dp0%%N_cmods.txt" for /f "usebackq eol=# delims=" %%L in ("%~dp0%%N_cmods.txt") do set "RPP_CMODS=%%~L"
set "_RPP_VERBOSE="

:search
if defined RPP_PYTHON for %%P in ("%RPP_PYTHON%") do call :try_prefix "%%~dpP." && goto :ok
if defined _RPP_VERBOSE (
  echo  - python on PATH:
  python "%_RPP_CHECK%"
) else (
  python "%_RPP_CHECK%" >nul 2>nul
)
if not errorlevel 1 goto :ok
if exist "%USERPROFILE%\.conda\environments.txt" for /f "usebackq eol=# delims=" %%E in ("%USERPROFILE%\.conda\environments.txt") do call :try_prefix "%%~E" && goto :ok
for %%C in ("%RPP_CONDA%" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%USERPROFILE%\miniforge3" "%ProgramData%\miniforge3" "C:\miniconda3" "C:\anaconda3") do call :try_root "%%~C" && goto :ok
if defined _RPP_VERBOSE goto :fail
set "_RPP_VERBOSE=1"
echo.
echo RPP: looking for a Python with rpp, pycolmap and pyceres. Tried:
goto :search

:fail
set "_RPP_VERBOSE="
echo.
echo RPP: no Python with rpp, pycolmap and pyceres was found (see above).
echo  - If the notebooks' Python is listed and only rpp is missing, install it there:
echo      "...\python.exe" -m pip install -e <RPP folder>[sfm]
echo  - Otherwise run   import sys; print(sys.executable)   in a notebook and write that path, one line,
echo    into %~dp0rpp_python.txt   (or set RPP_PYTHON to it).
exit /b 1

:ok
set "_RPP_VERBOSE="
REM 0.96.0: this tree's own packages first (rpp and rpp_studies under src\) - an editable install made before the
REM split does not know the new top-level package rpp_studies; the freeze checkout sets its own PYTHONPATH before.
if defined PYTHONPATH ( set "PYTHONPATH=%RPP_HOME%\src;%PYTHONPATH%" ) else ( set "PYTHONPATH=%RPP_HOME%\src" )
exit /b 0

REM --- a conda installation: RPP_ENV first, then the base environment, then every environment in envs\ ------
:try_root
if "%~1"=="" exit /b 1
if not exist "%~1\python.exe" exit /b 1
if /i not "%RPP_ENV%"=="base" call :try_prefix "%~1\envs\%RPP_ENV%" && exit /b 0
call :try_prefix "%~1" && exit /b 0
for /d %%E in ("%~1\envs\*") do call :try_prefix "%%~fE" && exit /b 0
exit /b 1

REM --- one environment (a folder with python.exe): put it first on PATH, as conda activate does, and check it ----
:try_prefix
if not exist "%~1\python.exe" exit /b 1
set "_RPP_OLDPATH=%PATH%"
set "PATH=%~1;%~1\Library\mingw-w64\bin;%~1\Library\usr\bin;%~1\Library\bin;%~1\Scripts;%~1\bin;%PATH%"
if defined _RPP_VERBOSE (
  echo  - %~1
  "%~1\python.exe" "%_RPP_CHECK%"
) else (
  "%~1\python.exe" "%_RPP_CHECK%" >nul 2>nul
)
if not errorlevel 1 (
  set "CONDA_PREFIX=%~1"
  set "_RPP_OLDPATH="
  exit /b 0
)
set "PATH=%_RPP_OLDPATH%"
set "_RPP_OLDPATH="
exit /b 1

REM --- 0.99.0: MPPP_X -> RPP_X when RPP_X is not set (not MPPP_HOME) -------------------------------------------------
:_rpp_legacy
if /i "%~1"=="MPPP_HOME" exit /b 0
set "_RPP_N=%~1"
set "_RPP_N=RPP_%_RPP_N:~5%"
if defined %_RPP_N% ( set "_RPP_N=" & exit /b 0 )
call set "%_RPP_N%=%%%~1%%"
set "_RPP_N="
exit /b 0
