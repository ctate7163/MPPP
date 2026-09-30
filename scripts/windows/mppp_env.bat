@echo off
REM MPPP: make "python" a Python that runs the MPPP notebooks (pycolmap, pyceres, nbclient, ipykernel).
REM Called by the other .bat files; it changes the caller's environment (no setlocal here).
REM Looked for, in this order (the first that passes check_env.py is used):
REM   1. MPPP_PYTHON, or the path in mppp_python.txt next to this file (one line: ...\python.exe).
REM      To find the notebooks' own python, run   import sys; print(sys.executable)   in a notebook.
REM   2. "python" as it is (e.g. when started from an Anaconda Prompt in the right environment).
REM   3. every conda environment listed in %USERPROFILE%\.conda\environments.txt
REM   4. MPPP_CONDA and the usual miniconda / anaconda / miniforge folders: MPPP_ENV (default base) first,
REM      then every environment in their envs\ folder.
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
if not defined MPPP_ENV set "MPPP_ENV=base"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "_MPPP_CHECK=%~dp0check_env.py"
if not defined MPPP_PYTHON if exist "%~dp0mppp_python.txt" for /f "usebackq eol=# delims=" %%L in ("%~dp0mppp_python.txt") do set "MPPP_PYTHON=%%~L"
set "_MPPP_VERBOSE="

:search
if defined MPPP_PYTHON for %%P in ("%MPPP_PYTHON%") do call :try_prefix "%%~dpP." && goto :ok
if defined _MPPP_VERBOSE (
  echo  - python on PATH:
  python "%_MPPP_CHECK%"
) else (
  python "%_MPPP_CHECK%" >nul 2>nul
)
if not errorlevel 1 goto :ok
if exist "%USERPROFILE%\.conda\environments.txt" for /f "usebackq eol=# delims=" %%E in ("%USERPROFILE%\.conda\environments.txt") do call :try_prefix "%%~E" && goto :ok
for %%C in ("%MPPP_CONDA%" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%USERPROFILE%\miniforge3" "%ProgramData%\miniforge3" "C:\miniconda3" "C:\anaconda3") do call :try_root "%%~C" && goto :ok
if defined _MPPP_VERBOSE goto :fail
set "_MPPP_VERBOSE=1"
echo.
echo MPPP: looking for a Python with pycolmap, pyceres, nbclient and ipykernel. Tried:
goto :search

:fail
set "_MPPP_VERBOSE="
echo.
echo MPPP: no Python with pycolmap, pyceres, nbclient and ipykernel was found (see above).
echo  - If the notebooks' Python is listed and only nbclient or ipykernel is missing, install them there:
echo      "...\python.exe" -m pip install nbclient ipykernel
echo  - Otherwise run   import sys; print(sys.executable)   in a notebook and write that path, one line,
echo    into %~dp0mppp_python.txt   (or set MPPP_PYTHON to it).
exit /b 1

:ok
set "_MPPP_VERBOSE="
exit /b 0

REM --- a conda installation: MPPP_ENV first, then the base environment, then every environment in envs\ ------
:try_root
if "%~1"=="" exit /b 1
if not exist "%~1\python.exe" exit /b 1
if /i not "%MPPP_ENV%"=="base" call :try_prefix "%~1\envs\%MPPP_ENV%" && exit /b 0
call :try_prefix "%~1" && exit /b 0
for /d %%E in ("%~1\envs\*") do call :try_prefix "%%~fE" && exit /b 0
exit /b 1

REM --- one environment (a folder with python.exe): put it first on PATH, as conda activate does, and check it ----
:try_prefix
if not exist "%~1\python.exe" exit /b 1
set "_MPPP_OLDPATH=%PATH%"
set "PATH=%~1;%~1\Library\mingw-w64\bin;%~1\Library\usr\bin;%~1\Library\bin;%~1\Scripts;%~1\bin;%PATH%"
if defined _MPPP_VERBOSE (
  echo  - %~1
  "%~1\python.exe" "%_MPPP_CHECK%"
) else (
  "%~1\python.exe" "%_MPPP_CHECK%" >nul 2>nul
)
if not errorlevel 1 (
  set "CONDA_PREFIX=%~1"
  set "_MPPP_OLDPATH="
  exit /b 0
)
set "PATH=%_MPPP_OLDPATH%"
set "_MPPP_OLDPATH="
exit /b 1
