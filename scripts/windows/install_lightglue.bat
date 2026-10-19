@echo off
REM ======================================================================================================
REM MPPP v0p86 - install_lightglue.bat (Christian, 19 Oct 2026)
REM A GPU environment with COLMAP / pycolmap 4.2.1 built WITH ONNX support, so that matching can use SIFT + LightGlue.
REM
REM   install_lightglue.bat          install (or update) and test the environment mppp_gpu_lg; if the test passes and
REM                                  no MPPP run is going, MPPP uses it from the next block on
REM   install_lightglue.bat --off    go back to mppp_gpu (SIFT brute force); mppp_gpu_lg is kept
REM
REM Why a new environment: the pip pycolmap wheels (Windows) have no ONNX and no CUDA, and the conda-forge colmap
REM builds before build number 1 have no ONNX either.  The conda-forge colmap 4.2.1 builds "cuda_129..._1" and
REM "cuda_134..._1" depend on onnxruntime-cpp; this file clones mppp_gpu (it stays as it is, for going back) into
REM mppp_gpu_lg and installs that colmap, the CUDA pycolmap 4.2.1 and the CUDA onnxruntime-cpp into the clone, with
REM the same CUDA version as mppp_gpu.  Then it tests LightGlue on synthetic images (lightglue_check.py).
REM MPPP finds the environment through %LOCALAPPDATA%\mppp\gpu_python.txt (MPPP_GPU_PY overrides it).
REM Needs: miniconda / miniforge / anaconda, internet, ~3 GB.  About 10-20 min.  Log: _transfer\install_lightglue_log.txt
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "OLD=mppp_gpu"
set "NEW=mppp_gpu_lg"
set "PTR_DIR=%LOCALAPPDATA%\mppp"
set "PTR=%PTR_DIR%\gpu_python.txt"
set "LOG=%MPPP_HOME%\_transfer\install_lightglue_log.txt"
if not exist "%MPPP_HOME%\_transfer" mkdir "%MPPP_HOME%\_transfer"

if /i "%~1"=="--off" (
  if exist "%PTR%" del "%PTR%"
  echo MPPP uses %OLD% again ^(SIFT brute force^). %NEW% is kept; install_lightglue.bat switches back.
  timeout /t 15
  exit /b 0
)

echo ==== install_lightglue %DATE% %TIME% > "%LOG%"
call :say "MPPP: GPU environment with ONNX support for SIFT + LightGlue"

REM --- conda: the installation that holds mppp_gpu, else the first one found ---------------------------------
set "CONDA="
for %%C in ("%MPPP_CONDA%" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\miniforge3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
  if not defined CONDA if exist "%%~C\envs\%OLD%\python.exe" if exist "%%~C\condabin\conda.bat" set "CONDA=%%~C"
)
for %%C in ("%MPPP_CONDA%" "%LOCALAPPDATA%\miniconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\miniforge3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\anaconda3" "%USERPROFILE%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
  if not defined CONDA if exist "%%~C\condabin\conda.bat" set "CONDA=%%~C"
)
if not defined CONDA (
  call :say "FAILED: no conda installation found (set MPPP_CONDA to its folder, e.g. C:\Users\you\miniconda3)"
  goto :end_fail
)
set "CONDABAT=%CONDA%\condabin\conda.bat"
set "ENVDIR=%CONDA%\envs\%NEW%"
call :say "conda: %CONDA%"

REM --- CUDA version of mppp_gpu's colmap (cuda_129... -> 129); 129 if it cannot be read ----------------------
set "CU=129"
if exist "%CONDA%\envs\%OLD%\python.exe" (
  for /f "tokens=1-3" %%a in ('call "%CONDABAT%" list -n %OLD% 2^>nul ^| findstr /b /c:"colmap "') do (
    set "B=%%c"
    if /i "!B:~0,5!"=="cuda_" set "CU=!B:~5,3!"
  )
)
call :say "CUDA build: %CU% (as %OLD%)"

REM --- create: a clone of mppp_gpu (all its packages, incl. the pip ones), else a new environment --------------
if exist "%ENVDIR%\python.exe" (
  call :say "%NEW% exists: updating it"
) else if exist "%CONDA%\envs\%OLD%\python.exe" (
  call :say "cloning %OLD% into %NEW% (a few minutes) ..."
  call "%CONDABAT%" create -y -n %NEW% --clone %OLD% >> "%LOG%" 2>&1
  if errorlevel 1 (call :say "FAILED: conda create --clone, see the log" & goto :end_fail)
) else (
  call :say "no %OLD%: creating %NEW% from scratch (a few minutes) ..."
  call "%CONDABAT%" create -y -n %NEW% -c conda-forge --override-channels python=3.12 numpy scipy opencv pvl tifffile pillow pip >> "%LOG%" 2>&1
  if errorlevel 1 (call :say "FAILED: conda create, see the log" & goto :end_fail)
  "%ENVDIR%\python.exe" -m pip install "pdr>=1.4" colour-demosaicing piexif >> "%LOG%" 2>&1
)

REM --- install colmap 4.2.1 with ONNX (build number 1+), the CUDA pycolmap 4.2.1, CUDA onnxruntime --------------
call :say "installing colmap 4.2.1 (ONNX), pycolmap 4.2.1 (CUDA), onnxruntime-cpp (CUDA %CU%) - 5-15 min ..."
call "%CONDABAT%" install -y -n %NEW% -c conda-forge --override-channels "colmap=4.2.1=cuda_%CU%*_1" "pycolmap=4.2.1=*cuda*" "onnxruntime-cpp=*=*cuda%CU%" pillow numpy >> "%LOG%" 2>&1
if errorlevel 1 (
  call :say "  that combination did not solve; trying any build of colmap 4.2.1 CUDA %CU% and pycolmap 4.2 ..."
  call "%CONDABAT%" install -y -n %NEW% -c conda-forge --override-channels "colmap=4.2.1=cuda_%CU%*" "pycolmap=4.2.*=*cuda*" "onnxruntime-cpp=*=*cuda%CU%" pillow numpy >> "%LOG%" 2>&1
  if errorlevel 1 (call :say "FAILED: conda install, see the log (the solver's message is at the end)" & goto :end_fail)
)
echo installed: >> "%LOG%"
call "%CONDABAT%" list -n %NEW% 2>nul | findstr /r /c:"^colmap " /c:"^pycolmap " /c:"^onnxruntime" /c:"^cudnn " /c:"^python " > "%TEMP%\mppp_lg_list.txt"
type "%TEMP%\mppp_lg_list.txt"
type "%TEMP%\mppp_lg_list.txt" >> "%LOG%"
findstr /c:"onnxruntime" "%TEMP%\mppp_lg_list.txt" >nul || (call :say "FAILED: onnxruntime-cpp is not in %NEW%" & goto :end_fail)

REM --- test, in the new environment as conda activate would set it up -----------------------------------------
call :say "testing SIFT + LightGlue in %NEW% ..."
set "PATH=%ENVDIR%;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\bin;%ENVDIR%\Scripts;%ENVDIR%\bin;%PATH%"
set "CONDA_PREFIX=%ENVDIR%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONPATH="
"%ENVDIR%\python.exe" "%~dp0lightglue_check.py" "%MPPP_HOME%\src" > "%TEMP%\mppp_lg_check.txt" 2>&1
set "RC=%ERRORLEVEL%"
type "%TEMP%\mppp_lg_check.txt"
type "%TEMP%\mppp_lg_check.txt" >> "%LOG%"
if "%RC%"=="2" (
  call :say "NOT switched: LightGlue runs in %NEW% but too slowly for MPPP's blocks (see above). MPPP keeps %OLD%."
  goto :end_fail
)
if not "%RC%"=="0" (
  call :say "NOT switched: the LightGlue test failed (see above and the log). MPPP keeps %OLD%."
  goto :end_fail
)

REM --- switch, unless an MPPP run is going (it would change environment between two blocks) -------------------
set "BUSY=0"
for /f %%n in ('powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_sites|process_sites|run_scapes|align_scape|mppp\.sfm\.gpu' }).Count" 2^>nul') do set "BUSY=%%n"
if not "%BUSY%"=="0" (
  call :say "PASSED, but an MPPP run is going (%BUSY% Python processes): NOT switched yet. Run install_lightglue.bat again when it has finished - the environment is kept, so that takes a few minutes."
  goto :end_fail
)
if not exist "%PTR_DIR%" mkdir "%PTR_DIR%"
> "%PTR%" echo %ENVDIR%\python.exe
>> "%PTR%" echo # written by install_lightglue.bat %DATE% %TIME%; delete this file (or install_lightglue.bat --off) to go back to %OLD%
call :say "PASSED and switched: MPPP's GPU steps now run in %NEW% (%PTR%). install_lightglue.bat --off goes back."
echo.
pause
exit /b 0

:end_fail
echo.
echo Log: %LOG%
pause
exit /b 1

:say
echo %~1
>> "%LOG%" echo %~1
exit /b 0
