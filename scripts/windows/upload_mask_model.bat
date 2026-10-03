@echo off
REM ======================================================================================================
REM MPPP - upload the default mask model (src\mppp\data\models.json "default", now mppp_mask_v3) to Hugging Face
REM (v0p72.1), so that processing on any computer downloads the released file.
REM
REM   1. finds the notebooks' Python (mppp_env.bat) and installs huggingface_hub there if it is missing (asks)
REM   2. logs in to Hugging Face if this computer is not logged in yet: paste a token with WRITE access
REM      (huggingface.co -> Settings -> Access Tokens -> New token, type "Write")
REM   3. python -m mppp.mask.hub upload: exports checkpoints\<source checkpoint>.pt to the .safetensors if it is
REM      not there yet, refuses a file whose SHA-256 differs from the registry, creates the model repository
REM      (registry "hf_repo", ctate7163/mppp-mask) and uploads the file and docs\hf_model_card.md as its README
REM   4. python -m mppp.mask.hub verify: downloads it again from every registry URL and checks the SHA-256
REM
REM Another model: upload_mask_model.bat mppp_mask_v2
REM ======================================================================================================
setlocal
set "MODEL=%~1"
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%~dp0mppp_env.bat" || ( pause & exit /b 1 )
cd /d "%MPPP_HOME%" || ( pause & exit /b 1 )
set "PYTHONPATH=%MPPP_HOME%\src;%PYTHONPATH%"
if not defined MODEL for /f "usebackq delims=" %%M in (`python -c "from mppp.mask.hub import default_model_name; print(default_model_name())"`) do set "MODEL=%%M"
if not defined MODEL ( echo could not read the default model from src\mppp\data\models.json & pause & exit /b 1 )
echo Mask model: %MODEL%
python -m mppp.mask.hub list

python -c "import huggingface_hub" 2>nul
if errorlevel 1 (
  echo huggingface_hub is not installed in this Python.
  choice /C YN /M "Install it now (python -m pip install huggingface_hub)"
  if errorlevel 2 ( echo nothing uploaded & pause & exit /b 1 )
  python -m pip install huggingface_hub || ( pause & exit /b 1 )
)

REM logged in? (a token from an earlier login, or HF_TOKEN)
python -c "from huggingface_hub import whoami; print('Hugging Face user:', whoami()['name'])" 2>nul
if errorlevel 1 (
  echo Not logged in to Hugging Face. Paste a token with WRITE access when asked
  echo ^(huggingface.co -^> Settings -^> Access Tokens -^> New token, type Write^).
  python -c "from huggingface_hub import login; login(add_to_git_credential=False)" || ( pause & exit /b 1 )
)
REM the repository owner must be the logged-in user (or an organisation of theirs)
python -c "from huggingface_hub import whoami; from mppp.mask.hub import hf_repo; u=whoami(); r=hf_repo('%MODEL%'); o=r.split('/')[0]; ok=o in [u['name']]+[x['name'] for x in u.get('orgs', [])]; print('repository', r, '- you are', u['name']); raise SystemExit(0 if ok else 1)"
if errorlevel 1 (
  echo The registry repository is owned by another Hugging Face account than the one logged in.
  echo Log in as its owner ^(hf auth logout, then run this again^), or tell Claude to change "hf_repo" and the URL in
  echo src\mppp\data\models.json. Nothing uploaded.
  pause & exit /b 1
)

echo.
echo Uploading %MODEL% ...
python -m mppp.mask.hub upload --name %MODEL% || ( echo UPLOAD FAILED - see above & pause & exit /b 1 )
echo.
echo Checking the download ...
python -m mppp.mask.hub verify --name %MODEL%
if errorlevel 1 ( echo VERIFY FAILED - the file on Hugging Face is not the registry's & pause & exit /b 1 )
echo.
echo Done: %MODEL% is on Hugging Face and downloads with the registry's SHA-256.
pause
