@echo off
REM ======================================================================================================
REM MPPP 0.95.1 - trial_joints_0p95.bat (Christian, 28 Oct 2026): the 0.95.0 joints over the blocks already aligned
REM (0.94.x), NOTHING PROMOTED - a trial of the joint code before the freeze reruns, which saves days of alignment.
REM The 0.94 blocks have focus bins (not sequence bins), so the Mastcam-Z joint runs without sequence offsets; the
REM rest of 0.95.0 (one slope per eye, the paired low-state ZR centre, the fb hold, the worst frames, the thin-bin
REM cut; the Navcam stereo-difference blocks, the exact-tie skip, the pass rule, the stereo law check) is exercised.
REM
REM   trial_joints_0p95.bat                 everything below, one after another (~5-7 h)
REM   set MPPP_TRIAL=zcam                  before starting: only these parts, in this order (0.95.1: a list, e.g.
REM                                        "nz lawcheck hazcam zcam" to resume); parts: navcam tiefslope nz lawcheck
REM                                        hazcam zcam; all = every part (the default)
REM   set MPPP_ZOOMS=34                    only these Mastcam-Z zooms (0.95.1 default 34 48 63 79 110)
REM   trial_joints_0p95.bat --include-failed   arguments go to every "mppp calib"
REM
REM Writes %MPPP_SCAPES%\camera_analysis\trial_0p95_<date>\ :
REM   navcal_tied\          the Navcam joint (candidate navcal_tied\navcam_joint, promotion_check.json, report)
REM   navcal_tied_tiefslope\  the same with NR's f slope tied to NL's (--tie-f-slope): the comparison run
REM   nz34_joint\           the Navcam + Mastcam-Z 34 mm combined joint (~1 h; set MPPP_NZ=none to skip it); 29 Oct:
REM                         its Navcam law as model files in nz34_joint\navcam_law (the y-scale question)
REM   lawcheck\             the law in use, the trial candidate and the combined law on the same Navcam blocks
REM   hazcam_fit\           the Hazcam fit
REM   zcam<zoom>_joint\     the Mastcam-Z joints (34 over zcam34_joint + zcam_consensus; the others over both too);
REM                         0.95.1: the rig translation refined (its z per axis in the report), ZR's early-mission
REM                         principal-point offset, the eye-slope tests (the translation held, one slope for both eyes)
REM Running blocks are not waited for (--wait-h 0): the finished ones are taken as they are.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_ZOOMS set "MPPP_ZOOMS=34 48 63 79 110"
if not defined MPPP_TRIAL set "MPPP_TRIAL=all"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_SCAPES%\camera_analysis\trial_0p95_%TODAY%"
if not exist "%OUT%" mkdir "%OUT%"
echo %DATE% %TIME%  MPPP trial of the 0.95.0 joints -^> %OUT%
python -m mppp version
if errorlevel 1 ( echo mppp does not run - sync and install first ^(claude_sync.bat^) & pause & exit /b 1 )
python -m mppp cmods status
echo.
echo The camera models in use must be the ones the blocks were aligned with ^(a block aligned with other Navcam
echo models is left out of the Mastcam-Z joints^). Nothing is promoted by this trial.
echo.

set "PARTS=%MPPP_TRIAL%"
if /i "%PARTS%"=="all" set "PARTS=navcam tiefslope nz lawcheck hazcam zcam"
for %%P in (%PARTS%) do (
  set "KNOWN="
  for %%K in (navcam tiefslope nz lawcheck hazcam zcam) do if /i "%%P"=="%%K" set "KNOWN=1"
  if not defined KNOWN ( echo unknown part %%P in MPPP_TRIAL=%MPPP_TRIAL% - parts: navcam tiefslope nz lawcheck hazcam zcam & pause & exit /b 2 )
)
echo parts: %PARTS%; zooms: %MPPP_ZOOMS%
for %%P in (%PARTS%) do call :part_%%P %*
goto :end

:part_navcam
echo %DATE% %TIME%  Navcam joint over ecam_consensus -^> %OUT%\navcal_tied
python -m mppp --log "%OUT%\navcal_tied_log.txt" calib navcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\navcal_tied" --knot 250 --points 8000 --no-promote --wait-h 0 %*
if errorlevel 1 echo Navcam joint: FAILED - see %OUT%\navcal_tied_log.txt
exit /b 0

:part_tiefslope
echo.
echo %DATE% %TIME%  Navcam joint, NR's f slope tied to NL's ^(the comparison run^) -^> %OUT%\navcal_tied_tiefslope
python -m mppp --log "%OUT%\navcal_tied_tiefslope_log.txt" calib navcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\navcal_tied_tiefslope" --knot 250 --points 8000 --tie-f-slope --no-promote --wait-h 0 %*
if errorlevel 1 echo Navcam joint ^(tie f slope^): FAILED - see %OUT%\navcal_tied_tiefslope_log.txt
exit /b 0

:part_nz
if /i "%MPPP_NZ%"=="none" exit /b 0
echo.
echo %DATE% %TIME%  Navcam + Mastcam-Z 34 mm combined joint ^(a check^) -^> %OUT%\nz34_joint
python -m mppp --log "%OUT%\nz34_joint_log.txt" calib nz --group "zcam34_joint,zcam_consensus" --navcam-group ecam_consensus --root "%MPPP_SCAPES%" --out "%OUT%\nz34_joint" --wait-h 0
if errorlevel 1 echo the combined joint FAILED - see %OUT%\nz34_joint_log.txt
exit /b 0

:part_lawcheck
echo.
set "COMBINED="
if exist "%OUT%\nz34_joint\navcam_law\M2020_NL_fisheye_tangential.json" set "COMBINED=--law combined=%OUT%\nz34_joint\navcam_law"
echo %DATE% %TIME%  law check: the law in use, the trial candidate and the combined law -^> %OUT%\lawcheck
python -m mppp --log "%OUT%\lawcheck_log.txt" calib lawcheck --root "%MPPP_SCAPES%" --group ecam_consensus --law in_use= --law trial_0p95="%OUT%\navcal_tied\navcam_joint" %COMBINED% --out "%OUT%\lawcheck" --wait-h 0 %*
if errorlevel 1 echo law check: FAILED - see %OUT%\lawcheck_log.txt
if exist "%OUT%\lawcheck\law_validation.json" python "%MPPP_HOME%\studies\calibration\plot_lawcheck.py" "%OUT%\lawcheck"
exit /b 0

:part_hazcam
echo.
echo %DATE% %TIME%  Hazcam fit over ecam_consensus -^> %OUT%\hazcam_fit
python -m mppp --log "%OUT%\hazcam_fit_log.txt" calib hazcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\hazcam_fit" --no-promote --wait-h 0 %*
if errorlevel 1 echo Hazcam fit: FAILED - see %OUT%\hazcam_fit_log.txt
exit /b 0

:part_zcam
for %%Z in (%MPPP_ZOOMS%) do (
  echo.
  echo %DATE% %TIME%  Mastcam-Z %%Z mm joint over zcam34_joint,zcam_consensus -^> %OUT%\zcam%%Z_joint
  python -m mppp --log "%OUT%\zcam%%Z_joint_log.txt" calib zcam --zoom %%Z --group "zcam34_joint,zcam_consensus" --root "%MPPP_SCAPES%" --out "%OUT%\zcam%%Z_joint" --no-promote --wait-h 0 --eye-slope-tests %*
  if errorlevel 1 echo Mastcam-Z %%Z mm joint: FAILED - see %OUT%\zcam%%Z_joint_log.txt
)
set "MPPP_NOPAUSE=1"
if exist "%OUT%\zcam34_joint\zcam_joint.json" call "%MPPP_HOME%\scripts\windows\make_figures.bat" - "%OUT%\zcam34_joint"
exit /b 0

:end
echo.
echo %DATE% %TIME%  trial done -^> %OUT%
echo Read: navcal_tied\report.md ^(stereo_pp, the sequence of passes, the early offsets^), nz34_joint\report.md and
echo lawcheck\report.md ^(the combined law on the Navcam blocks; stereo chi2/dof^), zcam34_joint\report.md ^(the
echo rig translation and its z, the early-mission principal point, the eye-slope tests, worst frames, thin bins^).
if not defined MPPP_NOPAUSE_END pause
exit /b 0
