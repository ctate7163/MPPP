@echo off
REM ======================================================================================================
REM MPPP 0.95.0 - trial_joints_0p95.bat (Christian, 28 Oct 2026): the 0.95.0 joints over the blocks already aligned
REM (0.94.x), NOTHING PROMOTED - a trial of the joint code before the freeze reruns, which saves days of alignment.
REM The 0.94 blocks have focus bins (not sequence bins), so the Mastcam-Z joint runs without sequence offsets; the
REM rest of 0.95.0 (one slope per eye, the paired low-state ZR centre, the fb hold, the worst frames, the thin-bin
REM cut; the Navcam stereo-difference blocks, the exact-tie skip, the pass rule, the stereo law check) is exercised.
REM
REM   trial_joints_0p95.bat                 everything below, one after another (~3-5 h)
REM   set MPPP_TRIAL=navcam | hazcam | zcam | lawcheck | nz   before starting: only that part
REM   set MPPP_ZOOMS=34                    only these Mastcam-Z zooms (default 34 48 63)
REM   trial_joints_0p95.bat --include-failed   arguments go to every "mppp calib"
REM
REM Writes %MPPP_SCAPES%\camera_analysis\trial_0p95_<date>\ :
REM   navcal_tied\          the Navcam joint (candidate navcal_tied\navcam_joint, promotion_check.json, report)
REM   navcal_tied_tiefslope\  the same with NR's f slope tied to NL's (--tie-f-slope): the comparison run
REM   lawcheck\             the law in use against the trial candidate on the same blocks (stereo chi2/dof now real)
REM   hazcam_fit\           the Hazcam fit
REM   zcam<zoom>_joint\     the Mastcam-Z joints (34 over zcam34_joint + zcam_consensus; 48, 63 over both too)
REM   nz34_joint\           the N+Z check (set MPPP_NZ=1 to run it; ~1 h)
REM Running blocks are not waited for (--wait-h 0): the finished ones are taken as they are.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
if not defined MPPP_ZOOMS set "MPPP_ZOOMS=34 48 63"
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

if /i "%MPPP_TRIAL%"=="all" goto :navcam
if /i "%MPPP_TRIAL%"=="navcam" goto :navcam
if /i "%MPPP_TRIAL%"=="lawcheck" goto :lawcheck
if /i "%MPPP_TRIAL%"=="hazcam" goto :hazcam
if /i "%MPPP_TRIAL%"=="zcam" goto :zcam
if /i "%MPPP_TRIAL%"=="nz" goto :nz
echo unknown MPPP_TRIAL=%MPPP_TRIAL%
pause
exit /b 2

:navcam
echo %DATE% %TIME%  Navcam joint over ecam_consensus -^> %OUT%\navcal_tied
python -m mppp --log "%OUT%\navcal_tied_log.txt" calib navcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\navcal_tied" --knot 250 --points 8000 --no-promote --wait-h 0 %*
if errorlevel 1 echo Navcam joint: FAILED - see %OUT%\navcal_tied_log.txt
echo.
echo %DATE% %TIME%  Navcam joint, NR's f slope tied to NL's ^(the comparison run^) -^> %OUT%\navcal_tied_tiefslope
python -m mppp --log "%OUT%\navcal_tied_tiefslope_log.txt" calib navcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\navcal_tied_tiefslope" --knot 250 --points 8000 --tie-f-slope --no-promote --wait-h 0 %*
if errorlevel 1 echo Navcam joint ^(tie f slope^): FAILED - see %OUT%\navcal_tied_tiefslope_log.txt
if /i not "%MPPP_TRIAL%"=="all" goto :end

:lawcheck
echo.
echo %DATE% %TIME%  law check: the law in use and the trial candidate -^> %OUT%\lawcheck
python -m mppp --log "%OUT%\lawcheck_log.txt" calib lawcheck --root "%MPPP_SCAPES%" --group ecam_consensus --law in_use= --law trial_0p95="%OUT%\navcal_tied\navcam_joint" --out "%OUT%\lawcheck" --wait-h 0 %*
if errorlevel 1 echo law check: FAILED - see %OUT%\lawcheck_log.txt
if exist "%OUT%\lawcheck\law_validation.json" python "%MPPP_HOME%\studies\calibration\plot_lawcheck.py" "%OUT%\lawcheck"
if /i not "%MPPP_TRIAL%"=="all" goto :end

:hazcam
echo.
echo %DATE% %TIME%  Hazcam fit over ecam_consensus -^> %OUT%\hazcam_fit
python -m mppp --log "%OUT%\hazcam_fit_log.txt" calib hazcam --root "%MPPP_SCAPES%" --group ecam_consensus --out "%OUT%\hazcam_fit" --no-promote --wait-h 0 %*
if errorlevel 1 echo Hazcam fit: FAILED - see %OUT%\hazcam_fit_log.txt
if /i not "%MPPP_TRIAL%"=="all" goto :end

:zcam
for %%Z in (%MPPP_ZOOMS%) do (
  echo.
  echo %DATE% %TIME%  Mastcam-Z %%Z mm joint over zcam34_joint,zcam_consensus -^> %OUT%\zcam%%Z_joint
  python -m mppp --log "%OUT%\zcam%%Z_joint_log.txt" calib zcam --zoom %%Z --group "zcam34_joint,zcam_consensus" --root "%MPPP_SCAPES%" --out "%OUT%\zcam%%Z_joint" --no-promote --wait-h 0 %*
  if errorlevel 1 echo Mastcam-Z %%Z mm joint: FAILED - see %OUT%\zcam%%Z_joint_log.txt
)
set "MPPP_NOPAUSE=1"
if exist "%OUT%\zcam34_joint\zcam_joint.json" call "%MPPP_HOME%\scripts\windows\make_figures.bat" - "%OUT%\zcam34_joint"
if /i not "%MPPP_TRIAL%"=="all" goto :end
if not defined MPPP_NZ goto :end

:nz
echo.
echo %DATE% %TIME%  Navcam + Mastcam-Z 34 mm combined joint ^(a check^) -^> %OUT%\nz34_joint
python -m mppp --log "%OUT%\nz34_joint_log.txt" calib nz --group "zcam34_joint,zcam_consensus" --navcam-group ecam_consensus --root "%MPPP_SCAPES%" --out "%OUT%\nz34_joint" --wait-h 0
if errorlevel 1 echo the combined joint FAILED - see %OUT%\nz34_joint_log.txt

:end
echo.
echo %DATE% %TIME%  trial done -^> %OUT%
echo Read: navcal_tied\report.md ^(stereo_pp, the sequence of passes, the early offsets^), lawcheck\report.md
echo ^(stereo chi2/dof^), zcam34_joint\report.md ^(worst frames, thin bins; no sequence offsets on 0.94 blocks^).
if not defined MPPP_NOPAUSE_END pause
exit /b 0
