@echo off
REM ======================================================================================================
REM MPPP - bring GitHub and this git working copy together, then push (v0p50.1).
REM
REM Fetches github.com/<you>/MPPP first ("pull before push"):
REM   - GitHub has nothing new             -> push.
REM   - GitHub has commits of this history -> merge them here (git merge), then push; stops on a conflict.
REM   - GitHub has an unrelated history    -> (e.g. an earlier upload of MPPP) you choose:
REM       K  keep it as the branch github-before-v0p50 on GitHub and make main this history   (recommended)
REM       M  merge it into this history (this folder's files win where both changed; its other files are added)
REM       Q  quit, nothing changed
REM Afterwards _transfer\mppp_pc.bundle holds this history for Claude's next session.
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || (pause & exit /b 1)
if not exist ".git" (echo not a git working copy yet: run setup_github.bat first & pause & exit /b 1)
git remote get-url origin >nul 2>nul || (echo no GitHub remote yet: run setup_github.bat first & pause & exit /b 1)

echo fetching GitHub ...
git fetch origin || (echo cannot reach GitHub - check that you are signed in & pause & exit /b 1)
git rev-parse -q --verify origin/main >nul || goto :push
git merge-base --is-ancestor origin/main HEAD && goto :push

echo.
echo GitHub main has commits that this copy does not have:
git log --oneline -8 origin/main
echo.
git merge-base HEAD origin/main >nul 2>nul
if errorlevel 1 goto :unrelated

echo They share history with this copy: merging them in.
git merge --no-edit origin/main
if errorlevel 1 (
  git merge --abort
  echo The merge has conflicts - nothing changed. Tell Claude which files, or merge by hand: git merge origin/main
  pause & exit /b 1
)
goto :push

:unrelated
echo This is a different history (not made from this one), e.g. an earlier upload of MPPP.
echo Files on GitHub:
git ls-tree --name-only origin/main
echo.
choice /C KMQ /M "K = keep it as branch github-before-v0p50 and replace main (recommended), M = merge, Q = quit"
if errorlevel 3 (echo nothing changed & pause & exit /b 1)
if errorlevel 2 goto :merge_unrelated
git push origin "refs/remotes/origin/main:refs/heads/github-before-v0p50" || (pause & exit /b 1)
echo the old GitHub main is kept as the branch github-before-v0p50
git push --force-with-lease=main:origin/main -u origin main --tags || (pause & exit /b 1)
goto :done

:merge_unrelated
git merge --no-edit --allow-unrelated-histories -X ours origin/main -m "Merge the earlier GitHub history of MPPP" || (git merge --abort & echo merge failed - nothing changed & pause & exit /b 1)

:push
git push -u origin main --tags || (echo push failed & pause & exit /b 1)

:done
if not exist "_transfer" mkdir "_transfer"
git bundle create -q "_transfer\mppp_pc.bundle" --all && echo history for Claude: _transfer\mppp_pc.bundle
echo.
git log --oneline -3
echo GitHub is up to date.
pause
