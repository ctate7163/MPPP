# Roadmap: continuous integration of the MPPP pipeline, JOSS-compliant

Written for 0.90.0 (19 Oct 2026). The aim: every push runs the pipeline end to end on public data, the paper
builds from the same commit, and a reviewer can reproduce the headline numbers from a clean machine without
anything that lives only on the author's PC.

## What JOSS asks for, and where MPPP stands

| JOSS requirement | 0.90 | still to do |
|---|---|---|
| OSI license in the repo | Apache-2.0, LICENSE + NOTICE | — |
| Public repository, issue tracker | github.com/ctate7163/MPPP is private | make it public before submission; enable Issues |
| Installable, documented installation | `pip install -e .[all]`, README | a `pip install mppp` from PyPI (optional but reviewers like it); pin the pycolmap/pyceres pair in `envs/mppp.yml` |
| Example usage | README, notebooks 01–03 | a 10-minute example a reviewer can run: `mppp run` on the Rockytop test set (below) |
| API documentation | docstrings | a Sphinx or mkdocs site built in CI from the docstrings (`docs/` → GitHub Pages) |
| Automated tests | pytest, ~470 tests, two PDS products | the end-to-end block test (below); coverage report |
| Community guidelines | — | `CONTRIBUTING.md` (how to report, how to contribute, how to run the tests) and a `CODE_OF_CONDUCT.md`; both are checked by the JOSS editor |
| Statement of need, state of the field, references | `paper/paper.md` | the `@TODO:opacity` citation, figure 3, figure sizes, the generative-AI statement |
| Archive with a DOI at acceptance | — | Zenodo–GitHub integration; a tagged release `v1.0.0` |

## Stage 1 — now in `.github/workflows/tests.yml`

* Ubuntu, Python 3.11 and 3.12, CPU torch; `pip install -e .[all]`; pyflakes; the full pytest suite on the two
  PDS products in `tests/data/m20` (processing, masks off, project build, features, LightGlue matching and
  verification, calibration forms, waypoints, the error model's physics tests). No model is downloaded: the
  mask tests use a tiny synthetic checkpoint, LightGlue's weights are fetched by kornia once and cached by the
  runner (`MPPP_CACHE`).
* The paper job renders `paper/paper.md` with JOSS's own `openjournals-draft-action` and uploads the PDF as an
  artifact, after `scripts/build_paper.py --check-only` lists missing figures and placeholder citations.

What it does not cover: an alignment of a real block (no block fits in two images), the mask model, GPU code
paths, Windows.

## Stage 2 — the Rockytop test set (the reviewer's example)

1. Publish the Rockytop test set (several Navcam stations and a small Mastcam-Z 34 mm mosaic, < 1 GB of PDS
   RAD products as they are on the PDS, plus `sites.json`'s rockytop entry) on Zenodo with a DOI; `mppp fetch
   rockytop` (a new subcommand, or `scripts/fetch_testdata.py`) downloads and checksums it into `MPPP_PDS`.
2. A nightly workflow (`schedule:`) on a larger GitHub runner: `mppp run rockytop --zcam`, then a gate on the
   health report (registered fraction, median reprojection error, rig stability within the thresholds of
   `sfm/health.py`) and on the run time; the health table and `camera_shifts.png` are uploaded as artifacts.
   Expect 20–40 minutes on CPU with `max_num_features` lowered for CI (a `ci` settings file).
3. The same workflow builds the three notebooks with `jupyter nbconvert --execute` against the fetched data, so
   the README's examples are known to run.
4. Store the expected health numbers in `tests/data/rockytop_expected.json`; the gate compares with tolerances,
   not equality (SIFT and LightGlue are not bit-reproducible across machines).

## Stage 3 — GPU and Windows

* A self-hosted runner on the author's PC (or a cloud GPU runner) for the GPU paths: SIFT extraction in the
  CUDA environment, LightGlue on the torch GPU, the full-resolution timing. Run on demand (`workflow_dispatch`)
  and before a release, not on every push.
* A Windows job (`windows-latest`) that installs the pip wheels, runs the unit tests and calls `mppp process` on
  the two test products — this is what catches path and CRLF regressions in the `.bat` wrappers.

## Stage 4 — releases and the archive

* A tag `vX.Y.Z` triggers: tests, the paper, `python -m build`, a GitHub release with the wheel and the paper
  PDF, and (through the Zenodo integration) the archived DOI that JOSS asks for at acceptance. `CITATION.cff`
  and `CHANGELOG.md` are checked against the tag (`tests/test_package.py::test_version` already does this).
* `scripts/release_mask_model.py verify` runs in the release workflow so that the registry URLs serve the
  checksummed model.

## Making the Windows tooling JOSS-compliant

The pipeline itself runs from `mppp process` / `mppp align` on any platform; the Windows pieces are
conveniences and a reviewer must be able to ignore them. Three things keep them acceptable:

1. **GPU environment** — the two-environment arrangement (pip pyceres + conda-forge CUDA pycolmap) is a
   Windows packaging limitation, not a design choice. Document it as such (README *GPU on Windows*), keep the
   default path working without it (`MPPP_GPU_PYTHON` unset → CPU extraction), and test the subprocess bridge
   (`mppp.sfm.gpu`) with a fake Python in CI (it is). When conda-forge ships a pycolmap/pyceres pair that
   coexist, drop the second environment.
2. **COLMAP GUI files** — `open_in_colmap.bat` and `colmap_gui.ini` are now opt-in (`mppp align --gui`); the
   batch file looks for `COLMAP_BAT`, the PATH and the standard install folders only. No author paths remain
   in the package. A Linux/macOS equivalent is a one-line shell script that the same option could write.
3. **`.bat` wrappers** — they call the CLI and nothing else; every default comes from `MPPP_SCAPES`,
   `MPPP_HOME` (the folder two levels above the script) and `MPPP_GPU_PYTHON`; the Windows CI job runs them.
   Keep them in `scripts/windows/` and out of the paper: the paper describes the commands.

## Order of work

1. Public repository, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, Issues on; fix the paper's open items.
2. Rockytop test set on Zenodo; `fetch_testdata`; the nightly end-to-end workflow with health gates.
3. Docs site from the docstrings; coverage badge.
4. Windows job; GPU runner on demand.
5. Release workflow and the Zenodo archive; tag `v1.0.0` when the Rockytop run passes its gates on a clean
   runner.
