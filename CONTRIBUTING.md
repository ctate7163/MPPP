# Contributing to RPP

Thank you for your interest. RPP is developed by Christian Tate (Malin Space Science Systems) and released under
Apache-2.0.

**Questions and bug reports** go to the GitHub issue tracker. For a bug, include the RPP version
(`rpp version`), the command or function you called, the PDS product ids involved and the log
(`<WORK>/runs/<time>/log.txt`); a traceback is worth more than a description.

**Development setup**

```bash
conda create -n rpp -c conda-forge python=3.12 git -y
conda activate rpp
git clone https://github.com/ctate7163/rpp.git
cd rpp
pip install -e ".[all]"
pytest -q
python -m flake8 --select F src/rpp scripts
```

The quick tests (`pytest -q -m "not slow"`) take about half an hour on a laptop; `pytest -q tests/test_<module>.py`
runs one module. Set `RPP_TEST_FETCH=1` once so that the two PDS products some tests read are
downloaded from the PDS into the user cache (they are not in the repository; without them those tests are skipped).

**Pull requests**: one change per request, with a test in the module's test file (`tests/test_<module>.py`) and a
line in `CHANGELOG.md`. Library code logs through `logging.getLogger("rpp.<module>")`, never `print`; no
machine-specific paths; no camera-model files outside `src/rpp/data/cmods/` (new models go through
`rpp_studies.calibration.promote_cmods`). Research code that imports the package but is not part of the pipeline belongs in
`src/rpp_studies/`, as a study of its own. The core (`src/rpp`) never imports a study.

**Camera models and site definitions** are the author's decisions (`rpp cmods list`, `rpp sites list`); open an
issue before proposing a change to them.
