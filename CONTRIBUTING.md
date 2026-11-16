# Contributing to RPP

Thank you for your interest. RPP is developed by Christian Tate (Malin Space Science Systems) and released under
Apache-2.0.

**Questions and bug reports** go to the GitHub issue tracker. For a bug, include the RPP version
(`rpp version`), the command or function you called, the PDS product ids involved and the log
(`<WORK>/runs/<time>/log.txt`); a traceback is worth more than a description.

**Development setup**

```bash
git clone https://github.com/ctate7163/RPP.git
cd RPP
pip install -e ".[all]"
pytest -q                       # ~470 tests on the two PDS products in tests/data/m20 (about 10 minutes)
python -m flake8 --select F src/rpp scripts
```

**Pull requests**: one change per request, with a test in the module's test file (`tests/test_<module>.py`) and a
line in `CHANGELOG.md`. Library code logs through `logging.getLogger("rpp.<module>")`, never `print`; no
machine-specific paths; no camera-model files outside `src/rpp/data/cmods/` (new models go through
`rpp_studies.calibration.promote_cmods`). Research code that imports the package but is not part of the pipeline belongs in
`src/rpp_studies/` (a study of its own, or `misc/` when unsure).

**Camera models and site definitions** are the author's decisions (`rpp cmods list`, `rpp sites list`); open an
issue before proposing a change to them.
