# One-off experiments (moved from `scripts/` in v0p50)

Scripts written for one analysis; kept for the record and for re-running. They import MPPP from `src/` and the
tools in `scripts/` (e.g. `navcam_calibration_study.py`); run them from the MPPP folder, e.g.

    python studies\experiments\temperature_bins_experiment.py OUT.json D:\scapes\colmap\rockytop_colmap

| script | what |
|---|---|
| `pair_experiment.py` | v0p22.1 single-pair test: generic SfM on a Mars stereo pair |
| `lens_model_experiment.py` | v0p30 lens-model comparison on one project |
| `lens_terms_experiment.py` | v0p22.3 which Navcam lens terms the blocks need |
| `temperature_bins_experiment.py` | v0p31 within-block temperature bins (notebook 04 reads its JSON) |
| `error_sources.py` | error budget from scape folders |
| `run_v0p22_batch.bat` | the v0p22 batch (uses the retired `scripts/run_scapes.py` command line) |
