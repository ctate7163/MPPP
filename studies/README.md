# Research scripts (not installed)

`error/study_navcam.py` and `error/study_cross_station.py` are studies with the error model (`mppp.error`, under development). Run them from the repository after `pip install -e .`, e.g. `python studies/error/study_navcam.py --help`. The error-model self-test (`python -m mppp.error selftest`) runs `study_navcam.py` from here when it is present.
