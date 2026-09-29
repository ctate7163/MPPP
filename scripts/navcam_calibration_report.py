"""
Tables and figure of the v0p35 Navcam calibration study (mppp.sfm.navcal_report).

  python scripts/navcam_calibration_report.py RIG_DIR JOINT_DIR OUT_DIR [SCAPES.json LABEL_SAMPLES.json]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mppp.sfm.navcal_report import main  # noqa: E402

if __name__ == "__main__":
    main(*sys.argv[1:6])
