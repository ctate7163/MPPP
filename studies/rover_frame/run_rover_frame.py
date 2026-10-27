"""
Study 3A, the rover frame (0.94.4, Christian 23 Oct 2026) over the finished blocks - no new adjustment:

    python studies\\rover_frame\\run_rover_frame.py                      (run_pipeline's rover_frame stage)
    python studies\\rover_frame\\run_rover_frame.py --sites chal_rocks_tiny seitah_overlook_tiny

- the RSM angles (encoder, resolver, requested), the head frame, the rover attitude and motion counters, the
  Navcam plate temperatures, the suspension and arm angles from the labels (cached in
  ``<root>/camera_analysis/rover_frame_labels.json``);
- per Hazcam sequence and rover pose the body pose (the Hazcams are body-fixed) and its residual against the
  telemetry attitude; the body-pose changes between the sequences of one station (wheel settling, the arm);
- per mast frame (Navcam, Mastcam-Z) the attitude relative to that body in the label's head frame (the label
  kinematics removed): a constant mount plus the RSM pointing error and flexure, fitted against az, el and T;
- the camera positions in the body frame (Hazcams) and the head frame (mast cameras): constants whose scatter is the
  measurement.

Output: ``<root>/camera_analysis/rover_frame_<date>/`` (report.md, images.csv, body_poses.csv, body_changes.csv,
stations.csv, rover_frame.json, figures), the figures also in ``<repo>/Claude outputs/rover_frame_<date>/``.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes", help="the scapes root")
    ap.add_argument("--pds", default=os.environ.get("MPPP_PDS"), help="the PDS archive (default MPPP_PDS)")
    ap.add_argument("--groups", nargs="*", default=["ecam_consensus"], help="engineering-camera block groups")
    ap.add_argument("--zcam-groups", nargs="*", default=["zcam_consensus"], help="Navcam + Mastcam-Z block groups")
    ap.add_argument("--sites", nargs="*", default=None, help="these sites instead of the groups (both block kinds)")
    ap.add_argument("--out", default=None, help="the output folder (default <root>/camera_analysis/rover_frame_<date>)")
    ap.add_argument("--workers", type=int, default=8, help="processes reading the labels")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-kernels", action="store_true", help="no rover kernels (SPICE CK / SPK / FK, spiceypy)")
    ap.add_argument("--kernels-out", default=None,
                    help="the rover kernels' folder (default <root>/camera_analysis/rover_kernels_<date>)")
    ap.add_argument("--naif", default=None, help="the NAIF base kernels (default MPPP_NAIF or <cache>/naif, fetched)")
    a = ap.parse_args(argv)
    from rover_frame import analysis, blocks, labels, report
    root = Path(a.root)
    found = blocks.finished_blocks(root, a.groups, a.zcam_groups, a.sites)
    if not found:
        print(f"no finished block under {root}")
        return 2
    print(f"{len(found)} finished blocks: {', '.join(found)}")
    images = []
    for name, w in found.items():
        try:
            ims = blocks.block_images(name, w)
        except Exception as e:                           # noqa: BLE001 - one bad block does not stop the study
            print(f"  {name}: not read ({type(e).__name__}: {e})")
            continue
        print(f"  {name}: {len(ims)} registered images")
        images += ims
    stems = sorted({im["stem"] for im in images})
    pds = Path(a.pds) if a.pds else None
    states = labels.states(stems, pds, root / "camera_analysis" / "rover_frame_labels.json", workers=a.workers)
    print(f"label states: {len(states)} of {len(stems)}")
    res = analysis.run(images, states)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"rover_frame_{datetime.date.today():%Y%m%d}"
    done = report.write_all(res, out, {k: str(v) for k, v in found.items()}, plots=not a.no_plots)
    co = ROOT / "Claude outputs"
    if done["figures"] and co.is_dir():
        dst = co / out.name
        dst.mkdir(exist_ok=True)
        for f in done["figures"] + [done["report"]]:
            shutil.copy2(f, dst / Path(f).name)
        print(f"figures also in {dst}")
    print(f"report: {done['report']}")
    if not a.no_kernels:
        kout = Path(a.kernels_out) if a.kernels_out else root / "camera_analysis" / f"rover_kernels_{datetime.date.today():%Y%m%d}"
        try:
            kernels(res, kout, a.naif)
        except Exception as e:                           # noqa: BLE001 - the study's report stands without them
            print(f"rover kernels not written ({type(e).__name__}: {e})")
    return 0


def kernels(res, out: Path, naif=None) -> dict:
    """The rover kernels (study 3A, 0.94.4): the attitude intervals and Hazcam mounts of :mod:`rover_frame.estimate`
    written by :mod:`rover_frame.spice` into ``out``."""
    from rover_frame import estimate, spice
    spice.spiceypy()                                     # a clear message when spiceypy is missing
    ivs, gt = estimate.intervals(res["_rows"], res["_groups"])
    if not ivs:
        print("rover kernels: no block with a Hazcam body pose")
        return {}
    mounts = estimate.hazcam_mounts(res["_rows"], res["rows"], gt)
    print(f"rover kernels: {len(ivs)} intervals over {len({iv['block'] for iv in ivs})} blocks; "
          f"rear-to-front correction {', '.join(f'{v:.0f}' for v in gt['Q_mdeg'])} mdeg; NAIF base kernels:")
    nk = spice.naif_kernels(Path(naif) if naif else None)
    summary = spice.write(ivs, gt, mounts, out, nk)
    print(f"rover kernels: {out} ({len(summary['files'])} files, meta-kernel {summary['meta_kernel']})")
    return summary


if __name__ == "__main__":
    sys.exit(main())
