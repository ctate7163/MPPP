"""The finished blocks of study 3A and their adjusted image poses (``<WORK>/colmap/sparse/<model>``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

MODEL = "cahv_ba"


def finished_blocks(root: Path, groups: Sequence[str] = ("ecam_consensus",), zcam_groups: Sequence[str] = ("zcam_consensus",),
                    sites: Optional[Sequence[str]] = None, model: str = MODEL) -> Dict[str, Path]:
    """{block name: WORK} of the finished blocks of the site groups (``<site>`` for the engineering-camera block,
    ``<site>_zcam`` for the Navcam + Mastcam-Z one), or of ``sites`` (both kinds where finished)."""
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    gs = table.get("groups", {})
    want = []
    if sites:
        want = [(s, False) for s in sites] + [(s, True) for s in sites]
    else:
        for g in groups:
            want += [(s, False) for s in gs.get(g) or []]
        for g in zcam_groups:
            want += [(s, True) for s in gs.get(g) or []]
    out: Dict[str, Path] = {}
    for name, zcam in want:
        site = table["sites"].get(name)
        if not site or (zcam and not site.get("zcam")):
            continue
        w = work_folder(Path(root), name, zcam, site["sols"])
        if (w / "colmap" / "run_done.json").is_file() and (w / "colmap" / "sparse" / model).is_dir():
            out[f"{name}_zcam" if zcam else name] = w
    return out


def block_images(name: str, work: Path, model: str = MODEL) -> List[Dict[str, Any]]:
    """The registered images of a finished block with their adjusted and prior poses (study 3A input)."""
    import pycolmap
    from mppp.sfm.project import SfmProject
    root = Path(work) / "colmap"
    project = SfmProject.load(root)
    rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    by_name = {r["name"]: r for r in project.images}
    n_obs: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            n_obs[el.image_id] = n_obs.get(el.image_id, 0) + 1
    out = []
    for iid, im in rec.images.items():
        r = by_name.get(im.name)
        if r is None or not im.has_pose:
            continue
        T = im.cam_from_world()
        R = np.asarray(T.rotation.matrix(), float)
        C = -R.T @ np.asarray(T.translation, float)
        out.append({"block": name, "name": im.name, "stem": r.get("stem") or Path(im.name).stem,
                    "instrument": r.get("instrument", ""), "station": r.get("station"), "sol": r.get("sol"),
                    "sequence": r.get("sequence"), "sclk": r.get("sclk_key"),
                    "R_c_w": R, "C_w": C, "prior_R_c_w": np.asarray(r["prior_R_w2c"], float),
                    "prior_C": np.asarray(r.get("prior_C_label") or r["prior_C"], float), "observations": n_obs.get(iid, 0),
                    "camera_temperature_degC": r.get("camera_temperature_degC"),
                    "zcam_optics_temperature_degC": r.get("zcam_optics_temperature_degC"),
                    "offset": list(map(float, project.offset))})   # the block's ENU offset (rover kernels)
    return out
