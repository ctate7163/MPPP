"""
Camera-model solutions across scapes (v0p21, notebook 04).

A *solution* is what notebook 03 leaves in a COLMAP project folder: the
refined cameras and rigs (``error_input/summary.json``), the refined poses
(``error_input/native/frames.txt``), the image table (``project.json``) and,
if the processed folder is next to it, the MPPP manifest with each image's
label camera model.  This module compares them:

* :func:`camera_table`, :func:`reference_differences` - intrinsics per camera
  and scape, and how far each solution moves pixels relative to a reference
  (after removing the rotation a pose absorbs);
* :func:`stereo_pairs`, :func:`stereo_effect` - the left/right geometry
  (Navcam rig, Mastcam-Z simultaneous pairs) and what a difference in it does
  to disparity and range;
* :func:`focus_table`, :func:`fit_focus_model` - Mastcam-Z focal length
  against focus motor count across scapes;
* :func:`label_models` - the PDS label models (CAHVORE Navcam, CAHVOR
  Mastcam-Z) as :class:`mppp.cmod.CameraModel` in full-frame pixels;
* :func:`load_example`, :func:`undistort`, :func:`screen_mask` - images for
  a look at each camera, with the hardware mask screened in white.
"""
from __future__ import annotations

import copy
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..cmod import CameraModel, PixelCamera, compare_cameras, pixel_grid

PathLike = Union[str, Path]
PARAM_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
ZCAM_PIXEL_MM = 0.0074                  # Mastcam-Z detector pitch (7.4 um), for focal lengths in mm
FULL_FRAME = {"N": (5120, 3840), "Z": (1648, 1200)}


# ================================================================ loading
@dataclass
class Camera:
    """One COLMAP camera of a solution (full-frame pixels, corner origin)."""
    key: str
    group: str                      # NL, NR, ZL034, ZR034
    model: str
    width: int
    height: int
    params: np.ndarray              # refined
    initial: np.ndarray
    focus: Optional[float] = None   # Mastcam-Z focus motor count (bin median)
    n_images: int = 0
    n_obs: int = 0
    fixed: Tuple[str, ...] = ()

    @property
    def family(self) -> str:
        return self.group[:1]

    @property
    def refined(self) -> bool:
        return self.n_obs > 0 and not np.allclose(self.params, self.initial)

    @property
    def distortion(self) -> str:
        """'rational' (k4..k6 in the denominator), 'polynomial', 'fisheye_tangential' (v0p35: THIN_PRISM_FISHEYE),
        'fisheye' or 'pinhole'."""
        if self.model == "THIN_PRISM_FISHEYE":
            return "fisheye_tangential"
        if self.model == "OPENCV_FISHEYE":
            return "fisheye"
        if self.model != "FULL_OPENCV":
            return "polynomial" if self.model in ("OPENCV", "RADIAL", "SIMPLE_RADIAL") else "pinhole"
        return "rational" if np.any(np.abs(self.params[9:12]) > 0) else "polynomial"

    def pixel_camera(self, initial: bool = False) -> PixelCamera:
        return PixelCamera.colmap(self.model, self.initial if initial else self.params, self.width, self.height,
                                  name=self.key)

    def named(self, initial: bool = False) -> Dict[str, float]:
        p = self.initial if initial else self.params
        return {k: float(v) for k, v in zip(PARAM_NAMES, p)}


@dataclass
class Solution:
    label: str
    root: Path                                       # COLMAP project folder
    project: Dict[str, Any]
    summary: Dict[str, Any]
    cameras: Dict[str, Camera]
    images: Dict[str, Dict[str, Any]]                # name -> image row (+ refined R_w2c, C)
    rig_initial: Dict[str, Any] = field(default_factory=dict)
    rig_refined: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)    # PDS stem -> MPPP meta

    def groups(self, family: Optional[str] = None) -> List[str]:
        return sorted({c.group for c in self.cameras.values() if family is None or c.family == family})

    def cameras_of(self, group: str) -> List[Camera]:
        return sorted((c for c in self.cameras.values() if c.group == group), key=lambda c: (c.focus or 0, c.key))

    @property
    def navcam_distortion(self) -> str:
        n = [c.distortion for c in self.cameras.values() if c.family == "N"]
        return n[0] if n else "-"

    def camera_of_image(self, name: str) -> Camera:
        return self.cameras[self.images[name]["instrument"]]


def _find_root(path: PathLike) -> Path:
    p = Path(path)
    for cand in (p, p / "colmap", p.parent if p.name == "error_input" else p):
        if (cand / "project.json").is_file() and (cand / "error_input" / "summary.json").is_file():
            return cand
    raise FileNotFoundError(f"no COLMAP solution (project.json + error_input/summary.json) at {p} or {p / 'colmap'}:"
                            f" run notebook 03 to the export step first")


def _group_of(key: str, cam: Dict[str, Any]) -> str:
    if cam.get("group"):
        return str(cam["group"])
    m = re.match(r"(Z[LR]\d{3})", key)
    return m.group(1) if m else key[:2]


def _focus_of(key: str, cam: Dict[str, Any]) -> Optional[float]:
    if cam.get("focus_count_median") is not None:
        return float(cam["focus_count_median"])
    m = re.search(r"_F(m?)(\d{5})$", key)
    return None if not m else (-1.0 if m.group(1) else 1.0) * float(m.group(2))


def _read_frames(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """native/frames.txt -> {image id: (R_w2c, C)} (one frame per image in the native model)."""
    from scipy.spatial.transform import Rotation
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t = line.split()
        q = np.array(t[2:6], float)
        tv = np.array(t[6:9], float)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        if int(t[9]) != 1:
            raise ValueError(f"{path}: frame {t[0]} holds {t[9]} images; the native model of notebook 03 has one "
                             f"image per frame")
        out[int(t[12])] = (R, -R.T @ tv)
    return out


def _read_manifest(root: Path, processed_dir: Optional[PathLike]) -> Dict[str, Dict[str, Any]]:
    cands = [Path(processed_dir)] if processed_dir else []
    pd = root.parent / "processed"
    cands += [pd]
    for d in cands:
        files = sorted(d.glob("mppp_manifest_v*.json"), key=lambda f: f.stat().st_mtime) if d.is_dir() else []
        if files:
            m = json.loads(files[-1].read_text(encoding="utf-8"))
            return {Path(x["source_product"]).stem: x for x in m.get("images", []) if "failed" not in x}
    return {}


def load_solution(path: PathLike, label: Optional[str] = None, processed_dir: Optional[PathLike] = None) -> Solution:
    """``path``: the scape folder (``D:/scapes/<name>_colmap``) or its ``colmap`` project folder."""
    root = _find_root(path)
    project = json.loads((root / "project.json").read_text(encoding="utf-8"))
    ei = root / "error_input"
    summary = json.loads((ei / "summary.json").read_text(encoding="utf-8"))
    rows = {r["name"]: dict(r) for r in project["images"]}
    for fn in ("stations.csv", "poses.csv"):
        if (ei / fn).is_file():
            with (ei / fn).open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["name"] in rows:
                        rows[r["name"]].update({k: v for k, v in r.items() if k not in rows[r["name"]] or k == "instrument"})
    poses = _read_frames(ei / "native" / "frames.txt") if (ei / "native" / "frames.txt").is_file() else {}
    for r in rows.values():
        if r.get("image_id") in poses:
            r["R_w2c"], r["C"] = poses[r["image_id"]]
    n_obs: Dict[str, int] = {}
    n_img: Dict[str, int] = {}
    for r in rows.values():
        n_obs[r["instrument"]] = n_obs.get(r["instrument"], 0) + int(float(r.get("observations") or 0))
        n_img[r["instrument"]] = n_img.get(r["instrument"], 0) + 1
    ci, cr = summary.get("cameras_initial", {}), summary.get("cameras_refined", {})
    pc = project.get("cameras", {})
    cams = {}
    for key in sorted(set(ci) | set(cr) | set(pc)):
        c0 = dict(pc.get(key, {}), **ci.get(key, {}))
        c1 = cr.get(key, c0)
        if "params" not in c0 and "params" not in c1:
            continue
        fam = key[:1]
        w, h = c0.get("width") or FULL_FRAME.get(fam, (None, None))[0], c0.get("height") or FULL_FRAME.get(fam, (None, None))[1]
        p0 = np.asarray(c0.get("params", c1.get("params")), float)
        cams[key] = Camera(key, _group_of(key, c0), c1.get("model", c0.get("model")), int(w), int(h),
                           np.asarray(c1.get("params", p0), float), p0, _focus_of(key, c0), n_img.get(key, 0),
                           n_obs.get(key, 0), tuple(c0.get("fixed_params") or ()))
    return Solution(label or root.parent.name, root, project, summary, cams, rows,
                    summary.get("rig_initial") or project.get("rig") or {}, summary.get("rig_refined") or {},
                    _read_manifest(root, processed_dir))


def load_solutions(scapes: Dict[str, PathLike], processed_dirs: Optional[Dict[str, PathLike]] = None,
                   verbose: bool = True) -> Dict[str, Solution]:
    out = {}
    for name, path in scapes.items():
        try:
            out[name] = load_solution(path, name, (processed_dirs or {}).get(name))
        except FileNotFoundError as e:
            if verbose:
                print(f"{name}: skipped ({e})")
            continue
        if verbose:
            s = out[name]
            fams = ", ".join(f"{g} x{len(s.cameras_of(g))}" for g in s.groups())
            print(f"{name}: {len(s.images)} images, cameras {fams}; Navcam lens {s.navcam_distortion}; "
                  f"manifest {'yes' if s.manifest else 'no'}")
    return out


def solutions_table(sols: Dict[str, Solution]) -> List[Dict[str, Any]]:
    rows = []
    for n, s in sols.items():
        st = s.project.get("settings", {})
        rows.append({"scape": n, "images": len(s.images), "stations": len({r["station"] for r in s.images.values()}),
                     "cameras": len(s.cameras), "navcam_lens": s.navcam_distortion,
                     "sift_max_image_size": st.get("features", {}).get("max_image_size"),
                     "residual_median_px": s.summary.get("residual_median_native_px"),
                     "residual_rms_px": s.summary.get("residual_rms_native_px"),
                     "Mastcam-Z": "yes" if s.groups("Z") else "no"})
    return rows


# ================================================================ intrinsics
def camera_table(sols: Dict[str, Solution], family: Optional[str] = None, refined_only: bool = False) -> List[Dict[str, Any]]:
    """One row per camera and scape: refined parameters, change from the start values, images, observations."""
    rows = []
    for n, s in sols.items():
        for c in sorted(s.cameras.values(), key=lambda c: (c.group, c.focus or 0)):
            if (family and c.family != family) or (refined_only and not c.refined):
                continue
            r = {"scape": n, "camera": c.key, "group": c.group, "lens": c.distortion, "focus": c.focus,
                 "images": c.n_images, "observations": c.n_obs, "refined": c.refined}
            r.update({k: v for k, v in c.named().items() if k not in ("k5", "k6")})
            d = c.params - c.initial
            r.update({f"d{k}": float(v) for k, v in zip(PARAM_NAMES[:4], d[:4])})
            rows.append(r)
    return rows


def camera_temperature(sol: "Solution", cam: "Camera") -> Optional[float]:
    """v0p31: the camera temperature a refined camera stands for: a temperature-bin camera's own median, else the
    median over its images (manifest or project record); None without temperatures."""
    pc = sol.project.get("cameras", {}).get(cam.key) or {}
    if pc.get("temperature_median_degC") is not None:
        return float(pc["temperature_median_degC"])
    t = []
    for r in sol.images.values():
        if r["instrument"] != cam.key:
            continue
        v = (sol.manifest.get(Path(r["name"]).stem) or {}).get("camera_temperature_degC")
        if v is None:
            v = r.get("camera_temperature_degC")
        if v is not None:
            t.append(float(v))
    return float(np.median(t)) if t else None


def thermal_scale(model: Optional[Dict[str, Dict[str, float]]], group: str, T: Optional[float],
                  to_T0: bool = True) -> float:
    """Factor on fx, fy between a camera at temperature ``T`` and the reference temperature of ``model``
    ({group: {"ppm_per_degC", "T0_degC"}}): to T0 (``to_T0``) or from T0 to T.  1 without a model or temperature."""
    m = (model or {}).get(group)
    if not m or T is None:
        return 1.0
    s = 1.0 + 1e-6 * float(m["ppm_per_degC"]) * (float(T) - float(m["T0_degC"]))
    return 1.0 / s if to_T0 else s


def consensus_camera(sols: Dict[str, Solution], group: str, lens: Optional[str] = None,
                     min_observations: int = 1000, max_rms_px: Optional[float] = 1.5,
                     thermal: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[Camera]:
    """
    Observation-weighted mean of the refined cameras of ``group`` (one lens model;
    Navcam: the most common one unless ``lens`` is given).  Mastcam-Z focus bins
    are not averaged here.  ``max_rms_px`` (v0p22.1): cameras more than this far
    (rms over the frame, rotation removed) from the mean of the others are left
    out, one at a time, worst first - a scape whose few Navcam images all stand
    at one spot cannot fix its own intrinsics (methods section 14).  The
    left-out cameras are listed in ``excluded``.  ``thermal`` (v0p31: {group:
    {"ppm_per_degC", "T0_degC"}}): every camera's fx, fy are first scaled to the
    reference temperature T0 (:func:`camera_temperature`), so the consensus is
    the camera at T0 and the scapes' thermal offsets do not count as scatter;
    temperature-bin cameras then enter one per bin.
    """
    cams, names = [], {}
    for n, s_ in sols.items():
        for c in s_.cameras_of(group):
            if c.refined and c.n_obs >= min_observations:
                if thermal:
                    f = thermal_scale(thermal, group, camera_temperature(s_, c))
                    c = copy.copy(c)
                    c.params = np.array(c.params, float)
                    c.params[:2] *= f
                cams.append(c)
                names[id(c)] = n
    if not cams:
        return None
    if lens is None:
        kinds = [c.distortion for c in cams]
        lens = max(set(kinds), key=kinds.count)
    cams = [c for c in cams if c.distortion == lens]
    if not cams:
        return None

    def _mean(cs):
        w = np.array([c.n_obs for c in cs], float)
        P = np.array([c.params for c in cs])
        return (w[:, None] * P).sum(0) / w.sum(), int(w.sum())

    excluded = []
    while len(cams) > 2 and max_rms_px:
        worst = None
        for i, c in enumerate(cams):
            others = cams[:i] + cams[i + 1:]
            p_o, _ = _mean(others)
            ref = Camera("ref", group, c.model, c.width, c.height, p_o, p_o).pixel_camera()
            d = compare_cameras(ref, c.pixel_camera(), step=192.0)["rms_px"]
            if d > max_rms_px and (worst is None or d > worst[1]):
                worst = (i, d)
        if worst is None:
            break
        excluded.append({"scape": names[id(cams[worst[0]])], "camera": cams[worst[0]].key, "rms_px": float(worst[1])})
        cams.pop(worst[0])
    p, n_obs = _mean(cams)
    c0 = cams[0]
    out = Camera(f"{group} consensus ({lens}, {len(cams)} scapes)", group, c0.model, c0.width, c0.height, p, p,
                 None, sum(c.n_images for c in cams), n_obs)
    out.__dict__["excluded"] = excluded
    out.__dict__["thermal"] = (thermal or {}).get(group)
    return out


def write_navcam_consensus(cameras: Dict[str, Camera], rig: Optional[Tuple[np.ndarray, np.ndarray]],
                           out_dir: PathLike, sols: Optional[Dict[str, Solution]] = None,
                           repeatability: Optional[Dict[str, Any]] = None, step: float = 96.0) -> Dict[str, Path]:
    """
    Write the verified consensus Navcam cameras (v0p22.2) in the format of the
    shipped start cameras, so that ``SfmProject.create(navcam_cameras=out_dir)``
    starts a project from them: ``M2020_NL_rational.json``,
    ``M2020_NR_rational.json`` and, with ``rig``, ``M2020_N_rig.json``.
    ``cameras``: ``{"NL": Camera, "NR": Camera}`` from :func:`consensus_camera`
    (its ``excluded`` list is recorded).  With ``sols``, each contributing
    solution's rms distance from the consensus over the frame (rotation
    removed) is recorded as ``verification`` - the scape-to-scape
    repeatability of the calibration - together with ``repeatability``, any
    further numbers the caller wants kept with the file (the ``eps`` rows of
    notebook 05, the error analysis, say).  Returns the paths written.
    """
    from scipy.spatial.transform import Rotation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for g in ("NL", "NR"):
        c = cameras.get(g)
        if c is None:
            continue
        per_scape = []
        if sols:
            for n, s_ in sols.items():
                for cc in s_.cameras_of(g):
                    if cc.refined and cc.distortion == c.distortion:
                        # v0p31: with a thermal model, the consensus is first scaled to the camera's temperature
                        th = {g: c.thermal} if getattr(c, "thermal", None) else None
                        T = camera_temperature(s_, cc) if th else None
                        f = thermal_scale(th, g, T, to_T0=False) if th else 1.0
                        ref = copy.copy(c)
                        ref.params = np.array(c.params, float)
                        ref.params[:2] *= f
                        d = compare_cameras(ref.pixel_camera(), cc.pixel_camera(), step=step)
                        per_scape.append({"scape": n, "camera": cc.key, "observations": int(cc.n_obs),
                                          "T_degC": T, "thermal_scale": f,
                                          "rms_px": float(d["rms_px"]), "corner_rms_px": float(d["corner_rms_px"]),
                                          "excluded": any(e["scape"] == n for e in getattr(c, "excluded", []))})
        d = {"model": c.model, "width": int(c.width), "height": int(c.height), "params": [float(v) for v in c.params],
             "param_names": list(PARAM_NAMES[:len(c.params)]),
             "distortion": ("rational: radial (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2); k5 = k6 = 0"
                            if c.distortion == "rational" else c.distortion),
             "free_params": ["k4"] if c.distortion == "rational" else [],
             "pixel_origin": "corner of the first pixel (COLMAP)",
             "source": f"{c.key}: observation-weighted mean of the refined {g} cameras of the scapes listed in "
                       f"verification (notebook 04, write_navcam_consensus); {int(c.n_obs)} observations"
                       + (f"; at T0 {c.thermal['T0_degC']:.1f} degC" if getattr(c, "thermal", None) else ""),
             "verification": {"per_scape": per_scape, "excluded": list(getattr(c, "excluded", [])),
                              "repeatability": repeatability or {}}}
        if getattr(c, "thermal", None):
            d["thermal"] = dict(c.thermal, note="v0p31: the camera is at T0_degC; SfmProject.create scales fx, fy "
                                                "by 1 + ppm_per_degC 1e-6 (T - T0) to the camera temperature of the "
                                                "project's images")
        path = out_dir / f"M2020_{g}_rational.json"
        path.write_text(json.dumps(d, indent=1), encoding="utf-8")
        written[g] = path
    if rig is not None:
        R, t = np.asarray(rig[0], float), np.asarray(rig[1], float)
        per = {}
        if sols:
            for n, s_ in sols.items():
                gg = rig_geometry(s_.rig_refined)
                if gg is not None:
                    per[n] = [float(v) for v in Rotation.from_matrix(gg[0]).as_rotvec()]
        d = {"ref": "NL", "sensor": "NR", "R_sensor_from_ref": R.tolist(),
             "rotvec_rad": [float(v) for v in Rotation.from_matrix(R).as_rotvec()],
             "t_sensor_from_ref_m": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
             "per_solution_rotvec_rad": per,
             "note": "x_NR = R x_NL + t. Rotation: observation-weighted mean of the refined rigs of the solutions listed "
                     "(notebook 04, write_navcam_consensus). SfmProject.create(navcam_cameras=<this folder>) starts the "
                     "rig rotation here and keeps the project's CAHV translation (the baseline sets the scale)."}
        path = out_dir / "M2020_N_rig.json"
        path.write_text(json.dumps(d, indent=1), encoding="utf-8")
        written["rig"] = path
    return written


def reference_camera(group: str, lens: str = "rational") -> Optional[Camera]:
    """The start camera MPPP ships for ``group`` (Navcam rational JSON or Metashape XML).  v0p62: ``lens =
    "consensus"`` (or ``"fisheye_tangential"``) is the Navcam consensus in use (``cmods_dir()``,
    ``M2020_<group>_fisheye_tangential.json``) at its reference temperature, with its ``thermal`` terms attached."""
    from ..paths import cmods_dir, data_dir
    from .project import camera_from_colmap_json, camera_from_metashape_xml
    d = data_dir() / "cmods"
    if group in ("NL", "NR") and lens in ("consensus", "fisheye_tangential"):
        f = cmods_dir() / f"M2020_{group}_fisheye_tangential.json"
        if not f.is_file():
            f = d / f.name
        js = json.loads(f.read_text(encoding="utf-8"))
        p = np.asarray(js["params"], float)
        cam = Camera(f"{group} consensus", group, js["model"], int(js["width"]), int(js["height"]), p, p)
        cam.thermal = js.get("thermal")                      # {"ppm_per_degC", "T0_degC", "cx_px_per_degC", ...}
        cam.source = str(f)
        return cam
    if group in ("NL", "NR"):
        c = (camera_from_colmap_json(d / f"M2020_{group}_rational.json") if lens == "rational"
             else camera_from_metashape_xml(d / f"M2020_{group}0_frame.xml", ("b1", "b2")))
    elif (d / f"{group}_frame.xml").is_file():
        c = camera_from_metashape_xml(d / f"{group}_frame.xml", ("b1", "b2"))
    else:
        return None
    p = np.asarray(c["params"], float)
    return Camera(f"{group} shipped ({lens})", group, c["model"], c["width"], c["height"], p, p)


def reference_differences(sols: Dict[str, Solution], group: str, reference: Camera, step: float = 96.0,
                          min_observations: int = 1000,
                          thermal: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, Any]]:
    """Pixel differences of every refined camera of ``group`` from ``reference`` (rotation removed).  ``thermal``
    (v0p31): the reference is first scaled to each camera's temperature (:func:`thermal_scale`), so the difference
    is what remains after the temperature correction; ``T_degC`` and ``thermal_scale`` are in the rows."""
    rows = []
    ref0 = reference.pixel_camera()
    for n, s in sols.items():
        for c in s.cameras_of(group):
            if not c.refined or c.n_obs < min_observations:
                continue
            T = camera_temperature(s, c) if thermal else None
            f = thermal_scale(thermal, group, T, to_T0=False) if thermal else 1.0
            if f != 1.0:
                r2 = copy.copy(reference)
                r2.params = np.array(reference.params, float)
                r2.params[:2] *= f
                ref = r2.pixel_camera()
            else:
                ref = ref0
            d = compare_cameras(ref, c.pixel_camera(), step=step)
            rows.append({"scape": n, "camera": c.key, "lens": c.distortion, "observations": c.n_obs,
                         "T_degC": T, "thermal_scale": f,
                         **{k: d[k] for k in ("rms_px", "centre_rms_px", "edge_rms_px", "corner_rms_px", "max_px",
                                              "coverage", "rotation_deg")}, "_diff": d})
    return rows


def camera_at_temperature(cam: Camera, T: Optional[float]) -> Camera:
    """v0p62: a copy of a consensus camera moved to temperature ``T`` by its ``thermal`` terms (fx, fy by
    ppm_per_degC; cx, cy by cx_px_per_degC, cy_px_per_degC, all about T0_degC).  Unchanged without either."""
    th = getattr(cam, "thermal", None)
    if not th or T is None:
        return cam
    dT = float(T) - float(th["T0_degC"])
    c = copy.copy(cam)
    c.params = np.array(cam.params, float)
    c.params[:2] *= 1.0 + 1e-6 * float(th.get("ppm_per_degC") or 0.0) * dT
    c.params[2] += float(th.get("cx_px_per_degC") or 0.0) * dT
    c.params[3] += float(th.get("cy_px_per_degC") or 0.0) * dT
    return c


def _main_camera(sol: Solution, group: str, min_observations: int, lens: Optional[str]) -> Optional[Camera]:
    """The refined camera of ``group`` with the most observations (a temperature bin when the block has bins)."""
    cs = [c for c in sol.cameras_of(group) if c.refined and c.n_obs >= min_observations
          and (lens is None or c.distortion == lens)]
    return max(cs, key=lambda c: c.n_obs) if cs else None


def consensus_differences(sols: Dict[str, Solution], group: str, reference: Optional[Camera] = None,
                          min_observations: int = 1000, step: float = 96.0,
                          lens: Optional[str] = "fisheye_tangential") -> List[Dict[str, Any]]:
    """
    v0p62: per scape, its most observed refined ``group`` camera (``lens`` only) minus the consensus in use
    (:func:`reference_camera` ``"consensus"``) moved to that camera's temperature (:func:`camera_at_temperature`),
    rotation removed (:func:`mppp.cmod.compare_cameras`).  What is left is the scape-to-scape variation the consensus
    and its thermal terms do not explain.  Rows as :func:`reference_differences` plus ``reference``.
    """
    ref0 = reference or reference_camera(group, "consensus")
    rows = []
    for n, s in sols.items():
        c = _main_camera(s, group, min_observations, lens)
        if c is None:
            continue
        T = camera_temperature(s, c)
        ref = camera_at_temperature(ref0, T)
        d = compare_cameras(ref.pixel_camera(), c.pixel_camera(), step=step)
        rows.append({"scape": n, "camera": c.key, "lens": c.distortion, "observations": c.n_obs, "T_degC": T,
                     "reference": "consensus", **{k: d[k] for k in ("rms_px", "centre_rms_px", "edge_rms_px",
                                                                      "corner_rms_px", "max_px", "coverage",
                                                                      "rotation_deg")}, "_diff": d})
    return rows


def label_camera(sol: Solution, camera_key: str, drop_e: bool = True) -> Optional[CameraModel]:
    """
    v0p62: the median label model (by focal length) of the images of one refined camera - the labels are
    interpolated to each image's temperature, so a temperature-bin camera gets the labels of its own bin.
    ``drop_e``: E = 0, i.e. the model at infinite range.  E moves the entrance pupil along the axis with field angle,
    which only matters for points close to the camera; the fisheye mapping of a type-2/3 CAHVORE is its linearity
    parameter, not E, and stays.  O and R carry the distortion (they overlap with k1, k2, k3, p1, p2, s1, s2).
    """
    from dataclasses import replace
    lm = label_models(sol)
    ms = [lm[n] for n, r in sol.images.items() if n in lm and r["instrument"] == camera_key]
    if not ms:
        return None
    hs = [m.decompose()[1]["hs"] for m in ms]
    m = ms[int(np.argsort(hs)[len(hs) // 2])]
    if drop_e and m.E is not None:
        m = replace(m, E=np.zeros(3))
    return m


def label_differences(sols: Dict[str, Solution], group: str, min_observations: int = 1000, step: float = 96.0,
                      drop_e: bool = True, lens: Optional[str] = "fisheye_tangential") -> List[Dict[str, Any]]:
    """
    v0p62: per scape, its most observed refined ``group`` camera minus the PDS label model of the same images
    (:func:`label_camera`, E = 0 with ``drop_e``), rotation removed.  ``e_effect_1m_px``: rms over the frame of what
    E does to a point 1 m away (it falls with 1/range), for the record of why E can be dropped.
    """
    rows = []
    for n, s in sols.items():
        c = _main_camera(s, group, min_observations, lens)
        if c is None:
            continue
        lab = label_camera(s, c.key, drop_e=drop_e)
        if lab is None:
            continue
        w, h = c.width, c.height
        d = compare_cameras(PixelCamera.cahv(lab, w, h, as_is=True), c.pixel_camera(), step=step)
        e1 = None
        if drop_e:
            full = label_camera(s, c.key, drop_e=False)
            if full is not None and full.E is not None and np.any(full.E != 0):
                e1 = compare_cameras(PixelCamera.cahv(full, w, h, as_is=True), PixelCamera.cahv(lab, w, h, as_is=True),
                                     step=4 * step)["rms_px"]
        rows.append({"scape": n, "camera": c.key, "lens": c.distortion, "observations": c.n_obs,
                     "T_degC": camera_temperature(s, c), "reference": f"label {lab.kind}" + (" (E = 0)" if drop_e else ""),
                     "e_effect_1m_px": e1,
                     **{k: d[k] for k in ("rms_px", "centre_rms_px", "edge_rms_px", "corner_rms_px", "max_px",
                                          "coverage", "rotation_deg")}, "_diff": d})
    return rows


def _grid_image(uv: np.ndarray, v: np.ndarray):
    xs, ys = np.unique(np.round(uv[:, 0], 3)), np.unique(np.round(uv[:, 1], 3))
    G = np.full((ys.size, xs.size), np.nan)
    G[np.searchsorted(ys, np.round(uv[:, 1], 3)), np.searchsorted(xs, np.round(uv[:, 0], 3))] = v
    dx = (xs[-1] - xs[0]) / max(1, xs.size - 1)
    dy = (ys[-1] - ys[0]) / max(1, ys.size - 1)
    return G, (xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] + dy / 2, ys[0] - dy / 2)


def difference_maps_figure(rows: Sequence[Dict[str, Any]], title: str, vmax: Optional[float] = None,
                           cols: int = 4, arrows: bool = True):
    """
    v0p62: one panel per row of :func:`consensus_differences` / :func:`label_differences` /
    :func:`reference_differences`: |difference| in px as an image on one colour scale for all panels (``vmax``,
    default the 98th percentile over all panels), with the difference vectors as arrows on a coarse grid
    (``arrows``; one scale for all panels: a difference of ``vmax`` is 8 % of the frame width).  Returns the figure.
    """
    import matplotlib.pyplot as plt
    rows = list(rows)
    n = max(1, len(rows))
    cols = max(1, min(cols, n))
    if vmax is None:
        allv = np.concatenate([np.asarray(r["_diff"]["norm_px"])[np.isfinite(r["_diff"]["norm_px"])] for r in rows]) \
            if rows else np.array([1.0])
        vmax = float(np.percentile(allv, 98)) if allv.size else 1.0
    nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(4.2 * cols, 3.4 * nr), squeeze=False)
    im = None
    for ax, r in zip(axes.ravel(), rows):
        d = r["_diff"]
        uv = np.asarray(d["uv"], float)
        W, H = float(uv[:, 0].max() + uv[:, 0].min()), float(uv[:, 1].max() + uv[:, 1].min())   # the grid is centred
        G, ext = _grid_image(uv, np.asarray(d["norm_px"], float))
        im = ax.imshow(G, extent=ext, origin="upper", cmap="viridis", vmin=0.0, vmax=vmax, interpolation="bilinear")
        if arrows:
            dv = np.asarray(d["diff_px"], float)
            xs, ys = np.unique(np.round(uv[:, 0], 3)), np.unique(np.round(uv[:, 1], 3))
            keep = np.isin(np.round(uv[:, 0], 3), xs[::6]) & np.isin(np.round(uv[:, 1], 3), ys[::6]) & np.all(np.isfinite(dv), axis=1)
            keep &= np.linalg.norm(np.nan_to_num(dv), axis=1) > 0.03 * vmax        # no arrows on noise
            if keep.any():
                s = 0.08 * W / vmax                       # one arrow scale for all panels: vmax = 8 % of the width
                ax.quiver(uv[keep, 0], uv[keep, 1], dv[keep, 0] * s, dv[keep, 1] * s, angles="xy", scale_units="xy",
                          scale=1, color="w", width=0.004, headwidth=3)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
        T = r.get("T_degC")
        ax.set_title(f"{r['scape']} {r['camera']}" + (f" ({T:.0f} °C)" if T is not None else "")
                     + f"\nrms {r['rms_px']:.2f} px, centre {r['centre_rms_px']:.2f}, corners {r['corner_rms_px']:.2f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    if im is not None:
        cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="|difference| [px], all panels")
    return fig


def radial_profile(cam: Camera, n: int = 200, towards: str = "corner") -> Dict[str, np.ndarray]:
    """Image radius [px] against field angle along the ray towards the far corner (or the
    horizontal edge), and the departure from a pinhole (f tan theta) and an equidistant
    fisheye (f theta), with f = sqrt(fx fy)."""
    from ..colmap import project_camera
    p = cam.params
    f = float(np.sqrt(p[0] * p[1]))
    tx = np.array([cam.width, cam.height if towards == "corner" else p[3]]) - p[2:4]
    tx = tx / np.linalg.norm(tx)
    th = np.linspace(0, np.radians(75), n)
    X = np.c_[np.sin(th) * tx[0], np.sin(th) * tx[1], np.cos(th)]
    uv = project_camera(cam.model, p, X)
    r = np.linalg.norm(uv - p[2:4], axis=1)
    # the frame limit along this direction
    lim = np.linalg.norm(np.array([cam.width, cam.height]) - p[2:4]) if towards == "corner" else cam.width - p[2]
    ok = np.r_[True, np.diff(r) > 0] & (r <= lim * 1.02)
    ok = np.cumprod(ok).astype(bool)                       # stop at the first turning point
    return {"theta_deg": np.degrees(th[ok]), "r_px": r[ok], "pinhole_px": (f * np.tan(th))[ok],
            "fisheye_px": (f * th)[ok], "limit_px": lim, "f_px": f}


# ================================================================ stereo geometry
def _ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """Small rotation in the left-camera frame as (yaw about y, pitch about x, roll about z) [deg]."""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(R).as_rotvec()
    return float(np.degrees(rv[1])), float(np.degrees(rv[0])), float(np.degrees(rv[2]))


def stereo_pairs(sol: Solution, family: str = "N") -> List[Dict[str, Any]]:
    """
    Every simultaneous left/right exposure (same SCLK) of ``family``: the right
    camera's pose relative to the left one, refined and prior (label), with the
    difference as yaw (about the left y axis: shifts disparity), pitch (about x:
    vertical parallax) and roll (about z), and the baseline.
    """
    by: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in sol.images.values():
        if str(r["instrument"])[:1] != family or "R_w2c" not in r:
            continue
        by.setdefault(r["sclk_key"], {})[r["eye"]] = r
    rows = []
    for k, d in sorted(by.items()):
        if "L" not in d or "R" not in d:
            continue
        L, R = d["L"], d["R"]
        Rrel = R["R_w2c"] @ L["R_w2c"].T
        t = R["R_w2c"] @ (L["C"] - R["C"])                  # x_R = Rrel x_L + t
        R0 = np.asarray(R["prior_R_w2c"]) @ np.asarray(L["prior_R_w2c"]).T
        t0 = np.asarray(R["prior_R_w2c"]) @ (np.asarray(L["prior_C"]) - np.asarray(R["prior_C"]))
        dy, dp, dr = _ypr(Rrel @ R0.T)
        y, p, rr = _ypr(Rrel)
        rows.append({"sclk": k, "station": L["station"], "left": L["name"], "right": R["name"],
                     "left_camera": L["instrument"], "right_camera": R["instrument"],
                     "baseline_m": float(np.linalg.norm(t)), "baseline_prior_m": float(np.linalg.norm(t0)),
                     "yaw_deg": y, "pitch_deg": p, "roll_deg": rr,
                     "dyaw_mdeg": 1e3 * dy, "dpitch_mdeg": 1e3 * dp, "droll_mdeg": 1e3 * dr,
                     "R_rel": Rrel, "t_rel": t, "R_rel_prior": R0, "t_rel_prior": t0})
    return rows


def rig_geometry(rig: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(R_sensor_from_ref, t) of the first rig in a summary's ``rig_initial`` / ``rig_refined`` (notebook 03
    builds one rig, the Navcam pair; Mastcam-Z has none unless ``zcam_rig``)."""
    if not rig:
        return None
    # several rigs (v0p22: a Mastcam-Z pair that shares a clock also becomes a frame with a rig): the
    # Navcam one is the "N" entry, or the one with the longest baseline (0.424 m against 0.243 m)
    r = rig.get("N") or max(rig.values(), key=lambda x: float(x.get("baseline_m", np.linalg.norm(
        np.asarray(x.get("t_sensor_from_ref", x.get("t")), float)))))
    R = np.asarray(r.get("R_sensor_from_ref", r.get("R")), float)
    t = np.asarray(r.get("t_sensor_from_ref", r.get("t")), float)
    return R, t


def stereo_effect(left_a: PixelCamera, right_a: PixelCamera, rel_a: Tuple[np.ndarray, np.ndarray],
                  left_b: PixelCamera, right_b: PixelCamera, rel_b: Tuple[np.ndarray, np.ndarray],
                  ranges_m: Sequence[float] = (2, 5, 10, 20, 50), step: float = 128.0) -> List[Dict[str, Any]]:
    """
    Disparity error of stereo processing with geometry ``a`` when the truth is
    ``b``.  Points at each range (depth along the left optical axis) through a
    pixel grid of the left camera ``b`` are projected into both cameras with
    both geometries; the rotation of the left camera that a pose absorbs is
    removed first.  Returns per range the disparity error (disparity
    d = x_left - x_right; error = d_a - d_b; mean = bias, and spread), the
    vertical parallax (y), and the resulting range error at the image centre,
    dZ/Z = Z (d_a - d_b) / (f B): positive = processing with ``a`` puts points
    too far away.  Pixels where any of the
    four projections is untrustworthy (outside the frame, or a polynomial past
    its turning point) are left out (``coverage``).
    """
    uv = pixel_grid(left_b.width, left_b.height, step)
    db = left_b.rays(uv)
    da = left_a.rays(uv)
    ok = np.all(np.isfinite(db), axis=1) & np.all(np.isfinite(da), axis=1)
    uv, db, da = uv[ok], db[ok], da[ok]
    from ..cmod import _best_rotation
    Q = _best_rotation(db, da)                      # left b frame -> left a frame
    Ra, ta = rel_a
    Rb, tb = rel_b
    fb = float(np.median(np.abs(np.diff(left_b.project(np.array([[0, 0, 1.0], [1e-3, 0, 1.0]])), axis=0))[:, 0]) / 1e-3)
    B = float(np.linalg.norm(tb))
    ctr = np.linalg.norm(uv - np.array([left_b.width, left_b.height]) / 2, axis=1)
    c_sel = ctr < 0.15 * np.hypot(left_b.width, left_b.height)
    out = []
    for Z in ranges_m:
        Xb = db / db[:, 2:3] * Z
        Xa = Xb @ Q.T
        XRa, XRb = Xa @ Ra.T + ta, Xb @ Rb.T + tb
        pla, plb, pra, prb = left_a.project(Xa), left_b.project(Xb), right_a.project(XRa), right_b.project(XRb)
        good = left_a.valid(Xa, pla) & left_b.valid(Xb, plb) & right_a.valid(XRa, pra) & right_b.valid(XRb, prb)
        e = (pra - prb) - (pla - plb)                # right-minus-left image error of a against b
        e[~good] = np.nan
        dx, dy = -e[:, 0], e[:, 1]                  # disparity d = x_left - x_right: error d_a - d_b
        dxc = float(np.nanmean(dx[c_sel])) if np.any(c_sel & good) else float(np.nanmean(dx))
        out.append({"range_m": float(Z), "coverage": float(good.mean()), "disparity_bias_px": float(np.nanmean(dx)),
                    "disparity_sd_px": float(np.nanstd(dx)), "disparity_max_px": float(np.nanmax(np.abs(dx))),
                    "vertical_parallax_rms_px": float(np.sqrt(np.nanmean(dy ** 2))),
                    "vertical_parallax_max_px": float(np.nanmax(np.abs(dy))),
                    "centre_disparity_px": dxc, "range_error_centre_pct": float(100 * Z * dxc / (fb * B)),
                    "range_error_centre_m": float(Z * Z * dxc / (fb * B)), "f_px": fb, "baseline_m": B})
    return out


def navcam_stereo(sol: Solution, which: str = "refined") -> Optional[Tuple[PixelCamera, PixelCamera, Tuple[np.ndarray, np.ndarray]]]:
    """
    (left, right, (R, t)) of the Navcam pair of a solution; ``which``:
    ``"refined"`` (bundle-adjusted cameras and rig), ``"initial"`` (the start
    cameras and the rig from the label CAHV poses) or ``"label"`` (the label
    CAHVORE models and the label rig).
    """
    L, R = sol.cameras.get("NL"), sol.cameras.get("NR")
    # v0p31: with temperature bins the eye's images may all belong to bin cameras - take the most observed one
    if which == "refined":
        L = max(sol.cameras_of("NL"), key=lambda c: c.n_obs, default=L) if (L is None or L.n_obs == 0) else L
        R = max(sol.cameras_of("NR"), key=lambda c: c.n_obs, default=R) if (R is None or R.n_obs == 0) else R
    rig = rig_geometry(sol.rig_refined if which == "refined" else sol.rig_initial)
    if L is None or R is None or rig is None:
        return None
    if which == "label":
        ml, mr = median_label_model(sol, "NL"), median_label_model(sol, "NR")
        if ml is None or mr is None:
            return None
        return (PixelCamera.cahv(ml, L.width, L.height, "NL label", as_is=True),
                PixelCamera.cahv(mr, R.width, R.height, "NR label", as_is=True), rig)
    return L.pixel_camera(which == "initial"), R.pixel_camera(which == "initial"), rig


def mean_rig(sols: Dict[str, Solution], lens: Optional[str] = None,
             max_dev_mdeg: Optional[float] = 50.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Observation-weighted mean refined Navcam rig (rotation vector and translation averaged).
    ``max_dev_mdeg`` (v0p22.1): rigs farther than this from the median rotation are left out (a scape
    whose Navcam intrinsics are not observable also turns its rig; methods section 14)."""
    from scipy.spatial.transform import Rotation
    rv, tt, w = [], [], []
    for s in sols.values():
        g = rig_geometry(s.rig_refined)
        if g is None or (lens and s.navcam_distortion != lens):
            continue
        rv.append(Rotation.from_matrix(g[0]).as_rotvec())
        tt.append(g[1])
        w.append(sum(c.n_obs for c in s.cameras.values() if c.family == "N"))
    if not rv:
        return None
    rv, tt, w = np.asarray(rv), np.asarray(tt), np.asarray(w, float)
    if max_dev_mdeg and len(rv) > 2:
        med = np.median(rv, axis=0)
        dev = np.degrees(np.linalg.norm(rv - med, axis=1)) * 1e3
        keep = dev <= max_dev_mdeg
        if keep.sum() >= 2:
            rv, tt, w = rv[keep], tt[keep], w[keep]
    return Rotation.from_rotvec(np.average(rv, axis=0, weights=w)).as_matrix(), np.average(tt, axis=0, weights=w)


def disparity_to_range_error(disparity_px: float, range_m: float, f_px: float, baseline_m: float) -> float:
    """Range error [m] of a disparity error at a range: dZ = Z^2 dd / (f B)."""
    return range_m ** 2 * disparity_px / (f_px * baseline_m)


def zcam_pair_effects(sol: Solution, ranges_m: Sequence[float] = (5, 10, 20, 50), min_observations: int = 300,
                      reference: str = "label") -> List[Dict[str, Any]]:
    """
    Per Mastcam-Z stereo pair with at least ``min_observations`` in each image:
    the disparity error at the image centre (and its range error) of processing
    the pair with ``reference`` geometry when the refined one is the truth.
    ``reference``: ``"label"`` (the bin's start cameras = label median, and the
    label relative pose) or ``"median"`` (the pair's refined cameras with the
    median refined relative pose of all such pairs - how well one fixed
    Mastcam-Z rig would do).
    """
    from scipy.spatial.transform import Rotation
    pairs = [p for p in stereo_pairs(sol, "Z")
             if min(float(sol.images[p["left"]].get("observations") or 0),
                    float(sol.images[p["right"]].get("observations") or 0)) >= min_observations]
    if not pairs:
        return []
    if reference == "median":
        Rm = Rotation.from_rotvec(np.median([Rotation.from_matrix(p["R_rel"]).as_rotvec() for p in pairs], axis=0)).as_matrix()
        tm = np.median([p["t_rel"] for p in pairs], axis=0)
    rows = []
    for p in pairs:
        L, R = sol.cameras[p["left_camera"]], sol.cameras[p["right_camera"]]
        b = (L.pixel_camera(), R.pixel_camera(), (p["R_rel"], p["t_rel"]))
        if reference == "label":
            a = (L.pixel_camera(True), R.pixel_camera(True), (p["R_rel_prior"], p["t_rel_prior"]))
        else:
            a = (L.pixel_camera(), R.pixel_camera(), (Rm, tm))
        e = stereo_effect(*a, *b, ranges_m=ranges_m, step=103.0)
        row = {k: p[k] for k in ("sclk", "station", "left", "right", "left_camera", "right_camera", "baseline_m",
                                 "dyaw_mdeg", "dpitch_mdeg", "droll_mdeg")}
        if reference == "median":                          # rotation differences from the median rig instead
            dy, dp, dr = _ypr(p["R_rel"] @ Rm.T)
            row.update(dyaw_mdeg=1e3 * dy, dpitch_mdeg=1e3 * dp, droll_mdeg=1e3 * dr)
        row["focus_left"], row["focus_right"] = L.focus, R.focus
        for r in e:
            z = int(r["range_m"])
            row[f"disparity_px_{z}m"] = r["centre_disparity_px"]
            row[f"range_error_pct_{z}m"] = r["range_error_centre_pct"]
            row[f"vertical_parallax_px_{z}m"] = r["vertical_parallax_rms_px"]
        rows.append(row)
    return rows


# ================================================================ Mastcam-Z focus
def focus_table(sols: Dict[str, Solution], min_observations: int = 0) -> List[Dict[str, Any]]:
    """One row per Mastcam-Z focus-bin camera and scape: focus count, focal lengths (refined,
    start = label median of the bin, per-image label spread), images, observations."""
    rows = []
    for n, s in sols.items():
        nscale = navcam_focal_scale(s)
        for c in s.cameras.values():
            if c.family != "Z" or c.n_obs < min_observations:
                continue
            mem = [r for r in s.images.values() if r["instrument"] == c.key]
            lab = [float(r["label_f_px"]) for r in mem if r.get("label_f_px") is not None]
            # v0p42: the bin's median sol and focal-plane temperature (HEAD_FPA; manifest or project record), and
            # the block's Navcam focal scale (the Mastcam-Z focal lengths follow it through the shared points)
            sl = [float(r["sol"]) for r in mem if r.get("sol") is not None]
            T = camera_temperature(s, c)
            rows.append({"scape": n, "camera": c.key, "group": c.group, "focus": c.focus, "images": c.n_images,
                         "sol": float(np.median(sl)) if sl else None, "temperature_degC": T, "navcam_scale": nscale,
                         # v0p40: the focus state of the camera (mppp.sfm.backlash: "<bin>_reg" = regular)
                         "state": "regular" if str(c.key).endswith("_reg") else "backlash",
                         "observations": c.n_obs, "refined": c.refined,
                         "f_refined_px": float(np.sqrt(c.params[0] * c.params[1])),
                         "fx_refined_px": float(c.params[0]), "fy_refined_px": float(c.params[1]),
                         "f_start_px": float(np.sqrt(c.initial[0] * c.initial[1])),
                         "f_label_median_px": float(np.median(lab)) if lab else None,
                         "f_label_spread_px": float(np.ptp(lab)) if len(lab) > 1 else 0.0,
                         "aspect": float(c.params[1] / c.params[0]), "cx": float(c.params[2]), "cy": float(c.params[3])})
    return sorted(rows, key=lambda r: (r["group"], r["scape"], r["focus"] if r["focus"] is not None else 0))


def navcam_focal_scale(sol: Solution) -> Optional[float]:
    """v0p42: the block's Navcam focal scale, refined / start focal length averaged over the Navcam cameras with
    observations (1.0 when they were held); None without Navcam cameras.  A Mastcam-Z focus bin of a Navcam +
    Mastcam-Z block inherits this scale through the shared 3-D points (Three Forks, MPPP 0.22: +0.35 % in both)."""
    # v0p53 (audit item 1): a temperature-bin camera (NL_T-030-020) starts from the refined eye camera, so its ratio
    # is multiplied by the eye's refined / start ratio recorded before the thermal split
    pre = ((sol.project or {}).get("settings") or {}).get("navcam_focal_scale_pre_thermal") or {}
    r = []
    for c in sol.cameras.values():
        if c.family != "N" or c.n_obs <= 0:
            continue
        x = float(np.sqrt(c.params[0] * c.params[1]) / np.sqrt(c.initial[0] * c.initial[1]))
        if "_T" in str(c.key):
            x *= float(pre.get(str(c.key).split("_T")[0], 1.0))
        r.append(x)
    return float(np.mean(r)) if r else None


def fit_focus_model(rows: List[Dict[str, Any]], group: str, min_observations: int = 2000,
                    per_scape_offset: bool = True, min_focus: Optional[float] = -2000.0,
                    state: Optional[str] = "backlash", thermal: bool = False, trend: bool = False,
                    navcam_normalise: bool = False, T0: float = -15.0, sol0: float = 700.0,
                    reference_focus: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    f = f0 + a (focus - ref) [+ a constant per scape] fitted to the refined bins of
    ``group`` with at least ``min_observations`` (weights = observations).  The
    per-scape offsets measure how repeatable the focal length is from scape to
    scape at the same focus (temperature, zoom repeatability, the bundle adjustment).
    Bins below ``min_focus`` motor counts (default -2000) are left out, of the label fit too:
    there are few of them and they scatter far from the line.  ``state`` (v0p40): only the bins of that focus
    state (``"backlash"``, the dominant state; the regular-state ``<bin>_reg`` cameras sit about 1 % lower, at the
    label, and are summarised in ``regular_state``); None: all bins.

    v0p42: ``thermal`` adds b (T - ``T0``) with the bin's focal-plane temperature (HEAD_FPA, ``temperature_degC``;
    bins without one are left out), ``trend`` adds c (sol - ``sol0``) (use without ``per_scape_offset``: one scape's
    offset is its sol), ``navcam_normalise`` divides each bin's focal length by its block's Navcam focal scale
    (``navcam_scale``) first, ``reference_focus`` fixes ref (default: the weighted mean focus).  The result then
    carries ``thermal`` and ``trend`` dicts in the layout of the shipped focus model.
    """
    lo = -np.inf if min_focus is None else float(min_focus)
    use = [r for r in rows if r["group"] == group and r["refined"] and r["observations"] >= min_observations
           and r["focus"] is not None and r["focus"] >= lo and (state is None or r.get("state", "backlash") == state)]
    # v0p40: the regular focus state (f about the label) is left out of the line and reported as its offset
    reg = [r for r in rows if r["group"] == group and r.get("state") == "regular" and r["refined"]
           and r.get("f_label_median_px")]
    if thermal:
        use = [r for r in use if r.get("temperature_degC") is not None]
    if trend:
        use = [r for r in use if r.get("sol") is not None]
    if len(use) < 3 + int(thermal) + int(trend):
        return None
    x = np.array([r["focus"] for r in use], float)
    y = np.array([r["f_refined_px"] for r in use], float)
    if navcam_normalise:
        y = y / np.array([float(r.get("navcam_scale") or 1.0) for r in use])
    w = np.array([r["observations"] for r in use], float)
    ref = float(np.average(x, weights=w)) if reference_focus is None else float(reference_focus)
    scapes = sorted({r["scape"] for r in use})
    cols = [np.ones_like(x), x - ref]
    extra = []
    if thermal:
        cols.append(np.array([float(r["temperature_degC"]) - T0 for r in use]))
        extra.append("thermal")
    if trend:
        cols.append(np.array([float(r["sol"]) - sol0 for r in use]))
        extra.append("trend")
    n_fixed = len(cols)
    if per_scape_offset and len(scapes) > 1:
        for s in scapes[1:]:
            cols.append(np.array([1.0 if r["scape"] == s else 0.0 for r in use]))
    X = np.stack(cols, axis=1)
    W = w / w.sum()
    beta = np.linalg.lstsq(X * np.sqrt(W)[:, None], y * np.sqrt(W), rcond=None)[0]
    res = y - X @ beta
    # standard errors (weights as relative precisions, scaled by the weighted residual variance)
    try:
        Wn = W * len(y)                                   # relative weights, mean 1
        s2 = float(np.sum(Wn * res ** 2)) / max(len(y) - X.shape[1], 1)
        se = np.sqrt(np.clip(np.diag(np.linalg.inv((X * Wn[:, None]).T @ X) * s2), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    terms = {}
    for i, k in enumerate(extra):
        v, e = float(beta[2 + i]), float(se[2 + i])
        if k == "thermal":
            terms["thermal"] = {"T0_degC": float(T0), "sensor": "HEAD_FPA", "f_px_per_degC": v, "sd_px_per_degC": e,
                                "T_range_degC": [float(min(r["temperature_degC"] for r in use)),
                                                 float(max(r["temperature_degC"] for r in use))]}
        else:
            terms["trend"] = {"sol0": float(sol0), "f_px_per_sol": v, "sd_px_per_sol": e,
                              "sol_range": [float(min(r["sol"] for r in use)), float(max(r["sol"] for r in use))]}
    offsets = {scapes[0]: 0.0}
    for i, s in enumerate(scapes[1:]):
        offsets[s] = float(beta[n_fixed + i]) if per_scape_offset and len(beta) > n_fixed + i else 0.0
    mean_off = np.average(list(offsets.values()), weights=[sum(r["observations"] for r in use if r["scape"] == s)
                                                            for s in offsets])
    offsets = {s: v - mean_off for s, v in offsets.items()}
    lab = [r for r in rows if r["group"] == group and r.get("f_label_median_px") is not None and r["focus"] is not None
           and r["focus"] >= lo]
    lab_fit = None
    if len(lab) >= 3:
        xl = np.array([r["focus"] for r in lab]); yl = np.array([r["f_label_median_px"] for r in lab])
        bl = np.polyfit(xl - ref, yl, 1)
        lab_fit = {"f0_px": float(bl[1]), "slope_px_per_count": float(bl[0])}
    return {"group": group, "reference_focus": ref, "f0_px": float(beta[0] + mean_off),
            "slope_px_per_count": float(beta[1]), "slope_sd_px_per_count": float(se[1]),
            "slope_pct_per_1000": float(1e5 * beta[1] / beta[0]), **terms, "navcam_normalised": bool(navcam_normalise),
            "aspect": (float(np.average([r["fy_refined_px"] / r["fx_refined_px"] for r in use], weights=w))
                       if all(r.get("fx_refined_px") and r.get("fy_refined_px") for r in use) else 1.0),
            "rms_px": float(np.sqrt(np.sum(W * res ** 2))), "scape_offsets_px": offsets,
            "scape_offset_sd_px": float(np.std(list(offsets.values()))) if len(offsets) > 1 else 0.0,
            "n_bins": len(use), "scapes": scapes, "label": lab_fit, "min_focus": min_focus, "state": state,
            "regular_state": {"bins": len(reg), "f_over_label_median": float(np.median(
                [r["f_refined_px"] / r["f_label_median_px"] for r in reg])) if reg else None},
            "focus_range": [float(x.min()), float(x.max())]}


def zcam_boresight_table(sols: Dict[str, Solution], min_observations: int = 100) -> List[Dict[str, Any]]:
    """
    v0p42: per simultaneous Mastcam-Z stereo pair, the right-minus-left **equivalent boresight**: where the left
    principal point's ray lands in the right image, relative to the left principal point (px, parallax left out),
    ``eqx = (cxR - cxL) + fR yaw``, ``eqy = (cyR - cyL) - fR pitch`` (yaw, pitch of the refined right-from-left
    rotation), and the roll.  With the narrow Mastcam-Z field a principal point and the pointing trade, so only this
    difference is observable; against focus it measures how the two eyes' principal points move apart.
    """
    rows = []
    for n, s in sols.items():
        for p in stereo_pairs(s, "Z"):
            L, R = s.images[p["left"]], s.images[p["right"]]
            if min(float(L.get("observations") or 0), float(R.get("observations") or 0)) < min_observations:
                continue
            cL, cR = s.cameras[p["left_camera"]], s.cameras[p["right_camera"]]
            fR = float(np.sqrt(cR.params[0] * cR.params[1]))
            eqx = float(cR.params[2] - cL.params[2]) + fR * np.radians(p["yaw_deg"])
            eqy = float(cR.params[3] - cL.params[3]) - fR * np.radians(p["pitch_deg"])
            TL, TR = L.get("camera_temperature_degC"), R.get("camera_temperature_degC")
            rows.append({"scape": n, "sclk": p["sclk"], "sol": L.get("sol"), "focus_left": cL.focus,
                         "zoom": int(str(cL.group)[2:5]) if str(cL.group)[2:5].isdigit() else None,   # v0p53
                         "focus_right": cR.focus, "left_camera": cL.key, "right_camera": cR.key,
                         "eqx_px": float(eqx), "eqy_px": float(eqy), "roll_mdeg": 1e3 * float(p["roll_deg"]),
                         "temperature_degC": (0.5 * (float(TL) + float(TR)) if TL is not None and TR is not None
                                              else None), "baseline_m": p["baseline_m"]})
    return rows


def fit_zcam_boresight(rows: List[Dict[str, Any]], per_scape_offset: bool = True, thermal: bool = False,
                       T0: float = -15.0, reference_focus: float = 600.0, huber_k: float = 1.345,
                       bootstrap: int = 0, seed: int = 0) -> Dict[str, Any]:
    """v0p42: eqx, eqy (px) and roll (mdeg) of :func:`zcam_boresight_table` against the pair's mean focus
    (per count), with a constant per scape (``per_scape_offset``) and optionally the temperature; Huber IRLS
    (``huber_k`` robust sigmas; the eqy slope moved between 0.9 and 1.9 px per 1000 counts with ad-hoc sigma
    clipping).  ``bootstrap`` > 0 adds a bootstrap standard error of the slope (``sd_boot``; pairs of one sequence
    are correlated, so it is about twice the formal one).  Returns {quantity: {"slope_per_count", "sd", ...}}."""
    use = [r for r in rows if r["focus_left"] is not None and r["focus_right"] is not None
           and (not thermal or r.get("temperature_degC") is not None)]
    out: Dict[str, Any] = {"pairs": len(use)}
    if len(use) < 5:
        return out
    F = np.array([0.5 * (r["focus_left"] + r["focus_right"]) for r in use]) - reference_focus
    scapes = sorted({r["scape"] for r in use})
    cols = [np.array([1.0 if r["scape"] == s else 0.0 for r in use]) for s in scapes] if per_scape_offset \
        else [np.ones(len(use))]
    cols.append(F)
    if thermal:
        cols.append(np.array([float(r["temperature_degC"]) - T0 for r in use]))
    X = np.stack(cols, axis=1)
    k = len(cols) - 1 - int(thermal)

    def _huber(y, X, it=40):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        w = np.ones(len(y))
        for _ in range(it):
            r = y - X @ b
            sig = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-9)
            u = np.abs(r) / (huber_k * sig)
            w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
            b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        r = y - X @ b
        sig = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        cov = np.linalg.pinv((X * w[:, None]).T @ X) * sig ** 2
        return b, np.sqrt(np.clip(np.diag(cov), 0, None)), r, w, sig

    rng = np.random.default_rng(seed)
    for q in ("eqx_px", "eqy_px", "roll_mdeg"):
        y = np.array([r[q] for r in use], float)
        b, se, r, w, sig = _huber(y, X)
        out[q] = {"slope_per_count": float(b[k]), "sd": float(se[k]), "n": len(y),
                  "n_downweighted": int(np.sum(w < 1.0)), "scatter": float(sig),
                  "offsets": ({s: float(v) for s, v in zip(scapes, b[:len(scapes)])} if per_scape_offset
                              else {"all": float(b[0])})}
        if thermal:
            out[q]["per_degC"], out[q]["sd_per_degC"] = float(b[-1]), float(se[-1])
        if bootstrap:
            bs = []
            for _ in range(int(bootstrap)):
                i = rng.integers(0, len(y), len(y))
                if np.linalg.matrix_rank(X[i]) < X.shape[1]:
                    continue
                bs.append(_huber(y[i], X[i], it=15)[0][k])
            out[q]["sd_boot"] = float(np.std(bs)) if bs else None
    return out


def zcam_shared_terms(sols: Dict[str, "Solution"], zoom: int = 34, min_observations: int = 1000,
                      state: Optional[str] = "backlash") -> Dict[str, Any]:
    """
    v0p53: the terms one Mastcam-Z zoom shares across blocks, for its consensus focus model - per eye
    (``ZL034``, ``ZR034``) the observation-weighted median distortion (FULL_OPENCV k1..k6, p1, p2) and principal
    point (``pp0_px``) of the focus-bin cameras with at least ``min_observations`` (only ``state`` bins: the
    regular-state ``_reg`` cameras are left out by default), and the median Mastcam-Z rig rotation of the blocks
    (``rig``: the ``Z<zoom>`` rig of each block's refined rigs; the translation stays CAHV in every project).  With
    ZCAM_BIN_REFINE = "focal" the distortion and pp are those the blocks held (the label medians); a block run with
    "all" contributes refined ones.
    """
    from scipy.spatial.transform import Rotation
    from .project import FULL_OPENCV_NAMES
    out: Dict[str, Any] = {"zoom": int(zoom), "cameras": {}, "rig": None}
    for eye in ("L", "R"):
        g = f"Z{eye}{int(zoom):03d}"
        P, W, used = [], [], set()
        for n, s in sols.items():
            for c in s.cameras.values():
                if c.group != g or c.n_obs < min_observations or c.model != "FULL_OPENCV":
                    continue
                if state == "backlash" and str(c.key).endswith("_reg"):
                    continue
                P.append(np.asarray(c.params, float))
                W.append(float(c.n_obs))
                used.add(n)
        if not P:
            continue
        P, W = np.array(P), np.array(W)
        med = np.array([_wmedian_cols(P[:, j], W) for j in range(P.shape[1])])
        out["cameras"][g] = {"distortion": {"model": "FULL_OPENCV", "names": list(FULL_OPENCV_NAMES[4:]),
                                            "params": [float(x) for x in med[4:]]},
                             "pp0_px": [float(med[2]), float(med[3])], "n_cameras": int(len(P)),
                             "blocks": sorted(used)}
    rv, blocks = [], []
    for n, s in sols.items():
        r = (s.rig_refined or {}).get(f"Z{int(zoom):03d}")
        if r and (r.get("R_sensor_from_ref") or r.get("R")):
            rv.append(Rotation.from_matrix(np.asarray(r.get("R_sensor_from_ref", r.get("R")), float)).as_rotvec())
            blocks.append(n)
    if rv:
        rv = np.array(rv)
        m = np.median(rv, axis=0)
        out["rig"] = {"R_sensor_from_ref": Rotation.from_rotvec(m).as_matrix().tolist(),
                      "rotvec_rad": m.tolist(),
                      "spread_mdeg": float(np.degrees(np.max(np.linalg.norm(rv - m, axis=1))) * 1e3),
                      "blocks": blocks,
                      "note": f"median refined Z{int(zoom):03d} rig of {len(blocks)} blocks (rotation; translation from CAHV)"}
    return out


def _wmedian_cols(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def focus_model_json(fits: Dict[str, Dict[str, Any]], boresight: Optional[Dict[str, Any]] = None,
                     pp_eye: str = "ZR034", focus_range: Optional[Dict[str, Sequence[float]]] = None,
                     source: str = "", shared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v0p42: a focus-model file (the layout of ``cmods/M2020_ZCAM034_focus_model.json``) from
    :func:`fit_focus_model` results per eye and a :func:`fit_zcam_boresight` result (principal-point slopes, put on
    ``pp_eye``).  v0p53: ``shared`` (:func:`zcam_shared_terms`) adds each eye's distortion and ``pp0_px`` and the
    zoom's ``rig``.  Save it as JSON and pass the path as notebook 03's ``ZCAM_FOCUS_MODEL``
    (``SfmProject.create(zcam_focus_model_file=)``) to use it."""
    cams = {}
    for g, f in fits.items():
        if not f:
            continue
        c = {"f0_px": f["f0_px"], "reference_focus": f["reference_focus"], "slope_px_per_count": f["slope_px_per_count"],
             "aspect": f.get("aspect", 1.0), "fit_rms_px": f["rms_px"],
             "focus_range": list((focus_range or {}).get(g, f["focus_range"]))}
        if f.get("label"):
            c["label_f0_px"] = f["label"]["f0_px"]
            c["label_slope_px_per_count"] = f["label"]["slope_px_per_count"]
        for k in ("thermal", "trend"):
            if f.get(k):
                c[k] = dict(f[k])
        if boresight and "eqx_px" in boresight:
            on = g == pp_eye
            c["pp"] = {"cx_px_per_count": boresight["eqx_px"]["slope_per_count"] if on else 0.0,
                       "cy_px_per_count": boresight["eqy_px"]["slope_per_count"] if on else 0.0}
        sh = ((shared or {}).get("cameras") or {}).get(g)
        if sh:                                   # v0p53: one distortion and an absolute principal point per eye
            c["distortion"] = sh["distortion"]
            c["pp0_px"] = sh["pp0_px"]
        cams[g] = c
    out = {"cameras": cams, "state": "backlash", "pixel_mm": ZCAM_PIXEL_MM,
           "units": "see mppp.sfm.project.zcam_model_focal / zcam_model_pp_shift", "source": source}
    if (shared or {}).get("rig"):
        out["rig"] = shared["rig"]
    if shared and int(shared.get("zoom", 34)) in (110,):
        out["state"] = "single"                  # v0p53: no backlash state at 110 mm
    return out


# ================================================================ label models
def label_model_from_meta(meta: Dict[str, Any]) -> Optional[CameraModel]:
    """
    The label camera model of one image in full-frame pixels and its own camera
    frame, rebuilt from the MPPP manifest (``intrinsics_label``: the CAHV part,
    R1 and R2).  Before v0p21 the manifest did not keep O, E or the CAHVORE type:
    O is taken along A (the labels differ by ~0.1 deg) and Navcam CAHVORE as
    type 2 (fisheye), which is what the M2020 Navcam labels use.  v0p21
    manifests carry the full label model (``camera_model_label``); it is used
    when present.
    """
    s = float(meta["filename"].get("downsample_scale", 1.0))
    pad = meta.get("padding") or {}
    fam = meta["filename"]["family"]
    W, H = FULL_FRAME.get(fam, (None, None))
    if meta.get("camera_model_label"):
        g = meta["camera_model_label"]
        cm = CameraModel.from_label(g)
        cm, _ = cm.camera_frame()
    else:
        il = meta.get("intrinsics_label")
        if not il:
            return None
        K = np.asarray(il["K"], float)
        A = np.array([0.0, 0, 1])
        Hv, Vv = K[0].copy(), np.array([0.0, K[1, 1], K[1, 2]])
        d = il.get("dist_opencv", {})
        src = str(il.get("source", ""))
        cm = CameraModel(np.zeros(3), A, Hv, Vv, A.copy(), np.array([0.0, d.get("k1", 0.0), d.get("k2", 0.0)]),
                         meta={"source": "manifest intrinsics_label (O = A assumed)"})
        if "CAHVORE" in src:
            cm.E, cm.mtype = np.zeros(3), 2
        elif "CAHVOR" not in src:
            cm.O, cm.R = None, None
    return cm.rescaled(s, float(pad.get("left", 0)), float(pad.get("top", 0)), W, H)


def label_models(sol: Solution) -> Dict[str, CameraModel]:
    """PDS stem -> label model (full frame, camera frame) for every image of the solution in the manifest."""
    out = {}
    for r in sol.images.values():
        m = sol.manifest.get(r.get("stem") or Path(r["name"]).stem)
        if m:
            cm = label_model_from_meta(m)
            if cm is not None:
                out[r["name"]] = cm
    return out


def label_model_summary(sol: Solution) -> List[Dict[str, Any]]:
    """Per camera group: the spread of the label models (focal length, principal point, R1, R2)."""
    lm = label_models(sol)
    rows = []
    for g in sol.groups():
        ms = [lm[n] for n, r in sol.images.items() if n in lm and sol.cameras[r["instrument"]].group == g]
        if not ms:
            continue
        dec = [m.decompose()[1] for m in ms]
        hs = np.array([d["hs"] for d in dec]); hc = np.array([d["hc"] for d in dec]); vc = np.array([d["vc"] for d in dec])
        rows.append({"scape": sol.label, "group": g, "images": len(ms), "kind": ms[0].kind,
                     "hs_median": float(np.median(hs)), "hs_spread": float(np.ptp(hs)),
                     "hc_median": float(np.median(hc)), "vc_median": float(np.median(vc)),
                     "hc_spread": float(np.ptp(hc)), "vc_spread": float(np.ptp(vc)),
                     "r1_median": float(np.median([m.R[1] for m in ms])) if ms[0].R is not None else None,
                     "r2_median": float(np.median([m.R[2] for m in ms])) if ms[0].R is not None else None})
    return rows


def median_label_model(sol: Solution, group: str) -> Optional[CameraModel]:
    """The label model of ``group`` whose focal length is the median over the solution's images."""
    lm = label_models(sol)
    ms = [lm[n] for n, r in sol.images.items() if n in lm and sol.cameras[r["instrument"]].group == group]
    if not ms:
        return None
    hs = [m.decompose()[1]["hs"] for m in ms]
    return ms[int(np.argsort(hs)[len(hs) // 2])]


def attach_pds_labels(sol: Solution, pds_dir: PathLike, groups: Optional[Sequence[str]] = None,
                      verbose: bool = True) -> int:
    """
    Read the exact label camera models from the PDS products under ``pds_dir``
    (searched recursively) into the solution's manifest records
    (``camera_model_label``), for manifests written before v0p21.  Returns
    the number of images updated.
    """
    from ..labels import label_get, read_pds
    from ..select import iter_imgs
    want = {Path(r["name"]).stem: r for r in sol.images.values()
            if groups is None or sol.cameras[r["instrument"]].group in groups}
    # v0p30: also the camera temperature the label model was interpolated to, for manifests without it
    # v0p42: and the Mastcam-Z focal-plane temperature (HEAD_FPA) where it is missing
    want = {k: v for k, v in want.items() if k in sol.manifest and (
        not sol.manifest[k].get("camera_model_label") or "camera_temperature_degC" not in sol.manifest[k]
        or (k[:1] == "Z" and sol.manifest[k].get("camera_temperature_degC") is None))}
    n = 0
    if not want:
        if verbose:
            print(f"{sol.label}: label models and camera temperatures already in the manifest")
        return 0
    todo = set(want)
    for fp, fn in iter_imgs(pds_dir):
        if fn.stem not in todo:
            continue
        todo.discard(fn.stem)
        L, _ = read_pds(fp, load_image=False)
        g = label_get(L, "GEOMETRIC_CAMERA_MODEL")
        w, h = int(label_get(L, "IMAGE.LINE_SAMPLES")), int(label_get(L, "IMAGE.LINES"))
        cm = CameraModel.from_label(g, w, h)
        rec = sol.manifest[fn.stem]
        if not rec.get("camera_model_label"):
            rec["camera_model_label"] = dict(cm.to_label_dict(precision=12), width=w, height=h,
                                             frame="ROVER_NAV_FRAME", pixel_origin="centre_of_first_pixel")
        rec["camera_temperature_degC"] = _label_temperature(cm) if fn.stem[:1] != "Z" else _zcam_fpa(L)
        n += 1
        if not todo:
            break
    if verbose:
        print(f"{sol.label}: exact label models for {n} of {len(want)} images from {pds_dir}")
    return n


def _zcam_fpa(L: Any) -> Optional[float]:
    """v0p42: HEAD_FPA (Mastcam-Z focal-plane temperature, degC) of a parsed label."""
    from ..labels import label_get
    names = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE_NAME") or []
    vals = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE") or []
    for k, v in zip(names, vals):
        if str(k) == "HEAD_FPA":
            try:
                return float(getattr(v, "value", v))
            except (TypeError, ValueError):
                return None
    return None


def _label_temperature(cm: CameraModel) -> Optional[float]:
    m = getattr(cm, "meta", None) or {}
    if str(m.get("interpolation") or "").upper() != "TEMPERATURE":
        return None
    try:
        return float(m.get("interpolation_value"))
    except (TypeError, ValueError):
        return None


def camera_temperatures(sols: Dict[str, "Solution"], family: str = "N") -> List[Dict[str, Any]]:
    """
    v0p30: per scape and camera, the camera temperature of its images (the value the label camera model was
    interpolated to: ``camera_temperature_degC`` in the manifest, written by MPPP >= 0.30 or added by
    ``attach_pds_labels``) next to the refined focal lengths - for testing whether the scape-to-scape spread
    of the focal length follows temperature.  Rows without temperatures have ``n_temp = 0``.
    """
    rows = []
    for n, s in sols.items():
        for c in sorted(s.cameras.values(), key=lambda c: c.key):
            if c.family != family or not c.refined:
                continue
            t = []
            for r in s.images.values():
                if r["instrument"] != c.key:
                    continue
                v = (s.manifest.get(Path(r["name"]).stem) or {}).get("camera_temperature_degC")
                if v is None:
                    v = r.get("camera_temperature_degC")                          # v0p31: project record
                if v is not None:
                    t.append(float(v))
            t = np.asarray(t, float)
            named = c.named()
            rows.append({"scape": n, "camera": c.key, "eye": c.group, "images": c.n_images, "n_temp": int(t.size),
                         "observations": int(c.n_obs), "temperature_bin": bool(
                             (s.project.get("cameras", {}).get(c.key) or {}).get("thermal_bin")),
                         "temp_median_degC": float(np.median(t)) if t.size else None,
                         "temp_min_degC": float(t.min()) if t.size else None,
                         "temp_max_degC": float(t.max()) if t.size else None,
                         "fx": float(named["fx"]), "fy": float(named["fy"])})
    return rows


def focal_temperature_fit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Least-squares line fx = a + b (T - 0 degC) per camera over the scapes with temperatures; b in px/degC and ppm/degC."""
    out = {}
    key = "eye" if all("eye" in r for r in rows) else "camera"            # v0p31: temperature-bin cameras by eye
    for cam in sorted({r[key] for r in rows}):
        rr = [r for r in rows if r[key] == cam and r["temp_median_degC"] is not None]
        if len(rr) < 3:
            continue
        T = np.array([r["temp_median_degC"] for r in rr]); f = np.array([r["fx"] for r in rr])
        A = np.column_stack([np.ones_like(T), T])
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        res = f - A @ coef
        dof = max(1, len(rr) - 2)
        cov = np.linalg.inv(A.T @ A) * float(res @ res) / dof
        out[cam] = {"scapes": len(rr), "fx_at_0C": float(coef[0]), "px_per_degC": float(coef[1]),
                    "px_per_degC_sd": float(np.sqrt(cov[1, 1])), "ppm_per_degC": float(coef[1] / coef[0] * 1e6),
                    "residual_rms_px": float(np.sqrt(np.mean(res ** 2))),
                    "corr": float(np.corrcoef(T, f)[0, 1]) if T.std() > 0 else None}
    return out


def merge_temperature_bins(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v0p31: :func:`camera_temperatures` rows with the temperature-bin cameras of a scape merged into one row per
    eye (fx, fy and the median temperature weighted by observations; range over the bins), for the across-scape fit."""
    out, groups = [], {}
    for r in rows:
        if not r.get("temperature_bin"):
            out.append(r)
        else:
            groups.setdefault((r["scape"], r["eye"]), []).append(r)
    for (sc, eye), rr in groups.items():
        rr_t = [r for r in rr if r["temp_median_degC"] is not None]
        w = np.array([max(1, r["observations"]) for r in rr], float)
        wt = np.array([max(1, r["observations"]) for r in rr_t], float)
        out.append({"scape": sc, "camera": eye, "eye": eye, "images": sum(r["images"] for r in rr),
                    "n_temp": sum(r["n_temp"] for r in rr), "observations": int(w.sum()), "temperature_bin": False,
                    "bins": len(rr),
                    "temp_median_degC": float(np.average([r["temp_median_degC"] for r in rr_t], weights=wt)) if rr_t else None,
                    "temp_min_degC": min((r["temp_min_degC"] for r in rr_t), default=None),
                    "temp_max_degC": max((r["temp_max_degC"] for r in rr_t), default=None),
                    "fx": float(np.average([r["fx"] for r in rr], weights=w)),
                    "fy": float(np.average([r["fy"] for r in rr], weights=w))})
    # an eye whose frames were partly binned keeps its unbinned camera as well: merge it in by observations
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in out:
        k = (r["scape"], r.get("eye", r["camera"]))
        if k not in merged:
            merged[k] = r
            continue
        a = merged[k]
        wa, wb = max(1, a["observations"]), max(1, r["observations"])
        ta = [x for x in (a["temp_median_degC"], r["temp_median_degC"]) if x is not None]
        tw = [w for x, w in ((a["temp_median_degC"], wa), (r["temp_median_degC"], wb)) if x is not None]
        merged[k] = dict(a, images=a["images"] + r["images"], n_temp=a["n_temp"] + r["n_temp"],
                         observations=int(wa + wb), fx=(a["fx"] * wa + r["fx"] * wb) / (wa + wb),
                         fy=(a["fy"] * wa + r["fy"] * wb) / (wa + wb),
                         temp_median_degC=float(np.average(ta, weights=tw)) if ta else None,
                         temp_min_degC=min((x for x in (a["temp_min_degC"], r["temp_min_degC"]) if x is not None), default=None),
                         temp_max_degC=max((x for x in (a["temp_max_degC"], r["temp_max_degC"]) if x is not None), default=None))
    return list(merged.values())


def thermal_bin_rows(sols: Dict[str, "Solution"], experiment: Optional[PathLike] = None) -> List[Dict[str, Any]]:
    """
    v0p31: the within-block temperature-bin measurements - one row per scape, eye and bin with ``T_median_degC``,
    ``fx``, ``fy``, ``observations`` - from each solution's thermal stage (``settings["thermal"]`` of project.json;
    only stages whose bins were refined, not held) and, optionally, from the JSON of
    ``studies/experiments/temperature_bins_experiment.py`` (scapes already in ``sols`` are taken from the solution).
    """
    rows: List[Dict[str, Any]] = []
    for n, s in sols.items():
        th = (s.project.get("settings") or {}).get("thermal") or {}
        if th.get("held") or not th.get("rows"):
            continue
        for r in th["rows"]:
            rows.append(dict(r, scape=n, source="thermal stage"))
    if experiment:
        d = json.loads(Path(experiment).read_text(encoding="utf-8"))
        have = {_scape_key(r["scape"]) for r in rows}
        for n, e in (d.get("scapes") or {}).items():
            if _scape_key(n) in have:
                continue
            for r in e.get("rows") or []:
                rows.append(dict(r, scape=n, source="experiment"))
    return rows


def _scape_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum()).replace("colmap", "")


def thermal_model(sols: Dict[str, "Solution"], bin_fit: Optional[Dict[str, Any]] = None,
                  across_fit: Optional[Dict[str, Any]] = None, source: str = "auto",
                  ppm_per_degC: Optional[float] = None, min_observations: int = 1000) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    v0p31: the thermal model for :func:`consensus_camera` and :func:`reference_differences`,
    {eye: {"ppm_per_degC", "T0_degC", "source", "sd_ppm"}}.  ``source``: "within" (the slope of the within-block
    temperature bins, ``bin_fit`` from :func:`mppp.sfm.thermal.fit_focal_temperature`), "across" (the scape-to-scape
    line, ``across_fit`` from :func:`focal_temperature_fit`), "auto" (within if measured for the eye, else across),
    or "fixed" (``ppm_per_degC`` for both eyes).  T0 is the observation-weighted mean temperature of the refined
    cameras, so that the consensus stays where the data are.  None if no slope is available.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for eye in ("NL", "NR"):
        ppm = sd = None
        src = source
        if source == "fixed" and ppm_per_degC is not None:
            ppm = float(ppm_per_degC)
        if source in ("within", "auto") and bin_fit and eye in bin_fit:
            ppm, sd, src = bin_fit[eye]["ppm_per_degC"], bin_fit[eye].get("ppm_sd"), "within"
        if ppm is None and source in ("across", "auto") and across_fit and eye in across_fit:
            f = across_fit[eye]
            ppm, sd, src = f["ppm_per_degC"], f["px_per_degC_sd"] / f["fx_at_0C"] * 1e6, "across"
        if ppm is None:
            continue
        T, w = [], []
        for s in sols.values():
            for c in s.cameras_of(eye):
                t = camera_temperature(s, c)
                if c.refined and c.n_obs >= min_observations and t is not None:
                    T.append(t)
                    w.append(c.n_obs)
        if not T:
            continue
        out[eye] = {"ppm_per_degC": float(ppm), "sd_ppm": None if sd is None else float(sd),
                    "T0_degC": float(np.average(T, weights=w)), "source": src}
    return out or None


def updated_navcam_models(left: Camera, right: Camera, rig: Tuple[np.ndarray, np.ndarray], mtype: int = 2,
                          step: float = 48.0) -> Dict[str, Any]:
    """
    CAHVORE models (``mtype`` 2 = fisheye as in the labels, 3 = general with a
    fitted linearity) of a Navcam pair, in the LEFT camera frame (C_left = 0,
    x right, y down, z forward): the left model is fitted to ``left``, the
    right one to ``right`` and placed by ``rig`` (x_R = R x_L + t).  Returns
    the models, their fit residuals and PDS-style text.
    """
    from ..cmod import fit_to_colmap
    ml, fl = fit_to_colmap(left.model, left.params, left.width, left.height, "CAHVORE", mtype, mtype == 3, step)
    mr_cam, fr = fit_to_colmap(right.model, right.params, right.width, right.height, "CAHVORE", mtype, mtype == 3, step)
    R, t = rig
    mr = mr_cam.in_frame(R.T, -R.T @ t)                 # right camera in the left camera frame
    return {"left": ml, "right": mr, "fit_left": fl, "fit_right": fr,
            "text": "\n".join([f"/* {left.key}: CAHVORE type {ml.mtype}, left-camera frame, full-frame pixels "
                                f"(5120 x 3840, pixel-centre origin); fit rms {fl['rms_px']:.2f} px */", ml.label_text(9),
                                f"/* {right.key}: CAHVORE type {mr.mtype}, left-camera frame; fit rms {fr['rms_px']:.2f} px */",
                                mr.label_text(9)])}


def updated_zcam_models(sols: Dict[str, Solution], fits: Dict[str, Dict[str, Any]],
                        focus_counts: Sequence[float] = (0, 500, 1000)) -> Dict[str, Any]:
    """
    CAHVOR models of each Mastcam-Z eye at the given focus counts, from the
    fitted focal-length line (``fit_focus_model``) and the bins' principal
    point and distortion (held at the label median in the bundle adjustment),
    in each camera's own frame.
    """
    from ..cmod import fit_to_colmap
    out = {}
    for g, f in fits.items():
        if not f:
            continue
        cams = [c for s in sols.values() for c in s.cameras_of(g) if c.refined]
        if not cams:
            continue
        c0 = max(cams, key=lambda c: c.n_obs)
        aspect = float(np.median([c.params[1] / c.params[0] for c in cams if c.n_obs > 1000] or [1.0]))
        rows = []
        for fc in focus_counts:
            fval = f["f0_px"] + f["slope_px_per_count"] * (fc - f["reference_focus"])
            p = c0.params.copy()
            p[0], p[1] = fval / np.sqrt(aspect), fval * np.sqrt(aspect)
            cm, rep = fit_to_colmap(c0.model, p, c0.width, c0.height, "CAHVOR", step=24.0)
            rows.append({"focus": fc, "f_px": fval, "model": cm, "fit_rms_px": rep["rms_px"], "text": cm.label_text(9)})
        out[g] = rows
    return out


# ================================================================ images
def is_full_frame(sol: Solution, name: str) -> Optional[bool]:
    """True if the image covers the whole detector (no sub-frame padding, from the manifest); None if unknown."""
    r = sol.images[name]
    m = sol.manifest.get(r.get("stem") or Path(name).stem)
    if not m:
        return None
    return not any((m.get("padding") or {}).values())


def pick_example(sol: Solution, group: str, full_frame: bool = True, full_resolution: bool = True) -> Optional[str]:
    """
    The image of ``group`` with most observations.  ``full_frame``: prefer images
    that cover the whole detector (Navcam products are often sub-frames, padded
    to the frame by MPPP); ``full_resolution``: prefer downsample 1.  Each
    preference falls back when no image satisfies it.
    """
    cands = [r for r in sol.images.values() if sol.cameras.get(r["instrument"]) is not None
             and sol.cameras[r["instrument"]].group == group]
    if full_frame and any(is_full_frame(sol, r["name"]) for r in cands):
        cands = [r for r in cands if is_full_frame(sol, r["name"])]
    if full_resolution and any(float(r.get("downsample_scale", 1)) == 1.0 for r in cands):
        cands = [r for r in cands if float(r.get("downsample_scale", 1)) == 1.0]
    if not cands:
        return None
    return max(cands, key=lambda r: float(r.get("observations") or 0))["name"]


def load_example(sol: Solution, name: str, image_path: Optional[PathLike] = None,
                 mask_path: Optional[PathLike] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(RGB uint8, mask uint8 with 255 = used) of one image of the solution: the
    project's ``images/`` copy (RGBA: alpha = mask) and ``masks/<name>.png``."""
    import cv2
    p = Path(image_path) if image_path else sol.root / "images" / name
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.dtype != np.uint8:
        im = (im / (65535.0 / 255.0)).round().clip(0, 255).astype(np.uint8)
    mask = None
    if im.ndim == 3 and im.shape[2] == 4:
        mask = np.where(im[..., 3] >= 255, 255, 0).astype(np.uint8)    # v0p51: masked alpha may be > 0
        im = im[..., :3]
    mp = Path(mask_path) if mask_path else sol.root / "masks" / (name + ".png")
    if mask is None and mp.is_file():
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=2)
    return im[..., ::-1].copy(), mask


def radial_limit(cam: Camera, rho_max: float = 6.0) -> float:
    """
    Largest normalised radius rho = tan(field angle) up to which the lens model's
    radial mapping still increases (``inf`` if it does over 0..``rho_max``).  A
    polynomial model folds back beyond it: rays farther out would be drawn from
    pixels that belong to smaller angles.
    """
    p = np.asarray(cam.params, float)
    rho = np.linspace(0.0, rho_max, 60001)
    r2 = rho * rho
    if cam.model == "FULL_OPENCV":
        k1, k2, k3, k4, k5, k6 = p[4], p[5], p[8], p[9], p[10], p[11]
        rad = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
    elif cam.model == "OPENCV":
        rad = 1 + p[4] * r2 + p[5] * r2 ** 2
    elif cam.model in ("SIMPLE_RADIAL", "RADIAL"):
        rad = 1 + p[3] * r2 + (p[4] * r2 ** 2 if cam.model == "RADIAL" else 0.0)
    else:
        return float("inf")
    rd = rho * rad
    bad = np.flatnonzero(np.diff(rd) <= 0)
    return float(rho[bad[0]]) if bad.size else float("inf")


def undistort(image: np.ndarray, cam: Camera, mask: Optional[np.ndarray] = None, fit: str = "width",
              scale: Optional[float] = None) -> Dict[str, Any]:
    """
    Resample ``image`` (full frame at any downsample scale) to a pinhole camera
    of the same size and principal point.  ``fit``: ``"width"`` (default) keeps
    the whole horizontal field (the left and right edges at the principal
    point's height stay in the frame; corners are cut), ``"height"`` the
    vertical one, ``"same"`` keeps the focal length (a strong fisheye then
    loses its edges).  The mask is resampled with nearest neighbour.
    """
    import cv2
    from ..colmap import project_camera, scale_camera_params, unproject_camera
    h, w = image.shape[:2]
    s = w / cam.width if scale is None else float(scale)
    p = scale_camera_params(cam.model, cam.params, s)
    cx, cy = float(p[2]), float(p[3])                     # COLMAP corner origin
    if fit == "same":
        f = float(np.sqrt(p[0] * p[1]))
    else:
        k = 0 if fit == "width" else 1
        edge = np.array([[0.0, cy], [w, cy]]) if k == 0 else np.array([[cx, 0.0], [cx, h]])
        xy = unproject_camera(cam.model, p, edge)
        xy[np.linalg.norm(xy, axis=1) >= radial_limit(cam)] = np.nan
        c = cx if k == 0 else cy
        size = w if k == 0 else h
        room = np.array([c, size - c])
        with np.errstate(divide="ignore", invalid="ignore"):
            fs = room / np.abs(xy[:, k])
        f = float(np.nanmin(fs)) if np.any(np.isfinite(fs)) else float(np.sqrt(p[0] * p[1]))
    mx, my = np.empty((h, w), np.float32), np.empty((h, w), np.float32)
    xs = (np.arange(w) + 0.5 - cx) / f
    lim = radial_limit(cam)
    for y0 in range(0, h, 256):                           # in bands: a full Navcam frame at once needs GBs
        ys = (np.arange(y0, min(h, y0 + 256)) + 0.5 - cy) / f
        X, Y = np.meshgrid(xs, ys)
        src = project_camera(cam.model, p, np.stack([X, Y, np.ones_like(X)], axis=-1)) - 0.5  # cv2: centre origin
        src[np.hypot(X, Y) >= 0.999 * lim] = -1e4        # beyond the model's fold-over: no data
        mx[y0:y0 + len(ys)], my[y0:y0 + len(ys)] = src[..., 0], src[..., 1]
    out = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    om = None
    if mask is not None:
        m = mask if mask.shape[:2] == (h, w) else cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        om = cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return {"image": out, "mask": om, "f_px": f, "f_in_px": float(np.sqrt(p[0] * p[1])), "scale": s,
            "valid_field_deg": float(np.degrees(np.arctan(lim))) if np.isfinite(lim) else None,
            "principal_point_px": (cx, cy), "field_deg": float(np.degrees(np.arctan(cx / f) + np.arctan((w - cx) / f)))}


def screen_mask(image: np.ndarray, mask: Optional[np.ndarray], alpha: float = 0.3,
                hardware_only: bool = True) -> np.ndarray:
    """``image`` with a white screen of opacity ``alpha`` over the masked pixels (mask 0);
    ``hardware_only``: leave the black no-data pixels (padding, outside the lens) black."""
    if mask is None:
        return image
    out = image.astype(np.float32)
    sel = mask == 0
    if hardware_only:
        sel &= image.max(axis=2) > 0
    out[sel] = (1 - alpha) * out[sel] + alpha * 255.0
    return out.round().clip(0, 255).astype(np.uint8)


def distortion_field(cam: Camera, step: float = 256.0) -> Dict[str, np.ndarray]:
    """Pixel displacement of the lens model from a pinhole with the same fx, fy, cx, cy (arrows for a plot)."""
    from ..cmod import pixel_grid
    from ..colmap import unproject_camera
    uv = pixel_grid(cam.width, cam.height, step)
    xy = unproject_camera(cam.model, cam.params, uv)
    pin = xy * cam.params[:2] + cam.params[2:4]
    return {"uv": uv, "d_px": uv - pin}


def example_figure(image: np.ndarray, mask: Optional[np.ndarray], cam: Camera, title: str = "",
                   alpha: float = 0.3, fit: str = "width", grid: bool = True):
    """
    Original and undistorted image side by side, the hardware mask screened in
    white (opacity ``alpha``).  ``grid``: a line grid drawn straight in the
    undistorted image and through the lens model in the original, to show the
    distortion.  Returns the matplotlib figure and the undistortion record.
    """
    import matplotlib.pyplot as plt
    from ..colmap import project_camera, scale_camera_params
    und = undistort(image, cam, mask, fit=fit)
    lim = radial_limit(cam)
    h, w = image.shape[:2]
    fig, ax = plt.subplots(1, 2, figsize=(15, 15 * h / w / 2 + 0.9))
    ax[0].imshow(screen_mask(image, mask, alpha))
    ax[1].imshow(screen_mask(und["image"], und["mask"], alpha))
    if grid:
        p = scale_camera_params(cam.model, cam.params, und["scale"])
        f = und["f_px"]
        for k in np.linspace(-1, 1, 9):
            t = np.linspace(-1, 1, 400)
            for xs, ys in ((np.full_like(t, k * w / 2), t * h / 2), (t * w / 2, np.full_like(t, k * h / 2))):
                xs, ys = xs + w / 2 - und["principal_point_px"][0], ys + h / 2 - und["principal_point_px"][1]
                ax[1].plot(xs + und["principal_point_px"][0] - 0.5, ys + und["principal_point_px"][1] - 0.5,
                           color="#00e5ff", lw=0.6, alpha=0.7)
                X = np.c_[xs / f, ys / f, np.ones_like(xs)]
                uv = project_camera(cam.model, p, X)
                uv[np.hypot(X[:, 0], X[:, 1]) >= 0.999 * lim] = np.nan
                ok = np.all(np.isfinite(uv), axis=1) & (uv[:, 0] > -w * 0.05) & (uv[:, 0] < w * 1.05) & \
                    (uv[:, 1] > -h * 0.05) & (uv[:, 1] < h * 1.05)
                uv[~ok] = np.nan
                ax[0].plot(uv[:, 0] - 0.5, uv[:, 1] - 0.5, color="#00e5ff", lw=0.6, alpha=0.7)   # imshow: centre origin
    lim_txt = (f"; the model folds back beyond {und['valid_field_deg']:.1f} deg off-axis (black in the undistorted image)"
               if und["valid_field_deg"] is not None and und["valid_field_deg"] < 75 else "")
    ax[0].set_title(f"{title}\noriginal ({cam.distortion} lens model {cam.key}, f = {und['f_in_px']:.0f} px{lim_txt})",
                    fontsize=9)
    ax[1].set_title(f"undistorted to a pinhole, f = {und['f_px']:.0f} px, horizontal field {und['field_deg']:.1f} deg "
                    f"(fit = {fit!r})", fontsize=9)
    for a in ax:
        a.set_xlim(-0.5, w - 0.5)
        a.set_ylim(h - 0.5, -0.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    return fig, und
