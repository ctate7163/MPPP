"""One-off (0.96.0): write the notebooks - one pipeline example and one per study.  Run from the repo root.
Not shipped as a tool; kept so the notebooks can be regenerated if their JSON is hand-edited badly."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

HEAD = '''import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/conda: duplicate Intel OpenMP guard; before torch
import json
import logging
from pathlib import Path
import mppp
logging.basicConfig(level=logging.INFO, format="%(message)s")   # the progress lines
print("MPPP", mppp.__version__, "from", Path(mppp.__file__).parent)'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text, tags=None):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip("\n")}
    if tags:
        c["metadata"]["tags"] = list(tags)
    return c


def write(path, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT), len(cells), "cells")


# ------------------------------------------------------------------------------------------------ the pipeline example
pipeline = [
    md('''
# MPPP — the pipeline on the example data (Rockytop, sols 461–464)

The whole of MPPP's core on one site, start to finish: the PDS products selected and processed, the COLMAP block
aligned from the promoted camera models and the rover poses, its health checked, then the products of 0.96 — the
dense cloud, the station mesh, per-image depth and photometric maps, and the stereoscopic panorama — and the
exports to Metashape and Nerfstudio.  Every step is one function of `mppp.pipeline` or `mppp.sfm`, and every step
is also one `mppp` command (shown beside it), so a run can be scripted the same way.

**Data.** `mppp example fetch` downloads the example data set (`mppp_example_m2020_rockytop`: the PDS products of
sols 461–464 of Rockytop, Navcam and Mastcam-Z 34 mm) and prints the `MPPP_PDS` to set.  Your own PDS archive works
the same way: `pds_dir` is searched recursively; the sites and their sol ranges are in `mppp/data/sites.json`
(`mppp sites list`, or `Settings.sites_extra` for one that is not there).

**Kernel.** `pip install -e .[sfm,mask,ibr]`: pycolmap 4.2 and pyceres 2.6 for the alignment, torch + timm for the
terrain masks, open3d for the 0.96 products.  GPU feature extraction and matching run in `gpu_python`
(`MPPP_GPU_PYTHON`).  The research studies — the camera-model solves, the error analysis, the mask training, the
site maps — are not part of this notebook: they live in `mppp_studies` and have a notebook each under `studies/`.
'''),
    code(HEAD + '''
from mppp.pipeline import Settings
from mppp.sfm import check_ba_environment
import pycolmap, pyceres
print("pycolmap", pycolmap.__version__, "(CUDA:", pycolmap.has_cuda, ") | pyceres", pyceres.__version__)
check_ba_environment()        # stops here if this kernel cannot run the weighted BA (e.g. it is the conda CUDA env)'''),
    md('''
## 1  Parameters
Every option is a field of `mppp.pipeline.Settings` (`help(Settings)`); `<WORK>/mppp_settings.json` and the site's
`settings` in `sites.json` are read by `load_settings`.  The site below is the example data set; `SITE` can be any
name of `sites.json` or a WORK folder.
'''),
    code('''
SITE = "rockytop_example"                 # the example data set (mppp example fetch); any site of sites.json, or a WORK folder
settings = Settings(
    pds_dir=os.environ.get("MPPP_PDS", "/data/m2020"),       # the local PDS archive (searched recursively)
    scapes_root=os.environ.get("MPPP_SCAPES", "/scapes"),    # one WORK folder per site below it
    include_zcam=False,                   # True: the Navcam + Mastcam-Z block (the site's zooms; folder ..._zcam)
    add_nearby_waypoints=10,              # also the stations within this many metres of the site's stations; 0: off
    reprocess_all=False,                  # True: process every image again (new masks, radiometry, poses)
    workers=4,                            # image-processing worker processes
    variant="",                           # "" -> WORK/colmap; "name" -> WORK/colmap_name (other settings side by side)
    exclude=[],                           # images to leave out: "S026D0282", "sol:462", "seq:NCAM08111", "ZR0_0461_*"
    gui_files=True,                       # colmap_gui.ini and open_in_colmap.bat into the project
    # sites_extra={"my_site": (700, 712)},   # a site that is not in sites.json
    # infer_mask=False,                   # no terrain masks (quick look)
    # navcam_intrinsics="hold",           # keep the Navcam cameras at the promoted models
)
STOP_AFTER = None                         # part of the alignment only: "project", "features", "database", "match", "reconstruct", "health"
STATION = None                            # the station of the 0.96 products: "S026D0392", "26_392" or "sol:461"; None: the first''', tags=["parameters"]),
    md('''
## 2  Select and process the products — `mppp process SITE`
One product per exposure (the largest sub-frame, the highest version) into `<WORK>/processed`: radiometrically
normalised 8-bit PNGs (PDS names kept), the terrain masks (the released `mppp_mask` model), CAHV intrinsics and the
waypoint poses, and the manifest the alignment reads.  Images pointing above `sky_elevation_deg`, taken outside
`lmst_window_h`, or too saturated are skipped (the manifest lists them).  Images already processed with the same
configuration are reused; a changed configuration (another mask model, say) processes everything again.
'''),
    code('''
from mppp.pipeline import resolve_site, select_products, process_site
from mppp.sfm.workdir import load_manifest
site, settings = resolve_site(SITE, settings)
print(site.name, "sols", site.sols_text, "->", site.work, "| Mastcam-Z zooms:", site.zooms or "none")
paths, selection = select_products(site, settings)
print(len(paths), "products;", json.dumps({k: v for k, v in selection.items() if k != "dropped_subframes"}, indent=1))
processed = process_site(site.work, settings)
manifest, manifest_file = load_manifest(processed)
print(f"{manifest['n_processed']} images in {manifest_file.name}; {len(manifest['failed'])} failed, "
      f"{len(manifest.get('skipped', []))} skipped by the selection rules")
for f in manifest["failed"]:
    print("  FAILED", f["file"], "->", f["error"])'''),
    code('''
import matplotlib.pyplot as plt
from PIL import Image
m = manifest["images"][-1]
img = Image.open(processed / m["outputs"]["PNG8"])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(img); ax[0].set_title(Path(m["source_product"]).stem, fontsize=8)
if "mask" in m["outputs"]:
    ax[1].imshow(Image.open(processed / m["outputs"]["mask"]), cmap="gray"); ax[1].set_title("mask (white = include)")
for a in ax:
    a.axis("off")
print(json.dumps({k: m[k] for k in ("sol", "site", "drive", "LMST", "camera_temperature_degC") if k in m}, indent=1))'''),
    md('''
## 3  Align the block — `mppp align WORK`
One COLMAP camera per eye from the promoted models (`mppp cmods`), SIFT at native resolution inside the masks,
matching, a triangulation from the rover poses and a weighted bundle adjustment with position and attitude priors
(frames that do not align with the majority are excluded; with Mastcam-Z the Navcam block is solved first and
held), the health report with a top-down station map, and `error_input/` for the error analysis.

**Frame.** The block lives in East-North-Up metres of the landing site frame minus a rounded offset
(`project.json`, `error_input/summary.json`).  **Reruns** reuse the features while the images are the same and the
matches while the features and matching settings are the same.
'''),
    code('''
from mppp.pipeline import align, load_settings
WORK = site.work
settings = load_settings(WORK, base=settings)
report = align(WORK, settings, stop_after=STOP_AFTER)
print("run key", report.get("run_key"), "| verdict", report.get("verdict"), "| project", report.get("project"))'''),
    md('''
## 4  Alignment health — check before anything else
Each check has provisional warn and fail thresholds (set a priori, not validated limits).  The report is in
`<project>/health/` (`health.json`, `health.md`, `health.png`, `station_map.png`: a top-down view of the stations
with the prior-to-refined shift of every camera).
'''),
    code('''
from IPython.display import Image as NbImage, display
from mppp.pipeline import load_results
from mppp.sfm import health_table
proj, rec, report, summary = load_results(WORK, settings.variant)      # from disk: also after a kernel restart
print(health_table(report))
for name in ("health.png", "station_map.png"):
    f = Path(proj.root) / "health" / name
    if f.is_file():
        display(NbImage(filename=str(f)))
if report["verdict"] == "fail":
    print("\\nFAIL: fix the alignment (matching, priors, masks, thresholds) before any analysis.")'''),
    md('''
## 5  Cameras and tie points
The cameras start to refined (the Navcam eyes and their temperature bins; Mastcam-Z focus bins when the block has
them), the tie points by convergence angle, and the camera shifts from the prior poses.
'''),
    code('''
from mppp.sfm.reconstruction import camera_changes, convergence_statistics
for r in camera_changes(rec, proj):
    if r["missing"]:
        print(r["camera"], ": not in the reconstruction"); continue
    print(f"{r['camera']} ({r['images']} images){'  [temperature bin]' if r['thermal_bin'] else ''}: "
          + " ".join(f"{n} {a:.4g}->{b:.4g}" for n, a, b in zip(r["names"][:10], r["start"], r["refined"])))
conv = proj.settings["reconstruction"].get("convergence") or convergence_statistics(rec, proj)
print("tie points by convergence angle", conv["bins_deg"], "deg:", conv["points"], "| cross-station:", conv["cross_station_points"])
eps = (summary or {}).get("eps", {})
print(f"eps (rms residual / sqrt 2): same-station {eps.get('eps_intra_px', float('nan')):.3f} px, "
      f"cross-station {eps.get('eps_cross_px', float('nan')):.3f} px")'''),
    code('''
import csv
from mppp.sfm import plot_camera_shifts
with (Path(proj.root) / "error_input" / "poses.csv").open(encoding="utf-8") as f:
    pose_rows = list(csv.DictReader(f))
fig = plot_camera_shifts(proj, rows=pose_rows)
plt.show()
print("error_input:", Path(proj.root) / "error_input", "| run_done.json:", json.dumps(report.get("run_done") or {}, default=str)[:300])'''),
    md('''
## 6  The dense cloud — `mppp dense WORK`
The COLMAP dense reconstruction of the finished block: the images undistorted from the native-pixel model, the
rover masks for the fusion, patch-match stereo, fusion (`<project>/dense/fused.ply`).  Needs the COLMAP executable
with CUDA (`settings.colmap_exe` / `MPPP_COLMAP`); on a machine without it the cell reports and the next sections
read a `fused.ply` made elsewhere.
'''),
    code('''
from mppp.sfm.dense import run_dense
try:
    dense = run_dense(proj)                                    # prepare, stereo, fusion
    print(json.dumps({k: str(v) for k, v in dense.items()}, indent=1)[:800])
except Exception as e:                                         # no COLMAP with CUDA here
    print("dense reconstruction not run:", e)
fused = sorted(Path(proj.root).glob("dense*/fused.ply"))
print("fused clouds:", [str(f) for f in fused] or "none yet")'''),
    md('''
## 7  The station mesh — `mppp mesh WORK --station S026D0392`
0.96.  One station (a site and drive of the rover) of the block: its frames by camera family, the mast pivot fitted
from their centres and rotations, and the surface over the fused cloud — Poisson (COLMAP's Delaunay mesher when the
executable is given), then the cleanup: components under 1 %, rubber sheets, surface no fused point supports within
τ(z) = max(5 cm, 2 % z, 3 σz), everything beyond 60 m; decimation, Taubin smoothing, and a far sphere at 100 m so
every ray hits something.  Products under `<project>/ods/<station>/` with `mesh_report.json` and a preview.
'''),
    code('''
from mppp.ibr.station import select_station, station_images
from mppp.ibr.mesh import build_station_mesh, mesh_preview, read_mesh, read_points
stations = sorted({r.get("station") for r in proj.images if r.get("station")})
print("stations in the block:", stations)
spec = STATION or stations[0]
st = select_station(proj, rec, spec, families="N")
out = Path(proj.root) / "ods" / st.label
out.mkdir(parents=True, exist_ok=True)
(out / "station.json").write_text(json.dumps(st.to_json(), indent=1, default=float), encoding="utf-8")
print(st.label, len(st.images), "frames; pivot", [round(float(v), 3) for v in st.pivot], "m; fit", st.pivot_fit)
if fused:
    pts = read_points(fused[-1])
    rep = build_station_mesh(pts, st.pivot, out, radius_m=60.0, far_radius_m=100.0, station=st.to_json())
    V, F = read_mesh(out / "mesh.ply")
    mesh_preview(V, F, st.pivot, out / "mesh_preview.png", sphere=read_mesh(out / "far_sphere.ply"), radius_m=60.0)
    print(json.dumps({k: rep[k] for k in ("surface", "cleanup", "decimation", "coverage_by_range")}, indent=1, default=float)[:1500])
    display(NbImage(filename=str(out / "mesh_preview.png")))'''),
    md('''
## 8  Per-image depth maps and the photometric cube — `mppp depthmaps WORK --station ... --photometry`
Every frame of the station ray-cast through its own camera model into the mesh and the far sphere: depth, range,
normal and label maps (0 none, 1 mesh, 2 far) at half the native size, validated against the sparse points the
frame observes (`maps_report.md`).  Then, from the sun direction of each frame (`sun_enu` of the row, turned into
the block frame), the photometric cube: incidence, emission and phase angles and the cast shadow — `photometry/`
(npz), `photometry_maps/` (COLMAP `.bin` arrays), previews, `photometry_report.json`.
'''),
    code('''
from mppp.ibr.raycast import Scene, cast_station, load_maps
from mppp.ibr.photometry import photometric_cube, load_photometry
if (out / "mesh.ply").is_file():
    scene = Scene(read_mesh(out / "mesh.ply"), read_mesh(out / "far_sphere.ply"))
    mrep = cast_station(proj, rec, st, scene, out, families="N", scale=0.5)
    print(open(out / "maps_report.md", encoding="utf-8").read()[:2000])
    prep = photometric_cube(proj, rec, st, scene, out / "maps", out)
    print(json.dumps(prep["summary"], indent=1, default=float)[:1200])
    first = sorted((out / "maps").glob("*.npz"))[0]
    maps, pm = load_maps(first), load_photometry(out / "photometry" / first.name)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].imshow(maps["depth"], cmap="viridis"); ax[0].set_title(f"depth (m) {first.stem}", fontsize=8)
    ax[1].imshow(pm["incidence"], cmap="magma", vmin=0, vmax=90); ax[1].set_title("incidence (deg)")
    ax[2].imshow(pm["shadow"], cmap="gray"); ax[2].set_title("cast shadow")
    for a in ax:
        a.axis("off")'''),
    md('''
## 9  The stereoscopic panorama — `mppp ods WORK --station ...`
0.96 prototype: an omni-directional stereo pair (equirectangular, 8192 × 4096, 64 mm between the eyes) of the
station — each eye's rays cast into the scene, every hit coloured from the source frame that sees it best (the
smallest angle to the frame's axis, inside the frame and its mask).  Two images, `ods_left.png` and `ods_right.png`,
a `provenance.png` (which frame coloured what) and `ods.json` with the coverage.  Views in any VR photo viewer that
takes a top-bottom or side-by-side equirectangular pair.  The viewer itself is future work.
'''),
    code('''
from mppp.ibr.ods import render_station
if (out / "mesh.ply").is_file():
    orep = render_station(proj, rec, st, scene, out, width=4096, ipd_m=0.064, families="N", scale=0.5)
    for eye, e in orep["eyes"].items():
        c = e["coverage"]
        print(f"{eye}: coloured {100 * c['coloured']:.1f}%, mesh {100 * c['mesh']:.1f}%, far {100 * c['far']:.1f}%, no surface {100 * c['no_surface']:.1f}%")
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].imshow(Image.open(out / "ods_left.png")); ax[0].set_title("left eye")
    ax[1].imshow(Image.open(out / "ods_right.png")); ax[1].set_title("right eye")
    for a in ax:
        a.axis("off")'''),
    md('''
## 10  Exports — `mppp metashape WORK`, `mppp nerfstudio WORK`
The same block for other tools: a Metashape job (the cameras, the poses and the matches as a `.psx` build script)
and a Nerfstudio data set (`transforms.json`, the undistorted images and masks, the sparse cloud).
'''),
    code('''
from mppp.sfm.metashape import write_metashape_job
from mppp.sfm.nerfstudio import write_nerfstudio
print("metashape:", write_metashape_job(proj))
print("nerfstudio:", write_nerfstudio(proj, threads=4))'''),
]
write(NB / "pipeline_example.ipynb", pipeline)

# ------------------------------------------------------------------------------------------------ the calibration study
camera_models = [
    md('''
# Study — the camera models (`mppp_studies.calibration`)

How the promoted camera models of `mppp cmods` are made: the Navcam joint over many blocks (the focal-length and
principal-point laws in temperature, the rig between the eyes, the Hazcam laws), the Mastcam-Z joint (the lens
law in focus count per zoom, the backlash states, the rig), the rover-deck adjustment that ties the Mastcam-Z rig
to the Navcam one, and the promotion.  The study reads finished blocks (`mppp align`, the `error_input/` folders) and
never changes them; the pipeline reads only the promoted files.

The runs are long (hours over the consensus blocks) and are started from the command line — the freeze driver
(`scripts/windows/freeze.bat`, `mppp_studies.calibration.freeze`) runs the whole chain unattended.  This notebook
shows the commands, and reads and plots their results.  `docs/methods.md` describes the models.
'''),
    code(HEAD + '''
import mppp_studies
from mppp.calib import models
print("studies from", Path(mppp_studies.__file__).parent)'''),
    md("## 1  Parameters"),
    code('''
ROOT = Path(os.environ.get("MPPP_SCAPES", "/scapes"))     # the blocks root (the freeze blocks on the PC)
ANALYSIS = ROOT / "camera_analysis"                        # where the joints write (calib ... --out)
ZOOMS = (34, 63, 79, 110)
RUN = False                                                # True: run the joints from here (hours); False: read the results''', tags=["parameters"]),
    md('''
## 2  The promoted models in use
`mppp cmods` — the files the pipeline reads (`MPPP_CMODS`, else the package data): per eye a `.json` with the
law and its covariance; the rig; the Mastcam-Z lens model.  `describe` prints what is promoted and from where.
'''),
    code('''
print("\\n".join(models.describe()))
print(json.dumps(models.status(), indent=1, default=str)[:1500])'''),
    md('''
## 3  The Navcam joint — `mppp calib navcam`
All the consensus blocks' Navcam frames in one adjustment: each eye's fx, fy, cx, cy on a line in the camera
temperature (with the knee), the distortion shared, the rig between the eyes (rotation, the "deck" translation),
every frame's pose free.  The blocks' names come from `sites.json` (`ecam_consensus`).
'''),
    code('''
cmd = f"python -m mppp calib navcam --root {ROOT} --group ecam_consensus"
print(cmd)
if RUN:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "mppp", "calib", "navcam", "--root", str(ROOT), "--group", "ecam_consensus"], check=True)
runs = sorted(ANALYSIS.glob("navcam_joint_*")) if ANALYSIS.is_dir() else []
print("results:", [r.name for r in runs][-5:] or "none yet")'''),
    code('''
from IPython.display import Image as NbImage, display
if runs:
    res = runs[-1]
    st = json.loads((res / "navcam_joint.json").read_text(encoding="utf-8")) if (res / "navcam_joint.json").is_file() else {}
    for eye, e in (st.get("eyes") or {}).items():
        print(eye, {k: e[k] for k in e if k in ("fx", "fy", "cx", "cy", "slope_fx_ppm_per_K", "knee_degC")})
    for png in sorted(res.glob("*.png"))[:4]:
        display(NbImage(filename=str(png)))'''),
    md('''
## 4  The Mastcam-Z joint — `mppp calib zcam --zoom 34`
Per zoom: the lens law in the focus motor count (f, fy/fx, cx, cy on a line per backlash state, the distortion on
the line), a pooled diagnostic pass per sequence, the rig to the Navcam block, the per-block aspect gate, the
weak-geometry sequences.  The backlash states come from the labels and the blocks (`mppp zcam-states`, the found
list `mars2020_mastcam-z_low_backlash_observations.csv`).
'''),
    code('''
for z in ZOOMS:
    print(f"python -m mppp calib zcam --zoom {z} --root {ROOT} --group zcam{z}_consensus --zcam-pool sequence")
zres = {z: sorted(ANALYSIS.glob(f"zcam{z}_joint_*")) for z in ZOOMS} if ANALYSIS.is_dir() else {}
for z, rs in zres.items():
    if rs:
        j = rs[-1] / "zcam_joint.json"
        d = json.loads(j.read_text(encoding="utf-8")) if j.is_file() else {}
        print(z, rs[-1].name, "promotable:", d.get("promotable"), "| failed checks:", d.get("failed_checks"))
        for png in sorted(rs[-1].glob("zcam_efl_vs_focus*.png"))[:1] + sorted(rs[-1].glob("zcam_rig*.png"))[:1]:
            display(NbImage(filename=str(png)))'''),
    md('''
## 5  The rover deck and the mast head
`python -m mppp_studies.calibration.deck WORK --zcam-rig full --rig full`: the deck fiducials (seven surveyed
targets) in the bundle adjustment of one block, which ties the Mastcam-Z rig to the Navcam rig and gives the mast
model (`claude/mast-head-2026-11-08-...`).  `zcam_mast_head` draws the cameras on the mast head from the promoted
rigs; `deck_rig_trend` the rig drift over the mission from the deck pairs; `fiducial_check` reads the fiducials
only.
'''),
    code('''
from mppp_studies.calibration import zcam_mast_head
fig_dir = Path(os.environ.get("MPPP_OUTPUTS", "Claude outputs")) / "mast_head"
try:
    out = zcam_mast_head.main(["--out", str(fig_dir)])
    for png in sorted(fig_dir.glob("*.png"))[:2]:
        display(NbImage(filename=str(png)))
except SystemExit:
    pass'''),
    md('''
## 6  Promotion — `python -m mppp_studies.calibration.promote_cmods RESULT_DIR --note "..."`
A joint that passes its checks writes candidate files; `promote_cmods` copies them into the cmods folder with a
line in `CHANGES.md`, and `mppp cmods` shows them.  `calib zcam --promote-waive CHECK` promotes past a named failed
check (the 0.96 freeze waives the per-block aspect gate for 63, 79 and 110 mm: unobservable there).
'''),
    code('''
print("python -m mppp_studies.calibration.promote_cmods", ANALYSIS / "navcam_joint_YYYYMMDD", "--note 'the 0.96 freeze'")
print("python -m mppp cmods")'''),
]
write(NB / "studies" / "camera_models.ipynb", camera_models)

# ------------------------------------------------------------------------------------------------ the site maps study
sites_maps = [
    md('''
# Study — the sites and the maps (`mppp_studies.sites`)

The Mars 2020 sites of `mppp/data/sites.json` — their sol ranges, waypoints, zooms and official block numbers —
and the figures of the paper that place them on the terrain: the mission traverse with the rover sites, the
cluster zooms, the LMST histograms.  The waypoint table is downloaded (`mppp waypoints --refresh`); the DTM and
orthomosaic tiles come from the mission's released products (`maps` extra: rasterio, pyproj, adjustText).

`make_maps.bat` runs the whole thing on the PC (`mppp maps all`); here the pieces.
'''),
    code(HEAD + '''
from mppp.waypoints.sites import load_site_table
SITES = load_site_table()["sites"]
print(len(SITES), "sites in sites.json")'''),
    md("## 1  Parameters"),
    code('''
OUT = Path(os.environ.get("MPPP_OUTPUTS", "Claude outputs")) / "maps"
BACKGROUND = "ortho"                      # "dtm" or "ortho"
DPI = 150
FIGURES = [2, 3, 6]                       # the paper figures to draw (mppp maps figures ...)''', tags=["parameters"]),
    md('''
## 2  The sites
Every site: sols, the number of waypoints, the maximum separation, the zooms, the official block.  A site is added
by one line in `sites.json` (or `Settings.sites_extra`); `mppp sites list` prints this table.
'''),
    code('''
for name, s in sorted(SITES.items()):
    print(f"{name:28s} sols {s['sols']}  wp {s.get('n_waypoints', '?'):>3}  "
          f"sep {s.get('max_waypoint_separation_m', 0):6.1f} m  zcam {s.get('zcam') or '-'}  block {s.get('official_blocks') or '-'}")'''),
    md('''
## 3  The waypoints
The current waypoint table (the mission's `M20_waypoints` JSON): every end-of-drive position with its site, drive
and sol.  `sync_study_waypoints` keeps the study's copy next to the package snapshot.
'''),
    code('''
from mppp.waypoints import load_waypoints, snapshot_path
wps = load_waypoints()
print(len(wps), "waypoints; snapshot", snapshot_path())
print(wps[-1])'''),
    md('''
## 4  The figures — `mppp maps figures 2 3 6 --background ortho`
Figure 2: the traverse and the sites; figure 3: the cluster zooms with the sol numbers; figure 6: the Mars
orthomosaic to the rover sites (E–H the zooms).  `mppp maps zooms` draws every cluster; `mppp maps all` everything.
'''),
    code('''
import subprocess, sys
cmd = [sys.executable, "-m", "mppp", "maps", "figures", *map(str, FIGURES), "--out", str(OUT), "--dpi", str(DPI), "--background", BACKGROUND]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:])
from IPython.display import Image as NbImage, display
for png in sorted(OUT.glob("*.png"))[:3]:
    display(NbImage(filename=str(png)))'''),
    md('''
## 5  LMST histograms
The local mean solar time of every frame of the consensus blocks, per camera family — the lighting the models
were made under (`mppp_studies.sites.figures`).
'''),
    code('''
from mppp_studies.sites.figures import lmst_images, lmst_site_row, plot_lmst_histograms
blocks_root = Path(os.environ.get("MPPP_SCAPES", "/scapes"))
if blocks_root.is_dir():
    images, found = lmst_images([blocks_root])                 # {site: {stem: record}} from the newest manifests
    rows = [lmst_site_row(site, list(recs.values())) for site, recs in images.items()]
    print(plot_lmst_histograms(images, rows, OUT / "lmst_histograms.png"))
    display(NbImage(filename=str(OUT / "lmst_histograms.png")))
else:
    print("no blocks root at", blocks_root)'''),
]
write(NB / "studies" / "sites_maps.ipynb", sites_maps)

# ------------------------------------------------------------------------------------------------ the mask training: one cell added
mt = NB / "studies" / "mask_training.ipynb"
nb = json.loads(mt.read_text(encoding="utf-8"))
src_all = "".join("".join(c["source"]) for c in nb["cells"])
if "--compare" not in src_all:
    nb["cells"].append(md('''
## 0.96 — the recipes, the frozen test list and the comparison after training
`train_masks.bat --recipe A --test-list masks/test_list_v9.txt --compare`: recipe A adds the online augmentations
(`mppp_studies.mask_training.augment`: photometric, the LMST sampler, the angular scale, defocus, copy-paste);
`--test-list` holds the same frames out of every run so runs can be compared; `--compare` runs
`compare_mask_models` of the new checkpoint against the blocks' model right after the training and writes
`Claude outputs/mask_<checkpoint>/mask_verdict.json` — the freeze driver (`freeze.bat --mask-model auto`) reads
it and processes the blocks again with the new model when it wins by the margin.
''') | {"id": "c96"})
    nb["cells"].append(code('''
from mppp_studies.mask_training.train_mask import read_verdict_from_ab, DEFAULT_BASELINE, VERDICT_FILE
from mppp_studies.paths import outputs_dir
verdicts = sorted(outputs_dir().glob(f"mask_*/{VERDICT_FILE}"))
for v in verdicts[-3:]:
    d = json.loads(v.read_text(encoding="utf-8"))
    print(v.parent.name, "->", d["winner"], "margin", d["margin"], d["models"])
print("baseline:", DEFAULT_BASELINE)''') | {"id": "c97"})
    for c in nb["cells"][-2:]:
        c["source"] = c["source"].splitlines(keepends=True)
    mt.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(mt.relative_to(ROOT), "two cells added")

# the error analysis notebook: its title
ea = NB / "studies" / "error_analysis.ipynb"
nb = json.loads(ea.read_text(encoding="utf-8"))
c0 = nb["cells"][0]
s = "".join(c0["source"])
if s.startswith("# MPPP — 03 Error analysis"):
    s = s.replace("# MPPP — 03 Error analysis of COLMAP alignments", "# Study — the error analysis of the blocks (`mppp_studies.error`)", 1)
    c0["source"] = s.splitlines(keepends=True)
    ea.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(ea.relative_to(ROOT), "title")
