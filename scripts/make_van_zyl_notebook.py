"""Writes notebooks/van_zyl_ods_example.ipynb (0.99.33, tiny block 0.99.34; Christian, 18 Nov 2026: "Make a notebook that downloads van zyl
ecam frames and build the ods and models through the whole pipeline. Use this as an example to illustrate the process
and RPP design features").  Run from the repository root: ``python scripts/make_van_zyl_notebook.py``."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS = []


def md(text: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str, tags=()) -> None:
    c = nbf.v4.new_code_cell(text.strip("\n"))
    if tags:
        c.metadata["tags"] = list(tags)
    CELLS.append(c)


md(r"""
# Van Zyl Overlook, sol 52 (tiny): from the PDS to stereo panoramas, 3D models and the viewer

This notebook starts from nothing and ends with the station in the Rover Photogrammetry Viewer (RPV): it downloads
one station's Navcam frames from the PDS Imaging Node, aligns them with the promoted camera models and the rover's own
poses, builds the terrain, renders the omnidirectional stereo (ODS) panoramas, makes the textured 3D station models
and publishes them to a local copy of the viewer.  Along the way it points out how RPP is built and why.

**The station.** Perseverance at Van Zyl Overlook on sol 52 (site 3, drive 2046; the station `Sol0052_S003D2046`
of `rpp/data/single_sites.json`): six Navcam stereo pairs of the end-of-drive panorama and the front Hazcam pair, with
one context pair for the terrain - the Navcam pair taken mid-drive, at drive 2016, before the sol's last drive.  It is
the first station of the public viewer.

**A tiny block** (0.99.34; Christian, 19 Nov 2026: "Let's use a van zyl tiny with just the one site and its mid drive
imaging"): one sol, one site, 16 frames - small enough to download and align in minutes.  The packaged station also
has the sol 33 end-of-drive panorama (S003D1374, 63 m west) as context; the viewer's own copy uses it, this example
leaves it out.  The panoramas and models come out the same way; the terrain beyond about 20 m is thinner.

**What you need.**

| | |
|---|---|
| RPP | 1.0 or later, installed from the repository with `pip install -e ".[all]"` (the color-study cell uses `rpp_studies`, which is not in the PyPI package) |
| GPU | pycolmap built with CUDA for the dense step (patch-match stereo); the other steps run on the CPU |
| disk | about 2 GB (the PDS products, the processed frames, the dense cloud and the panoramas) |
| time | about an hour on a desktop with a GPU - most of it the dense step and the panoramas |
| network | the PDS Imaging Node (https://pds-imaging.jpl.nasa.gov) and the mask model's download |

Every step writes its products and a JSON record next to them and can be run again: a step whose inputs, settings and
code have not changed is skipped.  *This notebook was written for 0.99.34 and has not yet been run end to end on a
fresh machine - please report what breaks.*
""")

md(r"""
## 0  The environment

`rpp doctor` says what is set and what is missing (the PDS archive, the scapes root, the camera models, the mask
model, pycolmap and CUDA).  Paths come from the environment (`RPP_PDS`, `RPP_SCAPES`, ...) through `rpp.env`, never
from the code.
""")

code(r"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows/conda: torch and pycolmap both bring Intel OpenMP
import json
import logging
from pathlib import Path

import rpp
from rpp.cli import main as rpp_main                   # every shell command is also a function: rpp_main([...]) -> exit code
logging.basicConfig(level=logging.INFO, format="%(message)s")
print("RPP", rpp.__version__, "from", Path(rpp.__file__).parent)
rpp_main(["doctor"])
""")

md(r"""
## 1  Where things go

Everything of this example lives under one folder, `DATA`: its own PDS archive (only the products downloaded here,
so the selection below finds exactly these), the block's WORK folder, the station products and the viewer.
""")

code(r"""
from rpp import env as rpp_env
DATA = Path(rpp_env.get("EXAMPLE_DATA") or Path.home() / "rpp_van_zyl").expanduser()      # RPP_EXAMPLE_DATA, if set
PDS = DATA / "pds"                 # the archive layout of the PDS Imaging Node (mars2020_navcam_ops_calibrated/...)
SCAPES = DATA / "blocks"           # one WORK folder per block
STATIONS = DATA / "single_sites"   # one folder per station: mesh, maps, panoramas, 3D models
WEB = DATA / "rpv"                 # a local copy of the viewer: index.html, vendor/, panos.json, panos/
for d in (PDS, SCAPES, STATIONS, WEB):
    d.mkdir(parents=True, exist_ok=True)
LABEL = "Sol0052_S003D2046"        # the station, as the viewer lists it
print(DATA)
""", tags=("parameters",))

md(r"""
## 2  The station's frames, from the packaged configuration

The single-station sites are data, not code: `rpp/data/single_sites.json` names each station's frames by their PDS
product names (kept unchanged through the whole pipeline, so every file can be traced back to its product), its
context frames and the defaults of every step.
""")

code(r"""
from rpp.ibr import single_sites as SS
packaged = SS.load_config()
station = dict(packaged["stations"][LABEL])
frames = station["frames"]["images"]
SOL = 52
context = [c for c in station.get("context", []) if int(c.split("_")[1]) == SOL]    # the mid-drive pair only
print(station["note"][:400], "...\n")
print(len(frames), "station frames,", len(context), "context frames; e.g.", frames[0])
""")

md(r"""
## 3  Download from the PDS Imaging Node — `rpp pds fetch`

`rpp.pds.fetch_products` takes the calibrated radiance products (`RAD`, 16-bit) and their PDS4 labels (`.xml`) into
the archive layout.  The labels carry what the pipeline needs besides the pixels: the rover's position and attitude
from its own navigation (site, drive, the local level frame), the mast's pointing, the camera temperatures and the
exposure.  The front Hazcam frames at the station come along (`rpp pds fetch --cameras fcam`): the panoramas use
them to fill the ground next to the wheels that the Navcams cannot see.
""")

code(r"""
from rpp import pds
names = [Path(n).with_suffix(".IMG").name for n in frames + context]
rep = pds.fetch_products(names, PDS)
print({k: rep[k] for k in ("downloaded", "kept", "bytes")}, "| failed:", rep["failed"] or "none")
haz = pds.fetch_sols(PDS, [SOL], cameras=("fcam",), match="N0032046")    # the front Hazcams at site 3, drive 2046
print("front Hazcam:", haz["sols"], {k: haz.get(k) for k in ("downloaded", "kept")})
# what else the Navcams took on sol 52 at site 3 (names only, nothing downloaded): other mid-drive stops show up here
import re
listed = pds.fetch_sols(PDS, [SOL], cameras=("ncam",), match="N003", list_only=True)["names"]
drives = sorted({m.group(1) for n in listed for m in [re.search(r"_N003(\d{4})NCAM", n)] if m})
print("sol 52, site 3 Navcam drives:", drives)
""")

md(r"""
## 4  The terrain mask model and the camera models

* **Masks** (`rpp mask fetch`): a ConvNeXt segmentation network trained on hand-cleaned masks of Navcam, Hazcam,
  Mastcam-Z and WATSON frames marks the rover (deck, mast, wheels, arm) and the sky out of every frame before the
  features are found, so the alignment and the terrain are the terrain's.
* **Camera models** (`rpp cmods status`): the Navcam eyes as OpenCV thin-prism fisheye models with their temperature
  laws (focal length per degree C, the stereo rig's rotation per degree C and its slow drift over the mission), fitted
  jointly over many blocks and *promoted* only when a list of checks passes (residuals, the change against the models
  in use, block agreement, finite covariance).  A block starts from them; the models in use have a fingerprint that
  goes into every run's record.
""")

code(r"""
rpp_main(["mask", "fetch"])
rpp_main(["cmods", "status"])
""")

md(r"""
## 5  Select and process the products — `rpp process`

A *site* is a set of sols; this one is not in `sites.json`, so it is given in the settings (`sites_extra`).  One
product per exposure is chosen (the largest sub-frame, the highest version), radiometrically normalized to 8 and 16
bits, masked, and listed in a manifest with its label's facts.  (The panoramas are blended later from 16-bit linear
copies of the frames, `colmap/images16/`, not from the 8-bit ones.)
""")

code(r"""
from rpp.pipeline import Settings, resolve_site, select_products, process_site
from rpp.sfm.workdir import load_manifest
SITE = "van_zyl_tiny"
settings = Settings(
    pds_dir=PDS, scapes_root=SCAPES,
    sites_extra={SITE: {"sols": [[SOL, SOL]], "label": "Van Zyl Overlook tiny (example)", "zcam": []}},
    add_nearby_waypoints=0,        # only what was downloaded
    hazcams=("F",),                # the front Hazcams at the block's stations; () = Navcam only
    workers=4,
)
site, settings = resolve_site(SITE, settings)
print(site.name, "sols", site.sols_text, "->", site.work)
paths, selection = select_products(site, settings)
print(len(paths), "products selected")
processed = process_site(site.work, settings)
manifest, _ = load_manifest(processed)
print(manifest["n_processed"], "processed;", len(manifest["failed"]), "failed")
""")

code(r"""
import matplotlib.pyplot as plt
from PIL import Image
m = next(x for x in manifest["images"] if "NCAM03052" in x["source_product"])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(Image.open(processed / m["outputs"]["PNG8"])); ax[0].set_title(Path(m["source_product"]).stem, fontsize=8)
if "mask" in m["outputs"]:
    ax[1].imshow(Image.open(processed / m["outputs"]["mask"]), cmap="gray"); ax[1].set_title("mask (white = terrain)")
for a in ax:
    a.axis("off")
print(json.dumps({k: m[k] for k in ("sol", "site", "drive", "LMST", "camera_temperature_degC") if k in m}, indent=1))
""")

md(r"""
## 6  Align the block — `rpp align`

One COLMAP camera per eye from the promoted models (the camera temperature picks the focal length), SIFT features at
native resolution inside the masks, matching, triangulation from the rover's poses, and a weighted bundle adjustment
that keeps the rover's positions and attitudes as priors with their uncertainties.  The run has a *run key* (a hash of
the inputs and settings that change the result); a finished run with the same key is not done again.
""")

code(r"""
from rpp.pipeline import align, load_settings, load_results
from rpp.sfm import health_table
WORK = site.work
settings = load_settings(WORK, base=settings)
report = align(WORK, settings)
proj, rec, report, summary = load_results(WORK, settings.variant)
print("run key", report.get("run_key"), "| verdict", report.get("verdict"))
print(health_table(report))
""")

md(r"""
**The health checks** have warn and fail thresholds set before the data were seen (provisional, not validated
limits): the share of frames registered, the reprojection residuals per eye, the scale against the rover's own
odometry, the camera models' change, and more.  A failing block goes no further - fix the alignment first.
`<project>/health/` has the table, the figures and a map of the stations.
""")

md(r"""
## 7  The station products — `rpp single-sites`

The station steps come from a configuration (the packaged one's defaults, with this station's block); `--dry-run`
prints the commands, which are ordinary `rpp` commands:

* `rpp dense` - the dense cloud of the block's Navcam frames (COLMAP patch-match stereo through pycolmap, the rover
  masked);
* `rpp mesh` - the terrain, built once from the rover-masked cloud and shared by both variants: a height field within
  4 m of the mast (it cannot fold over a hole's rim), Poisson surfaces per zone beyond, holes filled by a smooth sheet
  hung under their rims; the with-rover variant (`--rover`) adds the Perseverance model where the rover stood (its
  pose from the fitted mast chain), the terrain-only variant (`--mask-rover`) does not;
* `rpp depthmaps` - every frame ray-cast into the scene: depth, normal and label maps;
* `rpp ods` - the stereo pair: each eye's rays (64 mm apart, on the ODS circle) cast into the scene and every hit
  colored by a weighted blend of the frames that see it (16-bit linear radiance; the provenance of every pixel kept),
  the far field beyond the mesh fitted from the frames, the sky; and the textured 3D station models.

Each step writes `step_<name>.json` with a key of its command, its inputs and the code: running the cell again skips
what is done.
""")

code(r"""
cfg = {k: v for k, v in packaged.items() if k != "stations"}
own = {k: station[k] for k in ("station", "frames", "note") if k in station}
own["context"] = context                           # the mid-drive pair only (see section 2)
cfg["stations"] = {LABEL: dict(own, work=str(WORK), title="Van Zyl Overlook (example)")}
has_f = any(Path(x["outputs"]["PNG8"]).name.startswith("FLF_") for x in manifest["images"])
if not has_f:                                      # no front Hazcam at the station: nothing to fill the ground with
    cfg["ods_args"] = [a for a in cfg.get("ods_args", []) if a not in ("--fill-cameras", "F")]
    cfg["depthmaps_args"] = [a for a in cfg.get("depthmaps_args", []) if a not in ("--add-cameras", "F")]
CFG = DATA / "van_zyl_single_sites.json"
CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
args = ["single-sites", LABEL, "--config", str(CFG), "--root", str(SCAPES), "--out", str(STATIONS), "--publish", str(WEB)]
rpp_main(args + ["--dry-run"])
""")

code(r"""
rc = rpp_main(args)                    # dense, mesh, depthmaps, ods (two variants), the 3D models, the viewer list
print("exit", rc)
OUT = STATIONS / LABEL
print(Path(STATIONS / "single_sites_summary.md").read_text(encoding="utf-8"))
""")

md(r"""
## 8  Look at the products

The mesh from above, the left eye of the terrain-only panorama, its provenance (which frame colored each pixel) and the
geometry panoramas (range and surface normal) that check the model without any image in them.
""")

code(r"""
from IPython.display import Image as NbImage, display
for f in ("mesh_preview.png", f"{LABEL}_360_3DTD.jpg", f"{LABEL}_normal_360_3DTD.jpg", f"{LABEL}_range_log_360_3DTD.jpg"):
    p = OUT / f
    print(f, "-", "missing" if not p.is_file() else f"{p.stat().st_size / 1e6:.1f} MB")
    if p.is_file():
        display(NbImage(filename=str(p), width=1000))
ods = json.loads((OUT / "ods.json").read_text(encoding="utf-8"))
print(json.dumps({k: ods[k] for k in ("width", "ipd_m", "sources", "seconds") if k in ods}, indent=1)[:1500])
""")

md(r"""
## 9  The 3D models

Two binary glTF files per station, in meters about the mast pivot (x east, y up, z south):

* `<station>_station_mesh.glb` - the terrain within 10 m (about 300,000 triangles, one 8192 x 8192 texture) and the
  rover (its own 4096 x 4096 texture).  The terrain texture is a *projection from the viewer's place*: its center is
  straight down from a point 0.5 m above the pivot, and its radius grows evenly with the angle from straight down, so
  every texel covers the same angle seen from there and the near ground gets the most texels per square meter;
  triangles that point cannot see get cells of their own in the corners.
* `<station>_station_scape.glb` - the same, but the texture runs on to the horizon and the sky: the terrain up to the
  skyline fills 90 % of the disc's radius, the sky the rest, and a sheet sampled from the scene carries it from 7 m to
  the far field and a dome at 3 km.  One surface from under the rover to the sky, with true parallax everywhere.
""")

code(r"""
from rpp.ibr.rover_surface import read_glb
for name in (f"{LABEL}_station_mesh.glb", f"{LABEL}_station_scape.glb"):
    f = OUT / name
    if not f.is_file():
        print(name, "missing"); continue
    doc, _ = read_glb(f.read_bytes())
    tris = {m["name"]: doc["accessors"][m["primitives"][0]["indices"]]["count"] // 3 for m in doc["meshes"]}
    print(f"{name}: {f.stat().st_size / 1e6:.0f} MB, {len(doc['images'])} textures, triangles {tris}")
rec_scape = json.loads((OUT / f"{LABEL}_station_scape_terrain.json").read_text(encoding="utf-8"))
print({k: rec_scape.get(k) for k in ("skyline_deg", "texels_per_deg", "texels_per_deg_around_horizon", "shell", "fade_m")})
display(NbImage(filename=str(OUT / f"{LABEL}_station_scape_terrain.jpg"), width=700))
""")

md(r"""
## 10  Color for display (a study)

The panoramas keep the blend in 16-bit linear radiance (`*_ods_left_16.png`; `ods.json` "linear16" gives the scale).
The 8-bit views are display transforms of it.  `rpp_studies.misc.color_enhance` makes three, each a fixed function
of the data with one parameter set per station written to a JSON - nothing learned, nothing painted:
A, base/detail (an edge-preserving split of log luminance: the large scale compressed, the detail multiplied, each
pixel's color ratios kept); B, Mantiuk et al. 2006; C, a decorrelation stretch of the terrain (the sky and the rover
left out).
""")

code(r"""
from rpp_studies.misc import color_enhance as CE
CE.run(str(OUT / f"{LABEL}_ods_left_16.png"), str(OUT / "color"), LABEL,
       depth=str(OUT / f"{LABEL}_ods_centre_depth.npz"),
       rover_depth=str(OUT / "with_rover" / f"{LABEL}_rover_ods_centre_depth.npz"))
for k in ("reference_gamma", "A_base_detail", "B_mantiuk", "C_decorrelation"):
    print(k); display(NbImage(filename=str(OUT / "color" / f"{LABEL}_pano_{k}.jpg"), width=1000))
""")

md(r"""
## 11  The viewer

`rpp single-sites --publish` copied the page (`index.html`, three.js in `vendor/`) into `WEB` and listed the station in
`panos.json`.  The cell below serves it on this computer.  The modes (the second menu):

| mode | what is drawn |
|---|---|
| terrain only | the rover-masked stereo pair |
| terrain with rover (projected) | the with-rover stereo pair |
| terrain with rover (3D model) | the terrain pair and the textured rover model |
| 3D terrain and rover | the station mesh within 10 m, the terrain pair beyond |
| 3D terrain to the horizon and rover | the scape model alone, from under the rover to the sky |
| Drive the rover (simulation) | the scape model with the rover driven over it (skid steering: arrow keys, or the two thumbsticks) |

On a screen the eye moves a little in the 3D modes (in front of the turning point, with the mouse, a slow sway when
left alone; P turns it off), so the near terrain and the rover shift against what is behind them.  A headset needs
the page over https (WebXR): publish to a web host (`--upload` puts the pictures on Hugging Face and the list on
GitHub Pages) or serve it with a certificate.
""")

code(r"""
import functools, http.server, threading
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
server = http.server.ThreadingHTTPServer(("localhost", 8000), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
for key in ("masked", "terrain3d", "scape3d", "drive3d"):
    print(f"http://localhost:8000/#{LABEL}/{key}/standard")
# server.shutdown() when done
""")

md(r"""
## 12  The same block for other tools

* `rpp nerfstudio WORK` - a Nerfstudio data set (`transforms.json`, the undistorted frames and masks, the sparse
  points): the route to 3D Gaussian splatting of the station, with the same cameras and poses;
* `rpp metashape WORK` - a Metashape job (cameras, poses and matches) for a cross-check of the alignment.
""")

code(r"""
from rpp.sfm.nerfstudio import write_nerfstudio
from rpp.sfm.metashape import write_metashape_job
print("nerfstudio:", write_nerfstudio(proj, threads=4))
print("metashape:", write_metashape_job(proj))
""")

md(r"""
## What the design gives you

* **Traceable**: PDS product names are kept from download to panorama; every product has a JSON record of its
  inputs, settings, code version and timings; every panorama pixel's provenance (which frame) is saved.
* **Calibrated, not tuned per scene**: the cameras come from mission-wide joint calibrations with temperature laws,
  promoted only through explicit checks; a block refines what its data can support and reports what it changed.
* **Checked before use**: provisional health thresholds stop a bad alignment before any product is made from it.
* **Repeatable and resumable**: run keys and step keys - the same inputs and code give the same products, and a
  finished step is not done again.
* **Physical units**: the panoramas blend radiance (16-bit linear); display transforms are separate and documented.
* **One terrain, several products**: the same mesh feeds the depth maps, both stereo panoramas, the 3D models and the
  exports, so they agree with one another.
""")


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "van_zyl_ods_example.ipynb"
    nbf.validate(nb)
    out.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"written: {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
