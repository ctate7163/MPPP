# Audit of 19 Oct 2026 (0.90.0, during the v0p86 run) and the plan for 0.91 → 1.0

Written at 22:30 PT while `run_v0p86` was in `navcam_runs` (7 of 16 blocks finished). The numbers below come
from those 7 blocks, the 18 Oct pass-2 consensus (promoted, 23 blocks) and the 18 Oct waypoint check; the 6 AM
update adds the rest of the run.

## 1. What the results say

### 1.1 Blocks against the joint model (new in 0.90: `mppp.calib.fit.navcam_block_table`)

Refined minus the promoted model at each temperature bin's own temperature and the block's sol:

| block | sol | T °C | images | bins | NL fx fy ppm | NR fx fy ppm | rig pitch roll mdeg | agrees |
|---|---|---|---|---|---|---|---|---|
| OEB Landing Site | 31 | −21.8 | 511 | 19 | +22 −20 | **−106 −97** | +4.1 +3.1 | no |
| Rochette | 180 | −20.3 | 42 | 8 | −41 −5 | +6 +24 | +1.5 +1.0 | yes |
| Seitah North Large | 278 | −15.8 | 295 | 43 | −28 −66 | **−104 −146** | +0.7 +0.9 | no |
| Issole Seitah South | 316 | −13.9 | 128 | 24 | −11 +5 | **−102 −102** | −0.2 +0.8 | no |
| Artuby Ridge | 342 | −19.0 | 62 | 9 | −22 −21 | +76 +67 | −0.9 −2.2 | yes |
| Chal Rocks | 360 | −19.3 | 83 | 14 | −58 −77 | −14 −66 | +1.4 +2.6 | yes |
| Enchanted Lake | 556 | −23.7 | 106 | 20 | +33 +33 | +43 +52 | −0.8 +1.8 | yes |

- **NL agrees everywhere** (|Δf| ≤ 77 ppm, mostly ≤ 40 ppm, ≈ 0.1 px): the thermal form and the early offsets
  are right for the left eye.
- **NR is −100 to −146 ppm at the three early blocks with the most images** (OEB, Seitah North Large, Issole:
  sols 31–316), and within ±70 ppm after sol 340. The early-mission offsets (`fx +0.197, fy +0.254 px`,
  one set for both eyes) fit NL; NR wants about −0.1 px instead of +0.2. **Finding: the early-mission term
  needs per-eye offsets** (4 → 8 parameters; the joint has 905 early images to fit them from).
- **Rig pitch and roll scatter ±2–4 mdeg about the drift model**, yaw is held. OEB (sol 31) is +4 mdeg in
  pitch — the early hinge (knot at sol 300) is fitted from 13 blocks; a steeper early pitch rate would fit OEB
  and Rochette alike. Keep watching with the 16 blocks.
- The principal point is held in the runs (cx, cy exactly 0): the table says nothing about it, by design.

### 1.2 Waypoint check — three corrections to the scale check (all in 0.90)

1. **Stations without a waypoint of their own** (`position_source` "site N drive 0 + label ORIGIN_OFFSET_VECTOR"
   — the rover's onboard estimate) entered the similarity fit and were flagged like surveyed ones. OEB has 12
   of 25 such stations, Rochette 3 of 6, Chal Rocks 5 of 10; their residuals are 0.6–1.8 m. Now only surveyed
   stations fit the scale; onboard stations are checked against the fit, reported, never flagged.
2. **The similarity was 3-D.** Waypoint elevations are decimetres rough; a free rotation turned them into a tilt
   that posed as rotation and scale (Enchanted Lake: 1.7°, −4300 ppm, with horizontal offsets of 0.1 m). It is
   now a yaw about the vertical with the scale in the east–north plane.
3. **No uncertainty on the scale.** It now carries a standard error from the scatter and the stations' lever
   arm. Tonight's 7 blocks: OEB −2366 ± 2969 ppm, Seitah North Large −177 ± 1040, Enchanted Lake −3994 ± 1045,
   the four small blocks translation-only. The 18 Oct table's ±3000 ppm block scales were noise of this size.

   Enchanted Lake is the remaining case: its 6 surveyed stations are two clusters (rover sites 23–24 at sols
   421–423, sites 27–29 at 556–586) 55 m apart with a 0.1 m offset between them — a localisation offset between
   rover sites, which a two-cluster block cannot tell from scale. **Plan: fit the scale over stations of one
   rover-site chain, and report the offset between chains** (`by_rover_site` already has the numbers).

   Consequence for the baseline study: with scale se of 1000–3000 ppm per block, the Navcam baseline (0.3 mm ≈
   700 ppm) is only measurable from the large single-chain blocks (Three Forks Large: 35 stations over 118 m;
   Seitah North Large; Bell Island Large; Origny Large) with proper weights — the study of 18 Oct used equal
   weights and found "too few blocks". The weights are now in `waypoint_flags.json`.

### 1.3 The consensus and the Mastcam-Z calibration

- Pass 2 of 18 Oct did **not converge** (f changed 184 ppm from pass 1; corners 0.48 px) and failed the
  **covariance** check (rank-deficient: the sd of the parameters is unavailable, so the promotion checks that use
  sd are skipped). Tonight's run is effectively pass 3, with k1(T) from the bins and the NR cx(T) term. If pass 3
  moves f by < 20 ppm it has converged; if it moves by ~100 ppm again, the iteration is chasing the NR early
  offsets of §1.1 and the per-eye term is the fix, not another pass.
- The rank deficiency is expected: the joint holds the rig translation and solves the principal points of both
  eyes with the rig yaw free — cx(NR) and yaw are the degenerate pair noted since 0.40. For the covariance,
  hold yaw (it is held in the runs anyway) or report the equivalent boresight instead of cx and yaw separately.
- The Z34 calibration of 19 Oct 01:00 was **not promoted** (too few passing blocks, 7 blocks with the v0p75
  start model). 0.87 changed the start model to the South Arm fit and the per-eye `zcam_guard`; the diagnosis
  (`zr_diagnosis.md`) shows the right eye now keeps every frame through stage 2. Expect a promotion decision
  from tonight's `zcam_calibrate`.
- **Matching** fell back to SIFT in this run (LightGlue not installed on the PC). The 0.90 path (kornia) is
  untested on the GPU PC; install it before the next run and compare one block's match count and time.

### 1.4 Health flags in the 7 blocks

- Artuby Ridge: `edge_to_centre_residual` 1.61 (warn 1.5): the outer quarter of the frame has 60 % larger
  residuals than the centre — the k3/k4 terms held at the consensus values do not fit this block's extremes of
  temperature (bins down to −45 °C in OEB; Artuby −19 °C though). Worth one look at residual-vs-radius per bin.
- Rochette: `attitude_residual_p95` 1.34° (the label pointing of one station); Issole: 8 outlier images (6 %).
- All 7 registered every image. The approach stations of Rochette: `unlocalized_registration` registered
  S006D2684 (581 inliers, 6.4 m shift) and failed S006D2648 (6 inliers) — the third-view rule is in but the
  2-image stations with < 10 inliers stay unplaced; the 10 m / 3° priors of 0.87 are the next step.

## 2. Code audit in light of the run

| found | state |
|---|---|
| `mppp.sfm.gpu` subprocess printed its result through `logging` after the print → logging conversion, so the parent read nothing (`check_gpu_python` IndexError). | fixed in 0.90 (the protocol lines are `print`) |
| Waypoint check: onboard stations in the scale fit; 3-D similarity; no scale se. | fixed (§1.2) |
| The waypoint table in use was the packaged snapshot unless someone ran `refresh=True`; a new station after the snapshot gets the onboard estimate silently. | `load_waypoints(refresh="auto")` in every processing run (cache ≤ 1 day), `mppp waypoints --refresh` in the run .bat, `python -m mppp` |
| v0p86 manifests record `intrinsics.source = "Metashape XML M2020_NR1_frame.xml"`: the processing still started from the XML frame models while the alignment used the promoted ones. | 0.90 removed the XMLs; processing takes the start camera from `mppp.calib` (the same model the alignment starts from). Every site reprocesses once (the run key changes). |
| `run_key` hashes the full `Settings`; adopting 0.90 realigns every site. | accepted (one more pass) |
| Tests: four stale after the 16-block group and the `_reconstruct` wrapper. | fixed; suite 480+ green (CPU) |
| Early-mission offsets are one set for both eyes. | **open — 0.91** (§1.1) |
| The baseline study weights blocks equally. | **open — 0.91**: weight by `scale_se_ppm`, single-chain blocks only |
| Hard-linked processed images (`images_png8` ↔ `colmap/images`) cannot be read through the file bridge; harmless for the pipeline. | note for the paper figures: copy one frame into `paper/` |

## 3. Plan

### 0.91 — the next run (after this one finishes and 0.90 is adopted)

1. **Per-eye early-mission offsets** in the joint (`calib.joint.early_keypoint_map`, `fit_early_offsets`,
   the camera file's `early_mission` with `NL`/`NR` sub-entries; `camera_at` reads either form). Test: the
   7-block table above with NR within ±50 ppm.
2. **Covariance**: hold the rig yaw in the joint (as in the runs) so the normal matrix is full rank and the sd
   checks run; report the stereo boresight (cx(NR) − cx(NL) × f ratio, yaw) as the stable quantity.
3. **Waypoint scale per rover-site chain**, between-chain offsets in the report; the baseline study weighted by
   the scale se and restricted to blocks with se < 1000 ppm.
4. **Unsurveyed stations**: the 10 m / 3° priors of 0.87 are in; add the health line (solved shift vs prior σ)
   and the chained start position from the nearest surveyed drive (handoff §3.5).
5. **LightGlue on the PC** (`install_lightglue.bat`), one block A/B against SIFT (matches, inliers, time).
6. Mastcam-Z: if `zcam_calibrate` promotes tonight, the 0.91 run uses it; if not, the per-zoom line fits need
   the bins' focus coverage reported per block before anything else changes.
7. Paper: per-eye offsets and the block table are a paragraph and a figure; figure 1 combined, figure 3
   (station map with the sparse cloud from `health/station_map.png` of Seitah North Large), the opacity
   citation, the generative-AI statement.

### 1.0 — the JOSS release

- The CI architecture of `docs/CI_ROADMAP.md`, in this order and with gates taken from this run:
  1. **Stage 1 (in place):** unit tests on 3.11/3.12, pyflakes, the paper build.
  2. **Stage 2 — the Rockytop test set** on Zenodo (Navcam 9 stations over 59 m, sols 460–530; Mastcam-Z 34 mm
     mosaic): `mppp fetch rockytop`; the nightly `mppp run rockytop --zcam` with gates: every image registered,
     median residual ≤ 0.25 px, `edge_to_centre_residual` ≤ 1.5, block-vs-joint |Δf| ≤ 100 ppm and rig ≤ 5
     mdeg (the table of §1.1 as a test), waypoint scatter ≤ 0.3 m, run time ≤ 45 min on the runner.
  3. **Stage 3 — Windows job** (the `.bat` wrappers call the CLI only) and the GPU runner on demand.
  4. **Stage 4 — release workflow** with the Zenodo DOI; `CITATION.cff` checked against the tag.
- Package: `mppp.error` either reaches its own tests on the Rockytop export or ships as `mppp.error` "under
  development" with the paper's one sentence — not both; `studies/` stays out of the wheel.
- Docs: a docstring site (mkdocs) from `docs/` in CI; `docs/methods.md` rewritten for the frozen forms (its body
  still narrates the rational and polynomial models).

### What to decide (Christian)

- Per-eye early offsets: go ahead for 0.91? (My reading of §1.1 says yes.)
- The 12 "moved, not deleted" items of the 0.90 delivery note.
- Whether the Rockytop test set may be published (PDS products are public; the sites file entry is ours).
