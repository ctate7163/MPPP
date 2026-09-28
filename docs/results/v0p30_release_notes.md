# MPPP v0.30.0 — release notes and findings (28 Sep 2026)

Repo `D:\code\MPPP`, scapes `D:\scapes\colmap`. Bundle `release_v0p22\mppp_v0p30p0.bundle` (tag `v0.30.0`). Notebooks `*_v0p30p0.ipynb` in `D:\code\MPPP\notebooks`.

## What changed

Image selection and processing
- Navcam brightness ×0.9 (`color.brightness_by_family = {"N": 0.9}`; notebook 03 `NAVCAM_BRIGHTNESS`), applied with the white balance. Mastcam-Z unchanged.
- LMST window `selection.lmst_window_h = [9, 17]` (inclusive) and saturation limit `selection.max_saturated_fraction = 0.05` (fraction of valid pixels at the product's max DN, 32767 for 16-bit RAD). Both raise a `SkippedImage` subclass (`LmstOutOfWindow`, `SaturatedImage`, like `SkyImage`) before radiometry and mask inference; the manifest lists them under `skipped` with the reason, and every processed image carries `lmst_h`, `saturated_fraction`, `brightness`.
- `process_images(workers=4)`: spawned worker processes, each loads the mask model once; sequential fallback if the pool cannot start (e.g. `__main__` from stdin) or dies.

Reconstruction
- BA problem build ~2× faster (per-image lookups hoisted); `linear_solver="auto"` = dense Schur ≤600 frames, iterative Schur (Schur-Jacobi) above. Ceres is already multi-threaded; on the 2-core sandbox the solvers tie, on the user's PC dense Schur should win for the usual 100–400-frame blocks.
- Exhaustive matching `block_size=100` (COLMAP default 50). SIFT extraction/matching is already GPU (or CPU-threaded) — there is no further speed-up from Python-side workers.
- Start cameras `M2020_NL/NR_rational.json`, `M2020_N_rig.json` replaced by the five-site 0.22.4 consensus (Three Forks, Belva, Bell Island, Olifants, Marble Mountain). Deltas vs the old shipped set ≤0.35 px; per-site repeatability 0.14–0.47 px rms (corners dominate); rig rotvec Δ ≤0.15 mrad.

Health and viewing
- `health/station_map.png`: top-down site view with prior→refined camera shifts; shown in notebook 03 §7 (via IPython Image, so it no longer falls to the Agg `plt.show()` warning).
- `block_rotation_deg` check (median world-frame rotvec of `R_prior.T @ R_refined`; warn 0.3°, fail 1°) plus `report["world_frame"] = {frame: "ENU", offset_enu_m, block_rotation_deg, ...}` — the alignment stays in the ENU frame the attitude/position priors define; the offset is recorded, never applied silently.
- Tie-point colours = mean of all valid track observations (`export.point_colors_from_tracks`); invalid/black pixels excluded; no-sample points mid-grey.
- `open_in_colmap.bat`: single click opens COLMAP GUI with database + images + refined model. Search order: notebook `COLMAP_BAT` (stored in `project.settings["colmap_bat"]`), `COLMAP_BAT_CANDIDATES` (incl. `D:\tools\colmap-x64-windows-nocuda\COLMAP.bat`), env `COLMAP_BAT`, PATH.

Notebook 03 (`03_colmap_alignment_v0p30p0.ipynb`): concise first cell with the 17-site table; `SITE="threeforks"`; `KEEP_ONLY_REMAINING=False`; `ADD_NEARBY_WAYPOINTS=10`; `SKY_ELEVATION_DEG=10`; `MAX_NUM_FEATURES=16000`; `MATCH max_distance=1.0`; 3-round schedule (24,10,8)/(12,4,4)/(8,2,2); `ATTITUDE_PRIOR_DEG=5`; new `WORKERS`, `COLMAP_BAT`, `LMST_WINDOW_H`, `MAX_SATURATED_FRACTION`, `NAVCAM_BRIGHTNESS`, `MATCH_BLOCK_SIZE`, `LINEAR_SOLVER`; `WORK = SCAPES_ROOT / (f"{SITE}_colmap_nav_zcam34" if INCLUDE_ZCAM34 else f"{SITE}_colmap")`.

## Review of the saved v0.22.4 runs

belva_crater (5 stations): WARN prior_scale 2.13 %, whole-block rotation 2.29° under the 5° attitude prior — the weakest network; marble_mountain WARN scale 1.94 %; threeforks WARN 16 excluded frames; bell_island WARN attitude p95; olifants PASS. The SKY rule at 10° removes the NCAM00501 sky-survey frames. Belva's 2.3° rotation is why `block_rotation_deg` now exists: with a loose attitude prior a small block can rotate as a whole and still pass every other check.

## Lens model: rational vs fisheye (working notes §11, `scripts/lens_model_experiment.py`)

Three Forks / Bell Island, same observations, cameras re-initialised per model:

| model | cost TF / BI | corner median px | field corner cells px | TF-vs-BI camera diff (shift removed) NL/NR |
|---|---|---|---|---|
| rational k1–k4 + p1,p2 | 71,724 / 196,977 | 0.255 / 0.228 | 0.078 / 0.062 | 0.42 / 0.71 |
| fisheye θ-poly, no tangential | 99,258 / 246,087 | 0.41 / 0.41 | 0.32 / 0.32 | 1.9 / 2.1 |
| fisheye + tangential (THIN_PRISM_FISHEYE, sx,sy held) | 70,628 / 195,659 | 0.210 / 0.208 | 0.025 / 0.054 | 0.16 / 0.45 |
| + thin prism free | 70,344 / 194,767 | 0.20 / 0.20 | — / 0.026 | 0.64 / 0.58 |

Findings: tangential terms are essential in either family (+25–38 % cost without them, as in §9). At equal parameter count the θ-polynomial beats the rational by 0.7–1.5 % in cost and 9–18 % in corner residual, and is more repeatable across sites. Thin-prism terms trade against the principal point (4–8 px common shift) — hold at zero. Decision: rational stays the v0.30 default (consensus, CAHVORE conversion, export, notebook 05 all built on it; gain is hundredths of a pixel in the corners). Candidate for a later version: `navcam_model="THIN_PRISM_FISHEYE"` end-to-end, judged on five-site repeatability.

## Consensus initialisation

Yes — shipped in v0.30. The five-site consensus is within 0.35 px of the previous start set and the per-site scatter (0.14–0.47 px rms) is smaller than the corner residual, so it is a better start and a sound hold value for weak networks (`NAVCAM_INTRINSICS="hold"`).

## Mask model export (user action)

The >20 MB safetensors cannot be delivered over the bridge. In the notebook kernel (mppp env):
`from mppp.mask.hub import export_safetensors; export_safetensors("convnext_tiny_s4_seg_20260925b.pt", name="mppp_mask_v3")`.
Until then `fetch_model` falls back to `checkpoints/convnext_tiny_s4_seg_20260925b.pt` automatically.

## Tests
`tests/test_v0p30.py` (10) plus the full suite; `test_v0p22_4::test_notebook_03_new_defaults` and `test_v0p22_2::test_notebook_03_offers_other_visits` relaxed to the new notebook defaults.

## Nine-site review (added the same day; details in working notes §12)

- All nine 0.22.4 blocks stay within 0.46° of the ENU frame the labels define (largest: Rockytop +0.40° in azimuth). The "rotation 2–3°" printed with the scale check was the waypoint layout turned against the solution, not the solution turned.
- Per-frame attitude scatter about the block rotation: median 0.10–0.35°, p95 0.23–0.69° (label pointing knowledge). New `attitude_residual_p95_deg` (warn 0.75°); the undivided p95 is now info only. `block_rotation_deg` sign corrected (world rotation prior → refined).
- Layout scale errors 0.1–2.1 % are 0.02–0.33 m at the stations: judged now as `prior_scale_error_m` (warn 0.5 m).
- Pearce Canyon's 16 unregistered images were right-only exposures: now kept in a one-sensor rig.
- Nine-scape consensus: repeatability 0.16–0.50 px, Taylorfjellet 0.71 px (focal +1.4 px in both eyes). Label focal does not vary with temperature; manifest now records `camera_temperature_degC`, notebook 05 §2a fits focal vs temperature. The type-3 CAHVORE fit of the NL consensus had failed (linearity 0.99, O tilted 51°, rms NaN): now multi-start.
- Notebook 03: every results cell prints the site, sol range and COLMAP folder first (`banner()`).
