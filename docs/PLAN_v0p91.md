# Plan for MPPP 0.91 (written 20 Oct 2026, after the 0.90 adoption)

0.90 is on the PC and on GitHub (main = 0.90 + the 20 Oct Navcam consensus; the four-block Z34 run is going).
0.91 is the next development release: the joint as one adjustment, the rig fitted for yaw and roll too, the site
maps in the package, the CLI carried further, and the fixes the 0.90 audit listed. Order of work at the end.

## 1. The Navcam joint as one bundle adjustment

**Now.** The joint (`calib.fit.fit_navcam` / `calib.joint`) solves the merged blocks for the two cameras, the rig
and the points, and gets the *global* terms — fx(T), fy(T) per eye, cx(T) per eye, k1(T) from the bins, the rig yaw
rate, the four early-mission offsets — one at a time from cost profiles (five adjustments per term, a parabola
through them), then refits; the rig pitch and roll drift are not in the joint at all but a between-block regression
on the blocks' rig epochs (`fit_rig_drift`), and the knot sits at sol 300 by hand. Each pass is ~90 adjustments and
the terms are solved sequentially, so correlated terms (NR cx(T) against the yaw rate; the early offsets of the
two eyes) converge slowly — pass 2 moved f by 50 ppm and the rig by 3.8 mdeg from pass 1.

**Target.** One adjustment whose unknowns are: the two cameras at T0 (12 + 12), the thermal terms (fx, fy, cx, cy,
k1 per eye: 10), the early-mission offsets *per eye* (8), the rig rotation at sol 0 (3) and its linear trends in
sol for pitch, yaw and roll (3; the pitch hinge rate and knot as 2 more if the data want it), the per-block rig
epochs tied to the trend (the epochs become residuals about the trend, not free rotations), the block poses and
the points. Every image's camera is then a function of its temperature and sol, and the covariance of the whole
set comes out of the one normal matrix.

**How, in pyceres.** COLMAP's cost functions take a camera's parameter block directly, so a camera that is a
function of (T, sol) needs one of:

1. **Per-image "virtual" camera blocks with linear constraints** — each temperature bin camera is its own block
   (as the runs do today), plus residual blocks that tie the bin's parameters to the global law
   (`p_bin − (p0 + a (T_bin − T0) + early)` with a small σ). This uses only COLMAP's cost functions and pyceres'
   existing residuals; the global terms enter as parameter blocks of a Python cost functor evaluated a few
   hundred times per iteration (cheap). The rig epochs tie the same way to `R(sol0) + rate·sol`.
   *Recommended: it is the current data structure plus constraint residuals, and it keeps Ceres' sparse solver on
   the 600k reprojections.*
2. A custom reprojection functor in Python taking (p0, thermal, early, pose, point) — correct but ~100× slower
   than COLMAP's compiled functors at this size; not for 2,700 images.
3. Keep the outer loop but make it a Gauss–Newton over all global terms at once (numerical Jacobian: one
   adjustment per term per iteration, ~35 adjustments per iteration instead of ~90 per pass, all terms moving
   together). A fallback if (1) stalls.

Deliverables: `calib.joint.joint_adjust_all` (option 1), `calib.fit.fit_navcam` using it with the profiles kept
only as a check, the model files carrying `rig.drift` with `sol0 = 0`, yaw and roll rates with their sd, and
the covariance of the global terms; `navcam_block_table` and `plot_rig_trends` reading the same rig file.
Acceptance: pass-to-pass change below the convergence limits (20 ppm, 0.05 px, 0.02 px corners, 1 mdeg) in two
passes on the 23 blocks; per-eye early offsets bring NR within ±50 ppm on OEB / Seitah North Large / Issole.

## 2. The rig: linear yaw and roll trends, fitted

- The runs hold yaw to the model and free pitch and roll per epoch; roll is poorly determined per epoch (±3 mdeg
  τ, 16 mdeg between epochs of one block). 0.91: **hold roll like yaw** in the runs (`navcam_rig_roll: hold`),
  free pitch per epoch; the joint (§1) fits the yaw and roll rates and the pitch trend with the hinge, all
  referred to sol 0, and `rig_at` applies them. The rig-trends figure keeps its format; its drift line is the
  joint's.
- Epoch rule: an epoch with fewer than 8 frames or a single station gets no rig of its own (tied to the nearest
  epoch) — the Rochette sol-341 case; same rule in the rig-residual study.
- Per-site exclusions stay the documented way to drop a station (`"exclude": ["S009D1914"]` in `sites.json`).

## 3. The M20 site maps into `mppp.waypoints.maps`

Per the handoff `claude/m20-maps-handoff-2026-10-20-for-v0p90.md` (its §6 is the specification; the code of
20 Oct is in `claude/m20-maps-code-2026-10-20/`, the reference PNGs in `Claude outputs\m20_maps_reference_20oct\`):

1. `mppp/waypoints/maps/{context,blocklist,background,figures,refdata}.py` — figure bodies verbatim, module
   globals → `MapContext`, paths → arguments, `rc_context` for the font scale, `RID` on a copy, one process or
   `plt.close('all')` per figure.
2. The terrain reader = `waypoints.terrain.terrain_grid` with the deterministic order (levels as given, files
   sorted by name) and the handoff's `draw_terrain` shading (0.55/0.45 × 0.86, `Z[::-1]` flip, contours 0.3/0.25)
   replacing `shaded_relief`; test: identical grid for two shuffled file orders.
3. `blocks_from_sites` from `official_blocks`; `plot_site_map` becomes a thin alias of `figures.traverse_map` and
   `paper_figures.py` calls the ported function (the 0.90 redraw goes).
4. `mppp maps all|zooms|figures …` with the §4 settings (`M20_DEMS`, `M20_ORTHOS`, `M20_BACKGROUND`, `M20_TAG`,
   `--sites`, `--terrain`), default output `Claude outputs\maps_<yyyymmdd>`; `scripts\windows\make_maps.bat` from
   the draft; extra `[maps] = matplotlib, scipy, adjustText, rasterio, pyproj`.
5. Tests of §6.7; acceptance of §6.8: the six PNGs at the handoff's pixel sizes from the 36-block table, then one
   render with the 1 m DTMs and one with `--background ortho` on the PC (never tested on the real rasters).
6. Waypoint snapshot: check the prototype's `M20_waypoints.json` and the package's give the same `LocalFrame` fit
   before the package's (refreshed daily now) becomes the map's source.

## 4. The CLI

- `mppp maps` (§3), `mppp calib` (`navcam consensus`, `zcam calibrate`, `rig refit`, `promote` — the study
  commands that the pipeline calls, as subcommands with the same options), `mppp pipeline` (the stage runner of
  `studies/calibration/run_pipeline.py` as a command: `run`, `status`, `stop`), `mppp figures` (the refined
  figures of `scripts/paper_figures.py`).
- `mppp sites`: `add`, `group`, `stats` (the waypoint statistics by the block rule) so `sites.json` is edited
  through the formatter, not by hand; `mppp waypoints` with `--age` and `--check` (the stations of a site that
  are in the table vs onboard).
- Every `.bat` becomes one line over the CLI (`docs/WINDOWS.md`); `MPPP_PDS`, `MPPP_SCAPES` read from the
  `mppp_*.txt` files as well as the environment, and `mppp doctor` prints what is set and what is missing (the
  10:43 failure was `MPPP_PDS` unset).

## 5. From the 0.90 audit (still open)

- Waypoint scale per rover-site chain; the baseline study weighted by `scale_se_ppm`, blocks with se < 1000 ppm.
- Unsurveyed stations: the health line (solved shift vs prior σ); the chained start from the nearest surveyed
  drive.
- The aspect question: hold fy = fx × the joint's aspect in the block runs and see whether Bunsen Peak's and
  Cheyava Falls' residuals rise.
- LightGlue through kornia on the PC: one block A/B against SIFT (matches, inliers, time); the ONNX path is gone.
- Mastcam-Z: with the 20 Oct four-block calibration promoted, the seven-block group (`zcam34_consensus` now
  has Three Forks North, Belva Crater and Taylorfjellet ready) runs next; the per-zoom line fits report their focus
  coverage per block.
- Paper: per-eye offsets and the block table as a paragraph and a figure; figure 1 combined, figure 3 from the
  station map, the opacity citation, the generative-AI statement, figure sizes.
- CI stage 2 (Rockytop test set on Zenodo, nightly end-to-end run with the gates of the audit).

## 6. Order of work

1. §2 run settings (roll held, epoch rule) and the per-eye early offsets — small, and they change the next run.
2. §3 the maps port (self-contained; acceptance on the PC).
3. §1 the one-adjustment joint (the largest piece; option 1 first).
4. §4 the CLI around what §1–3 produced; `.bat` files as one-liners.
5. §5 as the runs allow; the paper items last, against 0.91's numbers.
