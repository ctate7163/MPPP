# Plan for MPPP 0.91 (written 20 Oct 2026, after the 0.90 adoption)

0.90 is on the PC and on GitHub (main = 0.90 + the 20 Oct Navcam consensus; the four-block Z34 run is going).
0.91 is the next development release: the joint as one adjustment, the rig fitted for yaw and roll too, the site
maps in the package, the CLI carried further, and the fixes the 0.90 audit listed. Order of work at the end.

## 1. The Navcam joint as one bundle adjustment (Christian, 20 Oct)

**The requirement.** One set of Navcam parameters for every temperature and sol, solved at once. Each parameter
that depends on temperature is an offset at −20 °C and a slope (NL fx = fx₀ + a·(T − T0), likewise fy, cx, cy, k1
per eye); each rig angle is an offset at sol 0 and a slope in sol (pitch with its early-mission hinge); the
distortion terms (k1–k4, p1, p2 per eye) and the per-eye early-mission offsets are constants; and *all of them,
with the block poses and the points, are optimised in one bundle adjustment* — no profiles, no sequential
refits, no separate between-block regression for the rig. The unknowns:

| group | parameters |
|---|---|
| intrinsics at T0, per eye | fx₀, fy₀, cx₀, cy₀, k1₀, k2, k3, k4, p1, p2 (sx1 = sy1 = 0) |
| thermal slopes, per eye | fx, fy (ppm/°C), cx, cy (px/°C), k1 (/°C) |
| early mission, per eye | Δfx, Δfy, Δcx, Δcy (px) for images before sol 380 |
| rig | rotation at sol 0 (3), rates in sol for pitch, yaw, roll (3), the pitch hinge (early rate, knot) |
| per block | poses, points (the rig epochs are no longer free: an epoch's rig is the trend at its sol) |

Every image's camera is then the law evaluated at its own temperature and sol; one normal matrix gives the
covariance of the whole set.

**Afterwards, the figures.** The per-block, per-temperature-bin and per-sol values are *diagnostics* drawn over
the global solution: with the global parameters held, each bin (per block and eye) and each rig epoch is refined
alone, and the refined-minus-global values go on the refined figures — the bins on `plot_intrinsics_vs_temperature`
(the line is the global law), the epochs on `plot_rig_trends` (the line is the global trend referred to sol 0),
the blocks in `navcam_block_table` — so the figures show the scatter about one solution instead of a sequence of
passes.

**How, in pyceres.** COLMAP's reprojection functors take a camera parameter block as it is; a camera that is a
function of (T, sol) therefore needs one of:

1. **Image-group camera blocks tied to the law** (recommended): the bins stay camera blocks (as the runs have
   them), the rig epochs stay rig blocks, and *constraint residuals* tie each of them to the global parameters
   — `p_bin − law(p₀, slopes, early; T_bin, sol_bin)` and `R_epoch − trend(R₀, rates; sol)` with a σ small enough to
   be a constraint. The law's parameters are ordinary parameter blocks of a small Python cost functor (one
   residual per bin and epoch, a few hundred in all, cheap), the 600k reprojections keep COLMAP's compiled
   functors, and Ceres solves everything together. Rig epochs with too few frames need no rule any more: they
   contribute what they can and the trend holds them.
2. A Python reprojection functor over the law directly — exact, but ~100× slower at this size.
3. The outer loop kept, but as one Gauss–Newton over all global terms with numerical Jacobians — the fallback.

Deliverables: `calib.joint.joint_adjust_all` (option 1), `calib.fit.fit_navcam` using it (the profile code kept
only as a cross-check), the model files with `thermal`, `early_mission` per eye and `rig.drift` with `sol0 = 0`
and the fitted yaw/roll rates, the covariance of the global parameters in `consensus.json`, and the diagnostic
pass that produces the bins/epochs for the figures.  Acceptance: a second joint on the same blocks changes nothing
(the convergence limits are then a check, not a stage); per-eye early offsets bring NR within ±50 ppm on OEB,
Seitah North Large and Issole; the figures from the diagnostic pass.

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
- Mastcam-Z: with the 20 Oct four-block calibration promoted, the 11-block group runs next (`zcam34_consensus`:
  Chal Rocks, Rochette, Issole, Three Forks North, Enchanted Lake, Rockytop, Belva Crater, Bunsen Peak, South
  Arm, Taylorfjellet, Marble Mountain); the per-zoom line fits report their focus coverage per block.
- **Mastcam-Z p1, p2 refined** (form `zcam-v2`): on Rochette's 20 Oct block the tangential residual field is
  ZL p1 +7.5e-4, ZR −4.2e-4 (0.36 / 0.16 px at the frame corners, 0.095 / 0.058 px rms over the observations,
  consistent in sign over every focus bin), beyond what cx, cy absorb. Per eye, constant over focus, refined in
  the block runs and averaged in the calibration like k1, k2; `MODEL_LIMITS` get a `max_p` and a between-block
  scatter check for them.
- Paper: per-eye offsets and the block table as a paragraph and a figure; figure 1 combined, figure 3 from the
  station map, the opacity citation, the generative-AI statement, figure sizes.
- CI stage 2 (Rockytop test set on Zenodo, nightly end-to-end run with the gates of the audit).

## 6. Order of work

1. §2 run settings (roll held, epoch rule) and the per-eye early offsets — small, and they change the next run.
2. §3 the maps port (self-contained; acceptance on the PC).
3. §1 the one-adjustment joint (the largest piece; option 1 first).
4. §4 the CLI around what §1–3 produced; `.bat` files as one-liners.
5. §5 as the runs allow; the paper items last, against 0.91's numbers.
