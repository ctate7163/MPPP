# mppp_error — Mars Photogrammetric Precision Prediction, error model (v0p15)

A priori 3D precision prediction for rover-based surface reconstruction, from
station geometry alone. Method-agnostic: the same machinery covers stereo DEM,
SfM/MVS, and radiance-field targets, because all three are limited by how the
camera rays sample the surface.

`python -m mppp_error selftest` → **146/146 checks**, including a comparison against an
independently written naive reference implementation.

## Purpose

Given a set of camera stations (real rover waypoints or a hand-placed test
pattern) and an instrument model, predict — *before* any images are matched or
any reconstruction is run — the 3D precision achievable at every point on the
ground, under several assumptions about how well cross-station matching
actually works. Concretely, this module answers:

- **How much does cross-station correspondence (SfM/MVS) buy over
  operational fixed-baseline stereo**, and how does that depend on station
  spacing?
- **Is a candidate camera network well-conditioned** — connected, redundant,
  not one failed match away from splitting into unregistered pieces?
- **What does telemetry/VO/bundle-adjustment pose error cost** in final
  reconstruction precision, as a function of how far a station is from the
  anchor (by real driven path, not straight-line distance)?
- **How do these answers change** across error metrics (vertical precision,
  anisotropy, differential/slope error, normalized precision) and across
  acquisition geometries (spacing, instrument, real vs. synthetic)?

It is a *planning and diagnostic* tool, not a reconstruction pipeline: it
never touches actual imagery. `mppp_error.colmap` is the one bridge to real
reconstructions, for calibrating the model's free parameters (ε, θ_max, θ_c)
against an actual COLMAP solve.

## Key assumptions (consolidated)

Details for each are in the relevant section below; this is the checklist to
read before trusting a number.

- **Flat ground plane** by default (`FlatPlane`); `GriddedSurface` for a real
  DTM exists but is unused by the bundled studies. Verified fine for the
  bundled landing-site data (elevation flat to 1.1 m over 100 m radius) — not
  necessarily fine elsewhere.
- **ε (matching precision) is a swept/assumed constant**, not fit to real
  data by default. `mppp_error.colmap` can calibrate it from a real reconstruction;
  the bundled studies use ε_intra = 0.5 px as a working value.
- **θ_max (matchability cutoff) and θ_c (correlation decorrelation, θ_ρ) are
  literature-informed guesses**, not measured for Mars surface imagery.
  θ_ρ = 2° is a placeholder bounded below by the intra-station geometry the
  classical range equation already treats as independent; the `cluster`
  N_eff model using it is on by default, the angular gate is off.
- **Pose covariance is two-stage** (pose treated as fixed-with-known-covariance,
  then propagated), not jointly estimated with points. Generally conservative.
- **`pose_covariance_from_network`'s output is a lower bound** — grid cells
  are treated as independent tie points; real tie points are correlated and
  terrain where matching fails contributes nothing.
- **The occlusion mask (`DEFAULT_ROVER_MASK`) has unverified provenance** —
  carried over from an early prototype, not confirmed against rover CAD.
- **Instrument constants are working values, not verified specs** — MCZ-34's
  IFOV is linearly interpolated between the two published zoom endpoints;
  baselines are sourced but should be re-checked against the calibration
  paper before publication-grade use.
- **The site-image overlay's geometric assumptions are unverified** beyond a
  single visual check — north-up, square, exactly the stated real-world
  width, centred on the grid origin. See "Site imagery overlay" below.

## v0p2: figure customization, real-image cropping, rover glyph

- **300 DPI default** (was 125–130); figures are noticeably narrower
  (18.0 in vs. 21 in) since removing per-panel colour bars freed width.
- **Colour bars only on the last column of each row** ("inner" bars removed)
  — each row already shares one scale, so repeating the bar four times was
  redundant. `plot_map(..., show_colorbar=False)`.
- **Suptitle changed** to `"Waypoints of Perseverance with estimated
  reconstructed error: {metric label}"` plus a second line giving the primary
  (anchor) location's Sol / Site / Drive, read directly from the waypoint
  properties (`stations_from_rmcs` now returns `anchor_sol/site/drive` in its
  info dict, rather than being parsed back out of a display string).
- **Multi-metric figures.** `--metrics sigma_n,kappa,sigma_slope,G_n` (the
  default set) renders one figure per metric via a shared `ERROR_METRICS`
  registry and a single parameterized renderer, instead of a hard-coded
  σ_n-only figure. Filenames: `Mars2020_recon_error_Sol_{sol}_{metric}.png`.
  Unknown metric keys are skipped with a warning, not fatal.
- **PESSIMISTIC's angle parameter changed 12° → 15°.** The request said
  "change the pessimistic theta_c to 15" — **flagging this explicitly**:
  nothing named `theta_c` is wired into the four-case model (that name is
  reserved elsewhere for the *separate*, off-by-default error-correlation
  feature). The only angular knob PESSIMISTIC actually has is `theta_max_deg`
  (the matchability cutoff), so that's what was changed. If a literal θ_c
  (correlation decorrelation) was intended instead, say so and it's a
  one-line fix in `CASE_PARAMS`.
- **ε_intra now printed on every top-row caption**, not just
  PESSIMISTIC/OPTIMISTIC's. Confirming what was asked: yes, it's 0.5 px, the
  same value, for all four cases — `Instrument.eps_intra_px` is per-station
  and every station in these studies shares one `Instrument`.
- **Site image is now pre-cropped 25% off each side** before overlay
  (`trim_border_fraction`, `--site-image-trim-frac`, default 0.25) — the raw
  3333×3333 px file's true ground footprint is apparently ~100 m although its
  filename says 50 m, so trimming the outer 25% per side isolates the centre
  50 m × 50 m to match the grid axes. Verified this roughly doubles the
  apparent size of ground features in the crop (see the module docstring for
  the numeric check) — **this is a manual correction, not a measurement**; if
  25% isn't exactly right, `--site-image-trim-frac` is adjustable.
- **Sol numbers now labelled on every bottom-row panel** (previously only the
  top-left panel showed station labels at all) — `plot_map(..., labels=True)`
  on the improvement panels, and the image panel now calls the same
  `_station_markers` helper instead of a separate hand-rolled marker loop.
- **`--rover-icon` / `--rover-scale`**: an abstract top-down rover glyph
  (rectangular chassis, six wheel rectangles, a circle at the mast/camera
  axis) as an alternative to the plain rotated-square marker. Geometry is
  **arbitrary, not CAD-derived** — see `_rover_glyph`'s docstring. Default
  (no flag) keeps the original square markers.

## Versioning

Releases are tagged `v0pN` (v0.1, v0.2, ...), N incrementing by one every
delivery. **This is v0p15.** v0p7 → v0p8: package RENAMED `mppp` → `mppp_error` (it is the error-modelling component only, to be merged into the parent MPPP after that codebase is cleaned up); rover glyph rear shortened to the rear-wheel line with an optional rotation-centre marker and a 'rover convention' inset on the schematic. v0p6 → v0p7: rover glyph rotates about the middle-wheel centre with an asymmetric (shorter rear) deck; integration findings from the full MPPP bundle. v0p5 → v0p6: model simplification, real calibration constants, real occlusion profiles, per-case pose (see below). v0p4 → v0p5: error correlation is now ON by default with the simple `cluster` N_eff model — stations whose view directions at a cell fall within θ_ρ = `theta_c_deg` (2°, a placeholder) of each other are one look, at the precision of the best of them (greedy leader clustering). Redundancy is deliberately kept separate from ε_cross(θ): folding it into a band-pass ε is not equivalent (`test_cluster_neff_is_not_a_bandpass_eps`). The kernel models remain available via `correlation_kernel`. v0p3 → v0p4: mast circle enlarged. v0p3 was a single change over v0p2: the rover glyph was re-proportioned from a top-down Perseverance image (narrow chassis, six wheels floating outboard, mast circle at the front-starboard corner of the deck, where range is measured from). (v0p1 was the first delivery under this scheme,
carrying the real-data bundling and the LBS/IDEAL split. Two earlier
deliveries before v0p1 used an ad hoc `v2` tag, before this scheme was
requested.)

## v0p1: real data, LBS/IDEAL split, figure fixes

**Real M2020 data, bundled.** `mppp_error/data/` now ships an actual waypoint
GeoJSON (`M20_waypoints.json`, 698 features) and the real Mastcam-Z Butler
Landing vertical mosaic (`butler_landing_1_vertical_50m.jpg`). By default
`study_navcam.py` now builds its main figure from four REAL waypoints near the
landing site rather than a hand-placed pattern — see "Real waypoint data"
below. `--synthetic` restores the old hand-placed geometry; the spacing sweep
always uses it (a sweep needs a controllable spacing, which real waypoints
don't offer).

**IDEAL renamed to LBS; a new IDEAL takes its place.** The old single-best-pair
case is now called `lbs` ("long baseline stereo"). A new `ideal` case answers
"the best achievable using every possible pairwise long-baseline constraint,
properly combined" — see "LBS vs. the new IDEAL" below, including a bug this
change surfaced and fixed.

**Figure fixes:** stats boxes (mean/rms/median) on every quantitative panel;
axis labels only on the outer grid edge; shared colour scale within each row;
spacing sweep extended to 100 m and its two panels stacked vertically instead
of side-by-side, which overturned an earlier claim (see "The bracket is not
monotonic" below).

---

## Quickstart

```python
from mppp_error import (Grid, Station, MASTCAM_Z_34, ModelConfig, PoseModel,
                  solve_precision_field, compute_metrics, summarize)
from mppp_error.plotting import plot_panel

stations = [
    Station(xyz=[0, 0, 1.9],   az_deg=45,  name="S1", is_anchor=True),
    Station(xyz=[-14, 6, 2.1], az_deg=110, name="S2", path_m=16.0),
    Station(xyz=[9, -12, 1.7], az_deg=330, name="S3", path_m=34.0),
]
grid  = Grid.square(half_width_m=25, n=201)          # z=0 is the ground
field = solve_precision_field(grid, stations, ModelConfig(), PoseModel("none"))
m     = compute_metrics(field, coverage_threshold_m=0.05)
print(summarize(field, m))
plot_panel(field, m)
```

Command line:

```
python -m mppp_error selftest
python -m mppp_error demo  --eps 0.5 --radius 25
python -m mppp_error sweep --param eps --values 0.25,0.5,1,2
python -m mppp_error run   --waypoints traverse.geojson --sol 1500 --pose telemetry

# four-regime study on REAL bundled M2020 data (default), with the real
# Butler Landing site image, spacing sweep out to 100 m:
python -m mppp_error.study_navcam --out DIR

# hand-placed synthetic geometry instead (needed for --spacing sweeps):
python -m mppp_error.study_navcam --out DIR --synthetic --spacing 4
```

Existing notebooks: `from mppp_error.compat import fused_surface_error_field, make_report`.

---

## What the model does

Each image measurement is an **angular** observation and contributes a rank-2
precision:

    Λ_ray = (1/σ_⊥²)(I − u uᵀ),    σ_⊥ = ε · ψ · r

A stereo station is **not special-cased** — it is two rays separated by the
baseline. The cigar-shaped ellipsoid and the r²/b range scaling fall out of the
sum rather than being asserted. This matters because it handles, with no extra
machinery: near-field geometry where b/r is not small; mixed zoom per station;
sloped ground; and mono visibility, where one eye sees a point and the other is
occluded (a rank-2 constraint still usable for cross-station triangulation).

**Frame convention:** z = 0 is the ground; cameras sit above it. The solver
raises a specific error if no cell is visible from any station, which is almost
always this frame convention being violated.

---

## Design decisions encoded here

**No r/h obliquity factor.** It double-counts. Building the 3D covariance
correctly reproduces 1/cos(e) automatically, and gives the right answer at both
limits: Σ_zz → σ_ρ² at nadir (standard aerial stereo), Σ_zz → σ_t² at grazing
(a horizontal-looking camera measures *height* perpendicular to its line of
sight, where it is most precise). Applying r/h on top inflates the vertical
error precisely where it is best. `apply_cos_scaling` is accepted and ignored.

**The √2.** Because ε is defined per-image-per-axis (the standard σ_x'), two
rays give σ_transverse = εψr/√2 and σ_range = √2 εψr(r/b). The textbook
σ_Z = Z²σ_disp/(cb) hides the √2 inside σ_disp, the precision of the *measured
disparity*. If the two image measurements are independent, σ_disp = √2 σ_x' and
the forms agree exactly — verified numerically in the self-tests. v1 used
σ_t = εψr and σ_ρ = εψr²/b, so its **anisotropy ratio was off by a factor of 2**.

**ε factors out.** ε_intra is the only dimensional scale on Σ, so the reported
`G_n = σ_n/(ε·ψ·r_ref)` is ε-independent by construction and comparable across
zoom, range, instrument, and site. ε_cross does not enter the covariance — it
belongs in the gate weights as a precision ratio (ε_intra/ε_cross)². Self-test
`test_eps_scaling` verifies σ_n scales linearly with ε while G_n does not move.

**ε, ρ and θ_c are independent parameters.** ε is the magnitude of the random
error on one measurement; ρ is the correlation between the errors of two
different observations; θ_c is the angular scale over which ρ decays. ρ is not
derivable from the other two — it comes from sub-pixel interpolation bias
(pixel-locking), foreshortening bias, interior-orientation residual, and surface
definition ambiguity. Different moments of the same physics, measured separately.

**Angular gating and correlation default OFF.** θ_c (decorrelation) and θ_max
(matching failure) are within roughly a factor of two under the current scaling
argument, and neither has been measured for Mars surface imagery. Both are
implemented and switchable; defaults give a pure-geometry map with no
unvalidated parameters. The exponential kernel
ρ_ij = ρ_∞ + (1−ρ_∞)exp(−θ²/2θ_c²) is a modelling assumption imported from
geodetic covariance practice, not received wisdom from the stereo literature —
state it that way in the paper.

**Interior orientation is not in Σ_pose.** It is common to every station, so
modelling it per-station would let averaging remove it — exactly wrong. It
belongs in ρ_∞. Σ_pose carries only relative-to-anchor position and attitude;
the anchor has zero pose covariance by the free-network convention, because the
common-mode part is a pure datum shift that does not degrade internal quality.

---

## v1 → v2: what was fixed

**Physics**

| | v1 | v2 |
|---|---|---|
| Anisotropy ratio | off by 2× | derived from two rays, verified analytically |
| Obliquity | `apply_cos_scaling` divided by cos, not cos² — and double-counted anyway | removed |
| Baseline / IFOV | `baseline=0.42` (Navcam) with an IFOV matching neither MCZ endpoint | named instruments, values flagged VERIFY |
| Station elevations | `stations_xyz[:,2] = 1.9` discarded them | preserved |
| Pose error | absent | Σ_t + [r]×Σ_ω[r]×ᵀ, telemetry / BA modes |
| Units | ×100 inside the covariance, docstring said m² | SI throughout |

**Bugs**

- `np.isnan(x) is False` — identity comparison against a Python singleton, always
  `False` for `np.bool_`. Dead clause; "fixing" it to `== False` would have
  silently discarded every station and returned an all-NaN map.
- `geom_mean`/`rms_mean` computed *inside* the double loop from the full arrays:
  recomputed 62,500× and `NameError` if the first cells hit `continue`.
- `Prec_sum` (2×2) accumulated, never used.
- `np.atan2`/`np.asin` — NumPy ≥ 2.0 only.
- `imshow(extent=...)` used cell *centres*, not edges — half-cell offset on a
  metric product.
- `plot_all_basic_heatmaps` called a nonexistent function; `log_scale` printed
  "not implemented".
- `site_drive_for_sol`'s third return unpacked as `anchor_method`; it is
  `matched_sol`.

**Performance.** Analytic rank-2 precision, no inverse in the accumulation, one
batched 3×3 inversion per cell at the end, vectorised over the grid with
`einsum`. The v1 nested Python loop over 62,500 cells × N stations with two
`np.linalg.inv` calls each is gone. Also avoids inverting a covariance with
condition number ~(r/b)² ≈ 3.5×10⁶ at 25 m.

---

## Metrics

| Metric | Read it as |
|---|---|
| `sigma_n` | **Primary.** DEM vertical precision; comparable to HiRISE EP |
| `G_n` | σ_n in units of the pixel footprint. ε-independent benchmark currency |
| `sigma_t_min/max` | Lateral registration accuracy — not DEM quality |
| `kappa` | Anisotropy. κ≫1 → use `worst_direction()` to find where the next station goes |
| `n_eff` | Effective independent stations. n_eff ≪ n_vis → redundant geometry |
| `improvement` | best-single/fused. Pure network value, independent of ε |
| `sigma_slope` | Differential error — predicts *perceived* quality (VR/outreach) |
| `coverage_mask` | Area fraction meeting threshold. Headline number for ranking acquisitions |
| `gsd` | Resolution floor, independent of network geometry |

---

## A first result

Six stations along a synthetic drive, MCZ-34, ε = 0.5 px, 25 m grid:

| pose mode | median σ_n | improvement | median κ |
|---|---|---|---|
| none (pure geometry) | 0.083 cm | **23.2×** | 3.1 |
| bundle-adjusted | 0.237 cm | 6.9× | 10.4 |
| telemetry only | 1.53 cm | **1.16×** | 101 |

With telemetry-only pose the improvement collapses to ~1: six stations buy
essentially nothing over one, because each station's contribution is floored by
its own pose error. **That gap is the quantified value of bundle adjustment,
expressed in DEM precision** — and it says the pose term is not an optional
refinement but the dominant effect for multi-station rover networks. Worth
reproducing on the four real sites before trusting the magnitude.

---

## The four regimes

`python -m mppp_error.study_navcam` — Navcam mosaics, ε_intra = 0.5 px, BA pose
**derived from the network** (not assumed).

**Now built from four REAL M2020 waypoints by default** (`mppp_error/data/`), not a
hand-placed pattern. Anchor RMC `3_0` is the site-3 frame origin itself (sol
13, drive 0, "Site increment, no motion" — the localized landing-site frame,
built from imagery spanning sols 3–11, matching the bundled Butler Landing
panorama's date range), plus three nearby early drives: `3_110` (6.0 m),
`3_1266` (18.0 m straight-line — but **166.9 m of actual driven path**, per
real odometry, `dist_total_m`), `3_1398` (20.1 m straight-line / 191.0 m
driven). Elevation is flat to within 1.1 m over the whole 100 m radius this
dataset covers, so the flat-plane assumption holds cleanly here.

**The real geometry is honestly worse than the synthetic test pattern.** Every
waypoint within 40 m of the anchor falls in a 0°–146° azimuthal arc — nothing
on the far side. These were early checkout drives near the lander, not chosen
for photogrammetric coverage. This is itself an illustration of the paper's
premise: real acquisitions are usually far from what a network-design tool
would recommend, which is exactly the gap this tool is for.

Map fixed to **±25 m**, anchored on the landing site:

| case | ε_cross | θ_max | σ_n | G_n | × vs FIXED | κ |
|---|---|---|---|---|---|---|
| **FIXED** — fixed-baseline stereo only | — | — | 1.57 cm | 6.38 | 1.0× | 70 |
| **PESSIMISTIC** SfM+MVS | 2.0 px | 12° | 1.23 cm | 6.02 | 1.0× | 55 |
| **OPTIMISTIC** SfM+MVS | 0.6 px | 45° | 0.18 cm | 0.69 | 9.3× | 4.4 |
| **LBS** — best single pair | — | — | 0.26 cm | 1.03 | 6.2× | 3.3 |
| **IDEAL** — best of ALL pairs, combined | — | — | 0.18 cm | 0.72 | 8.9× | 3.0 |
| full fusion | — | — | 0.13 cm | 0.51 | 12.6× | 3.1 |

**PESSIMISTIC barely beats FIXED here (1.0×), which is real, not a bug.** The
real station-pair convergence angles are 14.6°–56.1° (printed by the view-graph
report). PESSIMISTIC's θ_max is 12° — *below every single pair's actual
convergence angle* — so nearly every cross-station link gets crushed by the
angular gate, and 46% of cells fall back to FIXED entirely. A poorly-spread
real network and a narrow-tolerance matcher compound into "cross-station
matching contributes almost nothing," which is a legitimate, checkable
operational finding, not an artifact.

### LBS vs. the new IDEAL

**Renamed:** the old single-best-pair case is now `LBS` ("long baseline
stereo"). It answers "what would one well-chosen long-baseline pair deliver" —
comparable to how stereo DEMs are traditionally produced from a single pair —
not "what is the best possible result."

**New `IDEAL`:** the best achievable by combining *every* possible pairwise
long-baseline constraint, not just the widest one. Implementation: clone every
station with its two eyes collapsed to a single centre ray, at perfect
(ε_intra) matching, no gating, and hand the clones to the same exact-geometry
solver used everywhere else (`link_mode='full'`). Since Fisher information adds
linearly across independent ray observations, summing every visible station's
own ray precision *once* automatically incorporates the joint constraint from
every possible pair simultaneously — there is no better combination achievable
under this ray model, and no risk of the double-counting a naive "sum over
every pair" would produce (which would count each station's own contribution
once per pair it participates in, wildly overstating precision).

Provable ordering, checked as a hard invariant on both synthetic and the real
landing-site geometry: **`full_fusion ≤ IDEAL ≤ LBS`**, always. `LBS` is left
off the four-panel figure for now (still computed, still printed in the table)
per the request that motivated this change.

> **An open question I want confirmed.** The request specified IDEAL as "the
> maximum of inter range error and the combined error of all possible LBS
> constraints of any pair of stations." I could not resolve "the maximum of
> inter range error and ___" to a construction consistent with "not just one
> LBS, but the best of all possible" — a literal max(single-pair, combined)
> would always just return the single-pair value, since combining more
> information provably never *increases* error. I implemented the "combined,
> properly fused" half of that sentence as IDEAL, and left the "maximum of
> inter range error" half unaddressed. If that phrase meant something
> specific — e.g. a per-axis maximum of range vs. transverse error, or a
> comparison against a different reference quantity entirely — tell me and
> I'll fix it in the next round.

**Bug this rename surfaced and fixed.** Building `IDEAL` from single-ray
station clones exposed a latent inconsistency in the old `LBS` (then
`ideal_long_baseline_field`): it picked its widest-baseline pair using
*two-eye* visibility (from the real stereo geometry) but then computed
precision from the *station-centre* ray. At an occlusion-mask boundary one eye
can see past the mask while the exact centre ray cannot, so a station could be
credited as visible by the eligibility test and then contribute geometry that
same test would have rejected — a real (if rare, ~0.1% of synthetic-geometry
cells) violation of "more information cannot increase variance." Fixed by
building both `LBS` and `IDEAL` from the same `_center_ray_clones` helper, so
eligibility and geometry are consistent by construction rather than by
coincidence. Re-verified on the real landing-site geometry: **zero violations
across all 14,641 grid cells** (`test_ideal_ordering_holds_on_real_data`).

### The spacing sweep changes the experiment design

Extended to 100 m (from an earlier 40 m) and the two panels are now stacked
vertically:

| spacing | span | FIXED | PESS | OPT | pess × | opt × | **bracket** |
|---|---|---|---|---|---|---|---|
| 0.4 m (≈ baseline) | 0.6 m | 4.24 | 2.79 | 1.60 | 1.5× | 2.7× | 1.8× |
| 1.5 m | 2.1 m | 4.23 | 1.31 | 0.69 | 3.2× | 6.2× | 1.9× |
| 5 m | 7.0 m | 4.24 | 0.65 | 0.33 | 6.6× | 12.8× | 2.0× |
| **10 m** | 14 m | 4.23 | 0.68 | 0.30 | 6.2× | **14.4×** | 2.3× |
| 40 m | 56 m | 4.35 | 2.32 | 0.41 | 1.9× | 10.5× | 5.6× |
| **70 m** | 98 m | 4.69 | 3.82 | 0.64 | 1.2× | 7.3× | **6.0×** |
| 100 m | 140 m | 5.25 | 4.89 | 0.89 | 1.1× | 5.9× | 5.5× |

Three findings, one of them a correction:

1. **Reconstruction gain peaks near 10 m spacing and then declines.** Wider
   stations triangulate better but overlap less and match worse.
2. **The bracket peaks around ~70 m, not at the far end of the sweep.** An
   earlier version of this README (and the script's own docstring, at the
   time) claimed the bracket "grows monotonically with spacing" — true only
   over the 1.5–40 m range then tested. Both PESSIMISTIC and OPTIMISTIC
   degrade toward FIXED as spacing keeps growing (matching gets harder for
   everyone at extreme baselines), so their *ratio* eventually stops widening
   too. This only showed up once the sweep was extended to 100 m, which is
   why the extension mattered beyond just "a longer x-axis."
3. At 0.4–5 m spacing the answer barely depends on ε_cross or θ_max. Good news
   for reconstruction, bad news for calibration: **to measure the parameters
   you want separated stations, but not unboundedly so — aim near the ~70 m
   bracket peak for this geometry, not the sweep's far end.**

### The spacing sweep now includes the instrument's own baseline

Default sweep spacings start at the instrument's stereo baseline itself (0.42 m
for Navcam) rather than an arbitrary 1.5 m minimum, and both panels mark it with
a vertical line. At that spacing a second *station* sitting b_inst away adds
almost nothing over the camera's own two eyes — improvement is only 1.5–2.7× —
which is the right sanity check at the left edge of the curve.

### ε_cross = 0 is not a distinct case

w_ij = (ε_intra/ε_cross)² is clipped at 1, so ε_cross below ε_intra changes
nothing — correctly, because ε is a measurement precision in image space and a
cross-station match cannot be more precise than the images themselves. All the
geometry is already carried by the ray directions. **ε_cross = ε_intra *is* the
perfect-matching case.**

## Real waypoint data (`mppp_error.waypoints.stations_from_rmcs`)

For hand-picked real stations (rather than everything-within-a-radius, which
for a real drive path often pulls in near-duplicate revisits alongside the
ones you actually want — `build_stations`'s radius selection can't tell them
apart):

```python
from mppp_error.waypoints import stations_from_rmcs
sts, info = stations_from_rmcs("mppp_error/data/M20_waypoints.json",
                               ["3_0", "3_110", "3_1266", "3_1398"],
                               instrument=my_instrument)
```

`path_m` is set from real rover odometry (`dist_total_m`, cumulative driven
distance from mission start) relative to the anchor, **not** straight-line
reconstruction — the difference matters: station `3_1266` is 18.0 m
straight-line from the anchor but 166.9 m of actual winding drive, so its
telemetry-based pose uncertainty (computed by `PoseModel('telemetry')` from
`path_m`) is far worse than straight-line distance alone would suggest.

## Result: how much does cross-station correspondence buy?

*This section documents `study_cross_station.py`, an earlier standalone
exploration with its own internal `run_cases`/`CASES` — it predates and does
not share code with `mppp_error.cases` (`run_four_cases`, `FIXED`/`LBS`/`IDEAL`), so
it still uses `OPS` for the fixed-baseline case and its own `ideal` for
perfect cross-station matching (not the "combine every pair" IDEAL defined
above). Kept as-is since it still runs correctly; `study_navcam.py` is the
more thorough and currently-maintained treatment.*


`python -m mppp_error.study_cross_station` — Navcam, 5 stations over a ~29 m footprint,
ε_intra = 0.5 px, BA pose.

| case | ε_cross | θ_max | median σ_n | G_n | × vs OPS | κ | cover <2 cm |
|---|---|---|---|---|---|---|---|
| OPS (fixed-baseline stereo) | — | — | 2.70 cm | 8.6 | 1.0× | 12.8 | 0.09 |
| PESSIMISTIC (dense NCC/SGM) | 2.0 px | 12° | 1.77 cm | 6.8 | **1.4×** | 66.6 | 0.60 |
| OPTIMISTIC (learned dense) | 0.6 px | 45° | 0.42 cm | 1.3 | **6.8×** | 8.2 | 1.00 |
| ideal (perfect matching) | — | — | 0.39 cm | 1.3 | 7.3× | 6.9 | 1.00 |

**The bracket is 1.4× to 6.8×.** That factor of 5 is the entire uncertainty, and
it is controlled by two numbers nobody has measured for Mars surface imagery.
`cross_station_sensitivity.png` shows the whole result as a surface over
(ε_cross, θ_max), with both cases marked.

Two things worth noting from the maps:

* **Gain is lowest near stations and highest in the far field.** Close in, a
  single station's fixed baseline already works; at range, r²/b dominates and
  only cross-station geometry helps. That is exactly where the science targets
  are.
* **The predicted view graph separates the cases by three orders of magnitude**
  (λ₂ = 0.63 optimistic vs 0.00084 pessimistic). The pessimistic network has
  edge redundancy 0.6 and articulation stations S2, S3 — a chain that splits if
  matching fails at either. λ₂ is a far sharper discriminator than σ_n, so it
  is the cheapest thing to check against real data first.

## Calibrating from real data (`mppp_error.colmap`)

Everything the model needs is already in a COLMAP sparse reconstruction.
Validated against a synthetic model with known ground truth:

| quantity | truth | recovered |
|---|---|---|
| ε_intra | 0.400 px | 0.401 px |
| ε_cross | 1.100 px | 1.101 px |
| ε ratio | 2.75 | 2.747 |
| θ_max (half-survival) | 27.4° | 23.4° |
| stations (by clustering) | 5 | 5 |

Key functions:

* `calibrate_eps_from_residuals` — ε_intra vs ε_cross from **true per-observation**
  reprojection residuals. COLMAP's `point.error` is a track-level scalar, so
  anything binned against a per-observation quantity comes out flat; the residual
  is recomputed from poses, camera model and keypoints.
* `match_survival` — the measured θ_max. Counts matched pairs against
  **geometrically possible** pairs per point. Without that denominator you are
  only measuring how much scene sits at small convergence angle, which says
  nothing about the matcher.
* `measure_theta_c` — the decorrelation angle, from pairwise triangulation
  residual correlation vs. angular separation.
* `measured_view_graph` — the observed graph, to compare against the predicted one.

A **coarse sparse model is sufficient** for all of the above. Dense MVS is not
needed. What sparse tracks cannot give you: absolute accuracy against ground
truth, surface completeness, or anything between tie points. Sparse tracks are
also biased towards well-textured, well-matched terrain, so ε measured this way
is optimistic — label it as a lower bound.

## Measuring view-graph quality

There is a hierarchy, cheap to expensive, implemented in `viewgraph.py` and
`network.py`:

1. **Connectivity** — components. If split, relative pose is unrecoverable at
   any precision. Binary, cheapest, check first.
2. **Algebraic connectivity λ₂** (Fiedler value of L = D − W). Bounded by vertex
   and edge connectivity. Scale-free when normalised by λ_max. A proxy.
3. **Topological redundancy** — articulation stations, bridges, edge-removal
   tolerance. For a 3–7 station rover network this is usually the binding
   constraint: not "is the average good" but "is there one link whose failure
   splits the network".
4. **Parallel rigidity** — a connected graph is not necessarily solvable.
   Recovering translations from pairwise *directions* needs parallel rigidity
   (Özyeşil & Singer), strictly stronger than connectedness.
5. **Pose-block conditioning — the actual answer.** Build the BA normal matrix,
   Schur-eliminate the points, project out the datum modes, invert. The result
   *is* the pose covariance, which *is* what "view graph quality" means
   operationally.

`pose_covariance_from_network` implements (5), which closes the circularity
flagged earlier: **Σ_pose in BA mode is now derived, not assumed.**

Two implementation points. The datum modes (3 translation, 3 rotation, +1 scale
if free) are constructed **analytically** and projected out, rather than found by
an eigenvalue threshold — the signal spectrum spans 5 decades, so a threshold is
fragile. And scale is *not* free for a rover: each station's stereo baseline has
known length, so the defect is 6, not 7.

`relative_to(station)` re-references with the cross-covariance term,
Σ_rel[i] = Σ[i,i] + Σ[a,a] − Σ[i,a] − Σ[a,i]. Dropping the cross term would
double-count the shared part.

**The returned pose covariance is a lower bound**: grid cells are treated as
independent tie points, and terrain where matching fails contributes nothing.
Use `tie_correlation` for a defensible upper estimate and report both.

## Pose model tiers, from the literature

| mode | relative position error | source |
|---|---|---|
| `deadreckon` | ~10% of distance | MER design goal "at most 10%"; JPL Mars Yard wheel odometry "not better than 10%" |
| `telemetry` | ~2–3% of distance | MER VO ~3% of range walked; JPL 25 m runs below 2.5%; M2020 AutoNav ~2% per 100 m |
| `ba` | ~0.2% of distance | ground-in-the-loop bundle adjustment |
| `network` | derived | `pose_covariance_from_network` |

Independent stereo-VO analysis of Perseverance found 10–30 cm differences from
the telemetered trajectory on longer drives.

## The √2, and the precision convention

ε ≡ σ_x′: one image-coordinate measurement, one image, one axis. The textbook
σ_Z = Z²σ_d/(c·b) uses the **disparity** precision, and since d = x_L − x_R with
independent measurements, σ_d = √2·σ_x′. Substituting gives
σ_Z = √2·ε·ψ·r·(r/b) — so the √2 **is** in the standard equation, hidden inside
σ_d. The transverse √2 is different: the standard equation gives no transverse
error at all, and the usual σ_t = ε·ψ·r is the *single-image* answer.

`Instrument.precision_convention` is `'per_image'` (ε = σ_x′) or `'disparity'`
(ε = σ_d, divided by √2 internally). Verified equivalent in the self-tests.

## ε, ρ and θ_c

    ε² = ε²_ind + ε²_com,        ρ_∞ ≡ ε²_com / ε²
    C_ij = ε_i ε_j ρ(θ_ij),      C_ii = ε_i²
    Λ = Jᵀ C⁻¹ J
    N_eff = N / (1 + (N−1)ρ)     →  σ_fused = ε/√N_eff
    N → ∞:  N_eff → 1/ρ_∞,  σ_fused → ε√ρ_∞          (the floor)

**ρ is not a function of ε and θ_c.** Model the image-space error of observing a
point from direction **u** as a random field e(**u**): ε_cross is its *variance*
and ρ its *correlation* — the diagonal and off-diagonal of one covariance
function, Cov[e(u_i), e(u_j)] = ε_i ε_j ρ(θ_ij). Different moments of the same
process, and independent inputs unless you posit a generative model for e(**u**).

Four swappable kernels, because the shape is a modelling assumption and not
settled:

| kernel | form | note |
|---|---|---|
| `cosine` (default) | max(cos(πθ/2θ_c), 0)² | **compact support** — exactly zero beyond θ_c |
| `gaussian` | exp(−θ²/2θ_c²) | conventional, but never reaches zero |
| `exponential` | exp(−θ/θ_c) | heavier tail still |
| `tilt` | on transition tilt τ | the quantity governing affine matchability |

`cosine` is the default for a concrete reason: a Gaussian never reaches zero, so
with many stations tiny spurious correlations between widely separated views
accumulate and artificially suppress N_eff. Report under at least two kernels.



## v0p6: the model reduced to five measurable numbers

The parameter set was deliberately cut so that every remaining free number is
something the proposed experiment can measure. What changed and why:

| change | before | now | reason |
|---|---|---|---|
| ε_cross,0 | free (0.6 / 2.0 px) | **tied to ε_intra** (`eps_cross_px=None`) | a cross-station pair at zero convergence should match as well as a station's own pair; removes a parameter, and the assumption is checkable from the 1 m MDI pairs |
| gate shape *m* | 4 | **2, fixed** | *m* is a shape; no other parameter absorbs it. m=2 and m=4 put the information optimum at the same place (2^−1/2 = 4^−1/4 = 0.71·θ_max); m=1 would move it to 1.0·θ_max and destroy the plateau |
| τ (transition tilt) gate | on | **off** | redundant with θ on terrain flat over ~5 m from one mast height, and uninterpretable |
| ρ_∞ | 0 | 0 (unchanged) | assumed zero for now; measurable from the N+2 rotate-in-place pairs |
| θ_c naming | `theta_c_deg` | **`theta_min_deg`** (alias kept) | pairs with θ_max as the two edges of the useful baseline window |
| pose | one model for all cases | **per case** (`CASE_POSE`) | see below — this was a real error |

So the free parameters are now: **ε_intra, θ_min, θ_max**, plus two pose
fractions. Five numbers, each with a measurement.

### The useful baseline window

    theta_min * r  <  b  <  theta_max * r

Below θ_min·r a second station is the *same look* (redundant); above θ_max·r it
is *unmatchable*. Both edges are measured by the experiment. This is why the
optimum spacing is a **base-to-range ratio** (b_opt ≈ 0.3–0.4·r, verified
nearly independent of station count, with improvement scaling as √N) rather
than a fixed number of metres — and therefore transferable between sites.

### Per-case pose was wrong before v0p6

Every case previously shared one pose model. They are registered differently
and must not:

| case | pose mode | scales with |
|---|---|---|
| FIXED | `telemetry` | ~3 % of **path driven** — products are placed in a common frame by rover localisation, and drift follows the path |
| PESSIMISTIC / OPTIMISTIC / full fusion | `sfm` | ~0.2 % of **straight-line baseline to the anchor** — relative pose comes from bundle adjustment on the cross-station ties |
| LBS / IDEAL | `none` | zero — geometry-only ceilings |

The distinction is not cosmetic: landing-site waypoint `3_1266` is 18 m from
the anchor but **167 m of driving**, so telemetry pose error there is ~5 m
while SfM pose error is ~3.6 cm. That asymmetry *is* the argument for
cross-station ties, and it only appears once the two scalings are separated.
`test_pose_modes_path_vs_baseline` pins it. Attitude follows from position by
one rule, σ_att = σ_pos / `tie_range_m`, rather than a separate parameter.

**Consequence for the invariants:** `full_fusion ≤ IDEAL ≤ LBS` now holds only
**at equal pose**, because IDEAL/LBS are zero-pose ceilings while full fusion
carries SfM pose error. The ordering tests pass an explicit `PoseModel()` for
this reason; it is a statement about ray information, not about delivered
products.

## Real flight constants (v0p15)

Instrument constants now come from the flight CAHVOR frame models in
`params/m20_cmods/`, replacing values that had carried `VERIFY` flags since
v0p1:

| camera | f [px] | IFOV [rad/px] | was |
|---|---|---|---|
| Mastcam-Z 34 mm | 4720 | 2.119e-4 | 2.17e-4 (interpolated between 26 and 110 mm) |
| Mastcam-Z 110 mm | 14852 / 14830 | 6.738e-5 | not modelled |
| Navcam | 2950.9 / 2943.6 over 5120 px | 3.393e-4 | 3.3e-4 |

`MASTCAM_Z_110` is new, for the far-target long-baseline pairs.

## Real occlusion profiles (v0p15)

`data/M2020_occlusion_profiles.csv` holds hand-measured per-eye rover
occlusion profiles (minimum visible elevation vs. rover-frame azimuth) for
Zcam and Ncam, left and right. Measured by CT from rover-frame az-el projected
mosaics centred on the mast rotation axis, reading where the rover meets the
terrain.

    from mppp_error.core import load_occlusion_profiles, eye_masks_for
    prof = load_occlusion_profiles()          # zcam_left, zcam_right, ncam_left, ncam_right
    inst.eye_masks = eye_masks_for("zcam")    # [left, right]

**As delivered the Zcam and Ncam columns are identical** — one profile
duplicated across cameras, which is expected for now. The file format already
supports per-camera profiles, so replacing the Ncam columns with their own
measurements requires no code change. Left and right eyes DO differ and are
used per eye. This supersedes `DEFAULT_ROVER_MASK`, whose provenance was never
verified; that constant remains for backwards compatibility but new work
should use the CSV.

## G_n with mixed instruments

`ifov_ref` previously returned NaN when a network mixed Z34 and Navcam, which
silently blanked `G_n` for exactly the mixed networks this experiment proposes.
It now uses the **finest** IFOV present, so G_n reads as "how many
best-available-pixel footprints is the error"; `PrecisionField.mixed_instruments`
flags the case.

## v0p15: "yield" is now "completeness"

The quantity formerly called yield -- the expected number of matched
cross-station pairs per cell, and the fraction of the scene that receives a
valid cross-station tie -- is now `completeness` throughout (field
`PrecisionField.completeness`, `p_cross` unchanged).  This is the term the
robotic-stereo benchmarks (Middlebury, KITTI) use for the filled fraction of a
disparity map, and adopting it lets the three cases be scored on the same axes
as any stereo entry.  Precision (sigma_z) and completeness are the two
observables; the model's failure mode is holes, not blur.

## v0p14: integrating the 2026-09-15 robustness results (three corrections)

1. **Two illumination scales, not one.**  Matched-pair COUNTS decay with
   L0 = 2.3 h (cross-station stratum); INFORMATION per pair (S/eps^2) decays
   with L_eff = 1.46 h (single exponential, 0.061 dex; the earlier sub-hour
   plateau was an artefact of separating the channels).  `completeness` uses
   L0.  My analytic 1/L_eff = 1/L0 + 2/L_eps = 1.09 h was wrong by 26 %
   (wrong stratum for L0, and a linearisation invalid for dL/L_eps up to 1.5).
2. **Matches clump, so the yield threshold was ~10x too permissive.**
   P(cell has no cross-station match) = exp(-mu/k) with k = 9-40 at 1 m.  The
   hard threshold of 1 expected pair is replaced by a probability
   p_cross = 1 - exp(-mu/k), k = 20, which reproduces the hole pattern
   (AUC 0.75-0.89).  The display contour is now P = 0.5 (mu ~ 14).
3. **The cell model was still conflating intra and cross (v0p11 bug, second
   layer).**  link_mode='cross' summed EVERY station's rays at full weight,
   which is full fusion, not "own stereo".  Now
   E[Lambda] = Lambda_nearest + p_cross * sum_{j!=nearest} Lambda_j:
   FIXED is the p_cross = 0 limit, BEST the p_cross = 1 limit, and SfM sits
   between them by exactly the measured tie probability.  Belva medians:
   0.47 / 0.13 / 0.05 cm; Rockytop 0.43 / 0.22 / 0.05 (low yield -> SfM close
   to FIXED, as it should be).
4. **rho_inf = 0.22 was the wrong parameter.**  0.22 is the correlation at
   ZERO separation (repeat looks from one place), decaying to ~0 within
   theta_rho ~ 0.4 deg.  Setting it as the floor put 22 % correlation on every
   pair and made networks worse than one station.  Now rho_0 = 0.22,
   rho_inf = 0.  Test: 2/4/8 repeat looks give 1.28/1.55/1.77x (cap 2.1x);
   separated stations unaffected.
5. Texture proxies from surface SHAPE (roughness, relief, density) do not
   predict yield (r ~ 0, R^2 = 0.10).  Consistent with the gate being
   descriptor matching -- an APPEARANCE property.  Structure-tensor coherence
   from imagery is the proxy still to test.
6. Our 1 h window is shadow-tip distance ~0.3, half the orbital community's
   0.6.  Plan in hours, report in STD, quote <~0.3.  Do NOT use the delta-azimuth
   criterion at Jezero: near-zenith sun makes azimuth swing fast for no
   illumination change (it admits pairs 4.9 h apart).
7. Still provisional: S0 and theta_bar carry an unbounded systematic until the
   Belva zero-match tail is resolved by exhaustive matching; flat-eps is an
   inference until item 4 (raw-match residuals) runs.  A 12-point prediction
   for site 87 is frozen and hashed.

## v0p12: three cases, station rules, figure layout

* `cases.run_three_cases`: FIXED (link_mode='nearest' -- every cell takes the
  CLOSEST station that can deliver stereo, i.e. both eyes visible; a station
  seeing a cell with one eye is no baseline at all and produced NaN streaks
  before this rule), SfM (link_mode='cross', measured gate and redundancy),
  BEST (the ceiling: all stations equal, N_eff = N, dLMST = 0, epsilon and
  yield flat to 90 deg -- full fusion, no gate, no correlation).  Each carries
  a `completeness` field: 1 (own pair), expected matched cross pairs, and all
  possible cross pairs respectively.
* `waypoints.stations_from_rmcs(last_per_sol=True)`: one station per sol, the
  LAST position (highest drive count).  A sol's mid-drive localization is not
  an imaging station.  Belva Sol 784 had two entries (39_858, 39_926); only
  39_926 is kept.  Rockytop drops three mid-drive entries the same way.
* Rover glyph: three wheels per side equally spaced at -1/0/+1 m, a 2.0 m body
  centred on the wheels, mast a little inward of the forward-starboard corner.
* `plotting._sol_labels`: a Sol label on EVERY panel, pushed radially away from
  the cluster centroid with a leader line so it cannot sit on a glyph.
* `maps/make_three_cases.py`: 2x5 layout -- the site ortho (50 m span, cropped
  to the grid extent and offset by the ortho station's position relative to
  the anchor) occupies the left 2x2; sigma_z (top) and yield (bottom) for the
  three cases fill the right 3x2.  Low-yield masking is OFF by default
  (`MASK=False`); the yield threshold is drawn as a dashed contour instead
  (`OUTLINE=True`, `MIN_PAIRS=1.0`).  Consistency check: the black rover
  silhouette in each ortho lands under the glyph of the station that took it
  (39_926 at Belva, 26_1004 at Rockytop).
* NOTE the earlier Butler Landing ortho, also named `_vertical_50m`, was found
  to have a ~100 m true footprint.  These two are treated as 50 m on the
  user's statement; `ORTHO_SPAN` is the one knob to change if that is wrong.

## v0p11: precision and yield separated (BUG FIX)

link_mode='cross' replaces 'gated' for the SfM cases and fixes a real error.
'gated' multiplied a station's ENTIRE information by its tie weight w_i --
including its own left/right stereo, which needs no cross-station match. Under
the measured gate (w ~ 1e-4 over much of a 25 m map) that erased each station's
own stereo, the fusion lost to the best single station, and the fixed-fallback
silently restored FIXED on 99.8 % of cells. The three SfM maps were three
copies of the FIXED map.

The deeper error: the gate is a SURVIVAL PROBABILITY (a count of matches), and
it was being multiplied into Fisher information (the precision of one
measurement). For a single point a cross-station tie either exists or does not;
the gate governs what FRACTION of a cell's points get one. Now:

  * intra-station stereo: full weight, always;
  * cross-station coupling: full weight where present (eps is flat in theta --
    a match at 40 deg is as precise as one at 2 deg, there are just ~170x fewer);
  * the gate accumulates separately as `PrecisionField.completeness`, the
    expected matched cross-station pairs per cell (gate x number of POSSIBLE
    image pairs -- station-level averaging over-predicts by 0.2-0.6 dex);
  * link_mode='cross' gives a cell the cross-station geometry when
    completeness >= min_expected_pairs (default 1.0) -- a threshold on a count,
    not a weight on precision.

The fixed-fallback is no longer applied: under 'cross' the fused result cannot
be worse than the best single station, so the rule is unreachable and keeping it
would hide a regression. Verified: 0 cells worse than FIXED, 0 cells identical
to FIXED, at all three sites.

Report TWO maps per case: sigma_z of a matched point (geometry only, masked
where yield < 1) and the yield field. The difference between the cases is
COVERAGE, not blur -- 14 % / 39 % / 88 % of Belva cells clear one expected
matched pair under pessimistic / measured / optimistic.

## v0p9: the measured error model

The gate is now the form MEASURED on the Navcam archive (five sites, 1068
images, 5.2 M re-triangulated pairs), not the Gaussian guess:

    w_ij = A * (1 + theta/theta_c)^(-k) * exp(-|dLMST|/L0),
    theta_c = theta_bar/CV^2,  k = 1/CV^2

with eps_intra = 0.169 px (pooled; 0.154-0.183 per site), correlation kernel
'exponential' with theta_rho = 0.4 deg and rho_inf = 0.22 (caps redundant looks
from one place at 2.1x), and lmst_h on Station driving the illumination gate.
Three matching cases (CASE_PARAMS): pessimistic (A .4, theta_bar 2.4, CV .50),
measured (.4, 4.3, .36), optimistic (.7, 8.6, .36 -- unmeasured placeholder for
a learned matcher). The Gaussian gate survives as gate_form="gaussian" for
comparison only; it was rejected at chi2/dof 38-855.

Pose was also corrected by the archive. Bundle-adjusted poses reproduce sigma_z
to 1-4 %, so SfM cases carry NO pose term; the earlier "0.2 % of baseline"
tier put 2 mrad on 20 m stations and inflated anisotropy to kappa~94. FIXED
now carries the MEASURED telemetry registration error (Metashape vs telemetry:
0.15-0.5 m) as a constant, PoseModel(mode="registered", reg_sigma_m=0.3); the
path-based 'telemetry' tier applied dist_total_m across whole multi-sol
excursions (Rockytop: 454 m) and is wrong for archive clusters.

Consequence for reading the maps: "improvement over FIXED" is now dominated by
registration (~100x, spatially uniform). The geometry-only comparison is
against FIXED with pose 'none'. Report both, and say which.

## Name and scope

This package is **`mppp_error`**: the error-modelling component only. It is
deliberately standalone and has no dependency on the parent MPPP preprocessing
pipeline, so it can be developed and tested independently and merged in later
as `mppp/error/` (or kept as a sibling package) once that codebase is cleaned
up. Nothing here imports from the parent; the only shared artefacts are the
data files in `data/`, which are copies.

`mppp_error.colmap` is the intended seam: it reads a COLMAP sparse model
produced by the parent pipeline and recovers eps, theta_max and the view
graph. That is a file-format interface, not a code dependency, so the two
halves can be joined without either importing the other.

## Integrating with the full MPPP codebase

This package is the error-modelling component. The parent repository is the
preprocessing pipeline: PDS `.IMG` -> PNG + camera models -> Metashape / COLMAP
/ Nerfstudio, plus a ConvNeXt segmentation model that writes rover-hardware and
shadow masks into the PNG alpha channel.

### What was verified against the two real IMG files

Both bundled products (`ZL0_..._0709_...` Zcam and `NLF_..._0709_...` Navcam,
sol 709, site 33, drive 2864) were run end to end. `pds_reader` +
`image_processing` + `camera_models.pose_from_label` all work **after four
bug fixes**; corrected copies are in `integration/`.

| # | file | bug | effect |
|---|---|---|---|
| 1 | `image_processing_.py` | `RadiometryResult(scale=...)` but the field is `scale_to_radiance` | **`apply_radiometry` raised TypeError on every call** -- as delivered it could never have run |
| 2 | `image_processing_.py` | `np.int16(im32_bal * cfg.scale)` | radiance x 1e6 exceeds 32767, **wraps negative**, corrupting the percentile stretch. Now float32, quantised only on output |
| 3 | `pds_reader.py` | `np.moveaxis(arr, 0, -1)` applied unconditionally | correct for band-first 3-band data, but **transposes 2-D single-band images**: a Zcam frame read back 1200x1648 instead of 1648x1200, corrupting every downstream intrinsic |
| 4 | `pds_reader.py` | 4-digit regex for sol | `LOCAL_TRUE_SOLAR_TIME_SOL` is the *string* `'709'` (3 digits), so the regex missed and fell through to `LOCAL_MEAN_SOLAR_TIME='Sol-00709M...'`, matching `'0070'` -> **sol 70 for a sol-709 image** |

A fifth, found while wiring camera constants: **`load_intrinsics_from_xml`
cannot read `params/m20_cmods/`.** It expects OpenCV `FileStorage` (`<K>`/`<D>`
matrices) and raises `SystemError` on the plain `<f>/<cx>/<cy>/<k1>/<k2>` XML
actually shipped. `integration/camera_models.py` adds
`load_intrinsics_from_mppp_xml`, which also handles the convention that **cx/cy
in those files are offsets from image centre, not absolute pixels** (ZL034 has
`cx=30` on a 1648-wide frame). Its IFOVs agree with the v0p6 hard-coded
constants to 4 significant figures, so the hand transcription was right -- but
it is now loadable rather than copied.

Not a bug but worth knowing: the default `ColorConfig.white_balance` of
(1.0, 1.3, 1.0) gives Zcam previews a distinct green cast. The per-camera
gains are present but **commented out** in the dataclass; Z wants
(1.0, 1.40, 2.10). Wire the table up rather than leaving the generic default.

### Which duplicate is authoritative

* **`image_processing.py` vs `image_processing_.py`** -- both are 0 bytes and
  ~14 kB respectively in the zip; `MPPP.py` imports the trailing-underscore
  one. The uploaded file is the real one. **Rename to `image_processing.py`
  and delete the underscore variant** once merged.
* **`mask_predict.py`** is 0 bytes and nothing imports it. This is the missing
  piece -- see the question at the end.
* **`image.py` vs `image copy.py`** (both ~50 kB, differing) and
  **`config.json` vs `config copy.json`**, **`workspace.ipynb` vs
  `workspace copy.ipynb`** -- unresolved. `MPPP.py` imports neither `image`
  module, so both may be dead code superseded by `image_processing_`.
* **`src/error_maps/error_map.py`** is this package's direct ancestor and
  hard-codes one occlusion profile in module scope. **Delete on merge**;
  keeping two divergent error models in one tree is how they diverge further.
* **`src/mppp_old/`** -- superseded; keep only as history.

### Proposed organisation

    MPPP/
      pyproject.toml            <- make this an installable package; fixes the
                                   import chaos below
      params/                   <- single source of truth for constants
        m20_cmods/*.xml           camera models (read, never transcribe)
        M20_waypoints.json
        M2020_occlusion_profiles.csv
        M2020_taus_versus_L_s.csv
        Mars2020_waypoint_shifts.csv
      mppp/
        io/        pds_reader.py  file_io.py  readers.py  writers.py
        camera/    camera_models.py           (+ load_intrinsics_from_mppp_xml)
        process/   image_processing.py        (radiometry, colour, resize)
        masks/     mask_predict.py  train_convnext.py  profiles.py
        error/     core.py metrics.py cases.py network.py viewgraph.py
                   colmap.py waypoints.py sitemap.py plotting.py
                   study_navcam.py selftest.py          <- THIS PACKAGE
        cli.py                    <- the orchestrator now in MPPP.py
      models/      convnext_tiny_seg_best.pt  (git-lfs or fetched, not in repo)
      notebooks/   the .ipynb files, one level down, not beside the library
      data/        sample IMG products for tests

The single highest-value structural change is the `pyproject.toml` +
package layout. `MPPP.py` currently mixes `from pds_reader import PDSImage`
(flat) with `from MPPP.src.image_processing_ import ...` (package-absolute);
**only one can work depending on how it is launched**, which is why
`image_processing.py` ended up with a bare `from camera_models import ...`
that breaks under package import. Everything else is easier once this is fixed.

### Merge order

1. **Package layout + `pyproject.toml`, fix imports.** Mechanical, no
   behaviour change, unblocks everything else.
2. **Land the four bug fixes** from `integration/` with a regression test that
   reads both sample IMG files and asserts sol=709, 1648x1200, and that
   `apply_radiometry` returns finite non-negative radiance.
3. **Drop in `mppp/error/`** (this package) and run its 166-check self-test in
   CI. Delete `src/error_maps/`.
4. **Switch camera constants to `params/m20_cmods/`** via
   `load_intrinsics_from_mppp_xml`, removing the hard-coded IFOVs in
   `error/core.py`.
5. **Mask cross-validation.** The occlusion CSV is a *geometric* rover-frame
   az-el profile; the ConvNeXt model produces a *per-image* alpha mask. They
   describe the same hardware. Project the predicted alphas into rover-frame
   az-el, take the hardware/terrain boundary, and compare against the CSV.
   Disagreement means one of them is wrong -- and agreement generates
   per-camera profiles automatically, removing the Zcam/Ncam duplication.
6. **COLMAP adapter.** `error/colmap.py` already recovers eps_intra, eps_cross
   and theta_max from a sparse model (validated to <1 % against synthetic
   ground truth). Feed it real Metashape/COLMAP solves produced by the
   pipeline. **This is the step that turns the error model from predictive to
   calibrated** and is where the science is.

Steps 1-4 are mechanical; 5 and 6 are the ones worth doing carefully.

### Questions before I go further

1. **`mask_predict.py` and `convnext_tiny_seg_best.pt` are both absent** (0
   bytes / not in the zip). I can write the inference wrapper from the
   training notebooks, but I would be guessing at the preprocessing
   (normalisation, input size, class indices). Can you send the file, or the
   training notebook cell that defines the transform?
2. **`image.py` / `image copy.py`** -- are these dead code superseded by
   `image_processing_`, or is one of them live in a path I have not traced?
3. The occlusion CSV Zcam and Ncam columns are identical. Is that a
   placeholder, or are the two cameras genuinely close enough that one profile
   is intended for both?

## Open items

1. **Verify the instrument constants.** MCZ-34 IFOV is interpolated between the
   published 26 mm (283 µrad) and 110 mm (67.4 µrad) endpoints. Baseline 0.244 m
   is from Hayes et al. 2021; Bell et al. 2021 quotes 24.4 cm / 2.3° total toe-in.
2. **Verify the occlusion profile.** `DEFAULT_ROVER_MASK` is carried over from v1
   with unknown provenance. Confirm against CAD/ray-cast, and record whether it
   references the mast rotation centre or each camera.
3. **Measure θ_c and ρ_∞** on the four existing sites: triangulate multi-station
   tie points pairwise, correlate residuals against angular separation, fit
   ρ = ρ_∞ + (1−ρ_∞)exp(−θ²/2θ_c²). Self-contained, delegable, and no measured
   value exists for Mars surface imagery.
4. **Calibrate ε** by comparing predicted σ_n against achieved error at check
   points. Anchors every prediction downstream.
5. **Source an M2020 VO drift rate** to replace the placeholder 2%.
6. **Feed a HiRISE DTM** via `GriddedSurface`. The τ (transition-tilt) gate is
   uninteresting on a flat plane, and slope is where the four sites differ most.
7. **Per-eye occlusion masks** — the hook exists (`Instrument.eye_masks`); the
   annulus where only one eye sees the ground is exactly the geometry that makes
   the rover's occlusion shadow partially recoverable.

## Known approximations

- Two-stage pose handling (fixed pose with known covariance, then propagated)
  rather than joint estimation. Generally conservative.
- Pose errors independent between stations. For telemetry this understates
  correlation, since VO drift is cumulative.
- Correlation inflation uses the participation ratio N²/(1ᵀC1) rather than full
  GLS. Spot-check against GLS at a few cells before relying on it.
- `GriddedSurface` clamps to the nearest edge value outside the DTM footprint.


## Figure readability: stats boxes and shared axis labels

Each `sigma_n` and `improvement` panel now carries a small mean/rms/median box
(upper right, monospace, matches the panel's own units), and axis labels/tick
labels are shown only on the outer edge of the panel grid -- bottom row gets
"Relative easting", left column gets "Relative northing" -- rather than on
every one of the 8 panels. Both are `plot_map` parameters (`stats_box`,
`show_xlabel`/`show_ylabel`/`show_xticklabels`/`show_yticklabels`), not special
cases in `study_navcam.py`, so they're available anywhere `plot_map` is used.

Retried the Butler Landing image fetch for this round; still blocked
(`mcz-images.sese.asu.edu` not on the environment's egress allowlist, same as
before). Verified the crop/overlay path renders correctly in the new layout
using a synthetic placeholder — see `sitemap.py` below.

## Site imagery overlay (`mppp_error.sitemap`)

`study_navcam.py --site-image PATH_OR_URL --site-image-width-m 50` places a
north-up overhead mosaic behind the FIXED improvement panel (otherwise blank,
since FIXED has no improvement over itself), cropped to the same extent as the
other panels via `crop_to_extent`. **Now using the real, bundled image by
default** — no flag needed; `--no-site-image` disables it, `--site-image`
overrides the path.

**The real Butler Landing image is now bundled and working**
(`mppp_error/data/butler_landing_1_vertical_50m.jpg`, user-supplied). An earlier
attempt to fetch it by URL failed — `mcz-images.sese.asu.edu` is not on this
environment's network egress allowlist — that limitation is now moot since the
file is included directly. Visual check against the rendered figure: the
rover's actual position in the photo sits within a couple of metres of the
plotted anchor marker at the origin, supporting the north-up/flat-projection
assumption below for this product, though it hasn't been checked against a
second independent landmark.

**Alignment assumptions**, stated in the module docstring and not otherwise
checked: the image is north-up, square, of known real-world width, and
centred on the same point the grid is centred on. The crop/alignment
*mechanism* itself is verified against a synthetic image with a known feature
position (`test_sitemap_crop`), recovering it to within 0.05 m — that test
checks the code is correct, not that any particular real image satisfies
these assumptions. Verify against a second known feature before trusting
absolute pixel positions for anything quantitative.

## A note on two bugs this round exposed


Both slipped past "the script ran and printed patched" and were only caught by
looking at the rendered figure — worth recording since the same pattern nearly
recurred:

1. A `str.replace()` targeting code from an earlier edit pass found no match
   (subtle formatting drift) and silently no-opped. The script still exited 0
   and printed "patched." **`test_spacing_sweep_baseline_marker` now asserts on
   actual rendered pixels** — it counts tab:green pixels in the saved PNG and
   checks plotted content isn't collapsed into a sliver — specifically because
   this class of failure produces a working, crash-free script with a visibly
   wrong figure.
2. `set_xlim()` called before `set_xscale('log')` was silently discarded when
   the scale changed. Fixed by reordering; documented inline since matplotlib
   gives no warning when this happens.

**A third, this round:** implementing the new IDEAL case (which sums a single
centre ray per station) surfaced a latent inconsistency in the OLD single-pair
LBS case — it selected its widest-baseline pair using *two-eye* visibility but
then computed precision from the *station-centre* ray, and at an occlusion
boundary those two tests can disagree. This one wasn't caught by "the figure
looks wrong" (LBS wasn't even being plotted) — it was caught by a hard
invariant test (`full_fusion ≤ IDEAL ≤ LBS`, provable from Fisher-information
additivity) failing on ~0.1% of cells. That's the general pattern worth
keeping: geometric/statistical relationships that must hold by construction —
monotonicity, PSD ordering, exact reductions to a known-good closed form — make
better regression tests than "does it run," because they catch the class of
bug where the code executes cleanly and returns a plausible but wrong number.


## Known issues, suspected bugs & future work

A consolidated list, gathered from caveats scattered through the sections
above, for whoever picks this up next.

### Open questions needing a decision, not just code

- **The IDEAL definition may not match intent.** The request that produced it
  said "the maximum of inter range error and the combined error of all
  possible LBS constraints of any pair of stations." A literal
  `max(single-pair, combined)` always returns the single-pair value, since
  combining more independent information provably cannot increase error —
  that reading contradicts "not just one LBS, but the best of all possible"
  in the same message. Implemented the "combined, properly fused" half;
  the "maximum of inter range error" half is unaddressed. See "LBS vs. the
  new IDEAL" above.
- **"theta_c" in the PESSIMISTIC-angle request likely meant theta_max.**
  Changed `CASE_PARAMS["pessimistic"]`'s angle 12°→15° on that assumption.
  If an actual θ_c (error-correlation decorrelation angle) was intended, this
  is a one-line fix, but it's currently a distinct, off-by-default feature
  (`ModelConfig.use_correlation`) not wired into the four-case model at all.

### Verified working but with unverified inputs

- **Site image alignment checked against exactly one landmark** (the rover's
  own position vs. the anchor waypoint, within ~a couple of metres). North-up
  orientation, squareness, and the assumption that the *trimmed* image
  exactly spans 50 m are all unverified beyond that one check. The 25% trim
  fraction is a manual correction inferred from "the image looked too
  zoomed-in," not a measurement — if a second landmark's real-world
  separation is known, that would let this be verified (or corrected)
  properly instead of assumed.
- **`DEFAULT_ROVER_MASK`'s provenance is still unverified** against rover CAD
  or ray-tracing, carried over unchanged from the original prototype across
  every round so far.
- **MCZ-34's IFOV is a linear interpolation** between the two published zoom
  endpoints (26 mm / 110 mm), not a measured value at 34 mm specifically.
  Both instrument baselines should be re-checked against the calibration
  paper (Hayes et al. 2021) before any publication-grade use.
- **VO drift rate (`PoseModel.telemetry`) is a placeholder** (2-3% of
  distance driven), not sourced from a specific M2020 VO performance paper.
  `pose_covariance_from_network`'s BA-derived numbers don't have this
  problem, but are a lower bound (see below).

### Known approximations (by design, not oversight)

- **`pose_covariance_from_network`'s output is a lower bound on pose
  uncertainty.** Grid cells are treated as independent tie points; real tie
  points share matching bias and interior-orientation residual, and terrain
  where matching fails outright contributes nothing. `tie_correlation` gives
  an upper estimate but isn't calibrated either.
- **Two-stage pose handling.** Pose is treated as fixed-with-known-covariance
  and then propagated into point precision, rather than estimated jointly
  with the points. Generally conservative, but not exact.
- **`GriddedSurface` clamps to the nearest edge value** outside the DTM
  footprint rather than extrapolating or flagging invalidity — fine for the
  bundled flat-ground studies, worth revisiting before feeding it a real
  HiRISE-derived DTM with a sharp footprint boundary.
- **ε calibrated via `mppp_error.colmap` is optimistic.** Sparse tie points are
  biased toward well-textured, well-matched terrain — exactly the easy cases
  — so any ε measured this way is a lower bound on true matching error, not a
  central estimate.

### Things that look like they might be bugs but (as far as tested) aren't

- **The IDEAL/LBS/full_fusion ordering held on real data (0 violations across
  14,641 cells)** after the centre-ray-clone consistency fix — but that fix
  was only exercised on one real geometry (4 landing-site waypoints). Worth
  re-running `test_ideal_ordering_holds_on_real_data`-style checks against
  other real waypoint subsets as they become available, since occlusion-mask
  boundary interactions (the actual root cause of the original bug) are
  geometry-dependent and could recur in an unexercised configuration.
- **`crop_to_extent`'s `partial_coverage` flag trips even on a near-exact
  match** (a 50 m image against a ±25 m grid) because `Grid.extent_edges()`
  adds a half-cell pad beyond the nominal ±25 m. This is documented as
  expected behaviour, not a bug, but it means `partial_coverage` alone isn't
  a reliable signal of a *meaningfully* undersized source image — a caller
  wanting that distinction would need to compare against a tolerance rather
  than checking the flag directly.

### New in v0p6 — things to watch

* **θ_min = 2° is a placeholder with a lower bound, not a measurement.** It
  must exceed the intra-station convergence angle (0.42 m at 40 m = 0.6°),
  which the classical range equation already treats as independent — otherwise
  the model calls a station's own stereo pair redundant with itself. At 5° the
  spacing sweep flat-lined at 1× below 5 m for exactly this reason. If it is
  ever raised, re-check the sweep's small-spacing end.
* **The `sfm` pose mode needs an anchor.** It calls `bind_anchor` automatically
  inside `solve_precision_field`, but a `PoseModel(mode="sfm")` constructed and
  used outside the solver will silently fall back to `path_m` scaling. Call
  `bind_anchor(stations)` explicitly in that case.
* **`run_four_cases(pose=...)` overrides the per-case mapping.** Passing an
  explicit pose applies it to *every* case, which is right for invariant
  testing and wrong for delivered numbers. The default (`pose=None`) is what
  the figures use.
* **The four-case medians for "existing style" and "proposed" have converged**
  (14.9× vs 15.4× at ±20 m) now that both use SfM pose. The compact network's
  advantage is in *coverage* (1.00 vs 0.95), *anisotropy* (9.0 vs 4.7) and
  *connectivity* (λ₂ 6.87 vs 1.40), not in the median. Any slide claiming a
  median advantage should be rewritten around those three instead.

### Straightforward next steps

- **Rover glyph tuning.** Proportions in `_rover_glyph` (chassis 2.9 × 1.4 m,
  wheels 0.52 × 0.40 m on a 2.3 m track, mast 1.3 m forward / 0.58 m
  starboard of body centre) are eyeballed from a top-down render, not CAD. Fine for now given the
  explicit "don't make this too busy" framing, but a real M2020 footprint
  (chassis dimensions, wheel positions, actual mast-to-body-centre offset)
  would make this genuinely illustrative rather than merely suggestive.
- **`study_cross_station.py` is unmaintained relative to `study_navcam.py`.**
  Still runs, still uses its own independent `OPS`/`ideal` naming predating
  the FIXED/LBS/IDEAL split. Either fold it into the same `mppp_error.cases`
  machinery or clearly mark it deprecated so the two scripts' terminology
  stops silently diverging.
- **θ_c and ρ_∞ remain unmeasured for Mars surface imagery.**
  `mppp_error.colmap.measure_theta_c` exists and is tested against synthetic ground
  truth, but has not yet been run against a real reconstruction. This is
  probably the single highest-value next measurement: it would upgrade the
  correlation model from "a documented guess with a citation to geodetic
  practice" to an actual calibrated parameter.
- **No test yet exercises `--rmcs` with a hand-supplied, non-default
  waypoint list**, or the fallback path when a supplied RMC's `dist_total_m`
  field is absent (falls back to straight-line distance — code path exists,
  untested).
