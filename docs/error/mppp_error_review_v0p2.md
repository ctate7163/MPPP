# mppp_error v0p15 — code review at merge (MPPP v0p2, 22 Sep 2026)

The package was merged into `mppp.error` with no change to its numerics. Its 186 self-tests pass in the merged layout. This review was done by an independent pass, and the findings marked **confirmed** were reproduced with a script. Nothing below has been fixed, because each item either changes published numbers or is a modelling decision. The one exception is item 7a, a crash on a code path that could not have produced any number.

## Changes published numbers

**1. The SfM cell model puts a probability into Fisher information** (`core.py:971-979`). Confirmed.
- The v0p14 model is `E[Λ] = Λ_nearest + P·Σ_{j≠nearest} Λ_j`. This contradicts the v0p11 rule: "the gate is a survival probability … it was being multiplied into Fisher information."
- A few tied rays from other directions pin the weak range axis of the nearest pair. So inv(E[Λ]) falls steeply at tiny P, while the precision of a matched point does not move.
- Example cell (Navcam at 20 m, second station 8 m off):

  | P | inv(E[Λ]) σ_z | Mixture RMS √((1−P)σ_F² + Pσ_T²) |
  |---|---|---|
  | 0 | 0.77 cm | 0.77 cm |
  | 0.001 | 0.54 cm | 0.77 cm |
  | 0.01 | 0.24 cm | 0.77 cm |
  | 0.1 | 0.11 cm | 0.73 cm |

- Landing site, where the median P is 0.011: FIXED 0.42, SfM (E[Λ]) **0.13**, mixture 0.41 cm.
- inv(E[Λ]) is correct for one thing only: the per-point-normalised precision of an information-weighted cell mean over many points. It is not the "σ_z of a matched point" that the two-observable framing reports.
- Suggested fix, consistent with v0p11: report σ_z given a tie (the P = 1 geometry) together with the P map, or report the mixture. The v0p14 medians (Belva 0.47 / 0.13 / 0.05 cm) need re-deriving under whichever definition you choose.

**2. The correlation inflation ignores station weights** (`core.py:1060-1067, 1101-1138`). Confirmed.
- The default kernel is `"exponential"`, not `"cluster"` as the ModelConfig comment says.
- N/N_eff is computed over every visible station, including those carrying P ≈ 0.
- Synthetic two-station case: SfM comes out 7.5 % worse than FIXED. On Belva, 49 % of cells are inflated, by up to 7.6 %.
- The inflation is also applied to IDEAL but not LBS, so IDEAL ≤ LBS is not protected by construction.
- Fix: a weighted participation ratio (Σw)²/ΣΣ w_i w_j ρ_ij over contributing stations, or GLS.

**3. The ε estimators in `colmap.py` are biased low** (`colmap.py:265, 306`). Confirmed by simulation.
- Residuals after triangulation keep (2n−3)/2n of the variance per track. A 2-view track keeps ¼, so ε comes out ×0.5.
- In the synthetic check, true 0.40 / 1.10 px (intra / cross) was recovered as 0.19 / 0.89, and the cross/intra ratio went from 2.75 to 4.61.
- Dividing Σr² by Σ(2n−3)/n recovers 0.39 / 1.10.
- `calibrate_eps` and `calibrate_eps_from_residuals` also differ by √2 (one per-axis, one not), and COLMAP's stored `error` is a mean |r|, not an RMS.
- **Question:** was ε = 0.169 px (from "5.2 M re-triangulated pairs") corrected for this? If not, the true per-axis ε could be up to 2× larger. The 1-4 % σ_z agreement would only survive that if the observed σ_z was itself derived from the same residuals.

**4. The pixel scale behind ε = 0.169 px is unstated.**
- The Navcam IFOV in `core.py` (3.393e-4 rad) is the full-resolution 5120-px value; it agrees with 1/f from the XML to 0.1 %.
- If ε was measured on 1280- or 2560-px archive products, the angular error is understated 4× or 2×.
- **Question:** which resolution were the five-site Navcam products?

## Latent (no published number affected, as far as can be seen)

5. `colmap._project` is pure pinhole, so lens distortion ends up inside the residuals (about 10 px at the edge for f = 1400, k = −0.05). Fisheye parameters are misread, and there is no binary-model reader.
6. `measure_theta_c` builds its rays toward `p.xyz`, so every pair re-triangulates exactly onto the point and ρ ≈ 0 (rounding noise). No test calls it.
7. Waypoint handling:
   - (a) **Fixed at merge.** A second `_last_per_sol` shadowed the first, so `build_stations` (used by `python -m mppp.error run` and `compat.py`) raised `TypeError`. The feature-list version is renamed `_last_per_sol_features`, and `tests/test_error_model.py` covers it.
   - (b) **Not fixed, confirmed.** "Last per sol = highest drive" ignores site changes. On sol 1842 it keeps `87_5286` (`final='m'`, mid-drive) instead of `88_0` (`final='y'`), 4.2 m apart; 18 sols are affected, by up to 175 m. **Check whether the frozen site-87 prediction includes sol 1842.** Fix: order by (site, drive), or prefer `final=='y'`. This is pinned by a test so that any change is deliberate.
8. Left/right eye occlusion masks are swapped (`core.py:821-840`): eye 0 sits on the right but receives the left profile. The profiles differ by up to 30°, so this only matters in the near field.
9. Smaller items:
   - `PoseModel(mode="sfm")` raises `AttributeError` (`_anchor_xyz`).
   - The mast-offset rotation in `waypoints.py:44-47` is wrong; the offset defaults to 0, so it is unused.
   - `metrics.n_eff` uses a Gaussian kernel with ρ₀ = 1, not the configured kernel.
   - `theta_min_deg` is ignored by the non-cluster kernels.
   - `network.py` uses `eps_intra_px` and ignores `eye_masks`.
   - Several stale comments: ρ_∞ 0.22, L_eff 1.5 h, "cluster is default", Z34 baseline 24.3 vs 24.4 cm.
10. `lmst_h` is never set by any loader, so the illumination gate is 1 inside this package. The external script behind the deck may set it; `maps/make_three_cases.py` was not in the upload, so this is unverified.

## Tests that check the code against itself

These should be replaced by independent truth:
- The p_cross formula is re-typed rather than tested independently.
- "Completeness uses L0" compares two constants.
- In `test_two_ray_analytic`, the "textbook" value is the same expression as the code.
- The naive reference shares the eye-placement and mask conventions.
- The COLMAP round trip writes the true point positions, so it cannot see the degrees-of-freedom bias in item 3.

## Checked and correct

- Single-ray information (I−uuᵀ)/(εψr)².
- Two-ray σ vs √2·ε·ψ·r²/b, within 0.1 % from 5 to 40 m.
- N-station ring gives 1/√N.
- Pose marginalisation (Λ⁻¹+S)⁻¹.
- Gate θ_c = θ̄/CV², k = 1/CV²; midnight wrap.
- The 2.1× repeat-look cap.
- FIXED ⪰ SfM ⪰ BEST in Λ.
- COLMAP quaternion convention and camera centre.
- Azimuth and ENU conventions.
- IFOVs against the XML calibrations: Navcam 0.1 %, Z34 exact, Z110 0.07 %. The XMLs look hand-rounded (f = 4720, k1 = −0.45) rather than taken from flight CAHVOR models.

## Relation to `code/integration/`

Those modules come from the `MPPP.py` / `image_processing_` lineage, while MPPP v0p1 was built from the `image.py` / `workspace.ipynb` lineage. The four integration bugs are absent from v0p1 by construction:
- **Band order:** 2-D images are not transposed (Zcam reads 1200×1648; tested).
- **Sol:** taken from the fixed-width file name (709; tested).
- **Quantisation:** radiance stays float until clipped quantisation.
- **XML cx/cy:** read as offsets from the image centre (tested).

`integration/` can therefore be retired, with the new modules in its place.
