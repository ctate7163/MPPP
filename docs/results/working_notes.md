# Working notes: interim numbers (not methods)

**Status.** These are measurements made on particular scapes while the pipeline was being developed. They are kept here, out of `docs/methods.md`, the notebooks and the code, because they are data, results and discussion for the paper and not a description of the method. Every number is tied to the version that produced it and may be superseded by a later run; the paper's tables are made from the notebook outputs of the release used for the paper, not from this file. Where a later measurement contradicts an earlier statement, the later one is kept and the earlier is marked *withdrawn*.

Notebook outputs referred to: notebook 04 (`error_analysis/`: `eps.csv`, `gate_fits.csv`, `gate_forms` table, `rho_fits` table, `eps_vs_theta.png`, `rho.png`), notebook 05 (`camera_analysis/`).

## 1. Withdrawn statements

- *"ρ ≈ 0 (0.00–0.03 at 0.25–5°)"* (CHANGELOG 0.15.0; the shared-image estimator without bias correction). **Withdrawn.** The v0p22.2 estimator (pairs sharing no image, −1/(M−1) constraint bias removed, bootstrap SE; `decorrelation`, `fit_rho`) finds ρ not consistent with zero at the smallest separations on all three v0p22 sites: 0–0.1°: Airey Hill 0.157 ± 0.007, Bell Island 0.072 ± 0.004, Three Forks 0.090 ± 0.006; ≤ 0.02 beyond 0.25°, with a floor of ≈ 0.01 at Bell Island and Three Forks. Gaussian fits ρ_∞ + ρ_0 exp(−θ²/2θ_c²): ρ_0 0.19 / 0.07 / 0.06, θ_c 0.08° / 0.09° / 0.37°, ρ_∞ ≈ 0 / 0.012 / 0.013 (AH / BI / TF). The archive's ρ_0 = 0.22, θ_c = 0.4° (docs/error, v0p15) is of the same order at zero separation but decays faster.
- *"ε is flat with convergence angle"* (archive, and the v0p22 methods text). **Qualified.** Section 4 below: cross-station ε rises slowly with the track's largest ray angle (about +35 % from 0–2° to 20–45° at Bell Island); same-station ε is flat to 10°. At equal angle (3–5°) cross-station ε exceeds same-station ε (0.30 vs 0.24 px at Bell Island), so the cross excess is not the angle alone.
- *"The convergence the tie points reach is set mostly by where the rover stopped and pointed, not by the matcher"* (methods §13, v0p22). **Withdrawn as a conclusion.** It is an open question. Section 5 below gives what the SIFT variants did; a learned matcher comparison (SIFT + LightGlue / ALIKED / LoMa) on one site is the planned test, and the SIFT numbers are the baseline it is compared with.
- *"The gate shape (θ̄, CV) is held at the archive values"* — since v0p22.2 every gate parameter is fitted (`fit_gate`), and the angle form and illumination covariate are chosen by AIC (`compare_gate_forms`). The archive defaults (A 0.4, θ̄ 4.3°, CV 0.36, τ 2.3 h) remain only as `ModelConfig` defaults to be replaced by the fitted values.

## 2. Camera models (v0p20–v0p22.1)

**Rational Navcam model, first fit (v0p20).** Re-fitted to the Three Forks Navcam alignment (sols 684–693, 52 images) including the frame corners, scene held fixed: 0.6 px inside 80 % of the corner radius, 1.0–1.7 px beyond 85 % (85–96 % of the corner observations within 3 px). The v0p20 shipped values were the mean of two full bundle adjustments started from that fit, Three Forks and Bell Island (sols 1451–1467, 145 images); the two sites agreed to 0.5 % in k1 and k4 and to 1 px in f; median residuals 0.18 and 0.15 native px, against 0.23 and 0.19 with the polynomial. With the polynomial model at Three Forks, 45–57 % of the corner keypoints had verified matches but 19 % (0.85–0.90 of the radius) and 0 % (beyond 0.90) were triangulated: about 9 % of every frame was lost (`corner_triangulated_ratio` 0.20).

**Consensus (v0p22).** The shipped rational cameras are the observation-weighted consensus of Three Forks, Bell Island and Rockytop (1,540,661 observations), 0.15–0.17 px rms from the v0p20 values. The rig rotations of the three agree to 0.15 mrad.

**Label vs rational (v0p21).** The test product's Navcam label (CAHVORE type 2, E ≈ 10⁻⁸ m) agrees with the rational model to 2.7 px rms (1.7 centre, 4.5 corners) after a 0.09° rotation; read as perspective CAHVOR it is 250 px off and 30 % of the frame projects outside. CAHVORE fitted to the rational model: type 2 0.4 px rms (1 px corners), type 3 0.2 px; CAHVOR 9 px rms, 120 px in the corners. Older manifests without the full label model: taking O along A costs up to 4 px (Navcam) and 0.5 px (Mastcam-Z).

**Repeatability (v0p21, seven v0p15 polynomial scapes and three v0p20 rational solutions).** Rational cameras agree with their consensus to 0.1–0.55 px rms (under 1 px in the corners); polynomial ones to 0.6–4.7 px, 1–21 px in the corners. Label CAHVORE pair with the label rig: 1.3–1.9 px less disparity at the image centre than the refined geometry (ranges ≈ 1 % short at 10 m, 2 % at 20 m); refined geometries agree with their consensus to ≤ 0.4 px. Mastcam-Z: refined f rises 0.046–0.048 px per focus count (label 0.047–0.050) but lies 43–49 px (0.9–1.0 %) above the label at the same focus, repeating to 4–5 px between scapes; pairs processed with the label geometry have +1.6–1.8 px median disparity error at 10 m (±1.3–1.7 px pair to pair), one fixed median rig brings this to +0.1–0.2 px (±0.9–1.0 px). These Mastcam-Z numbers come from polynomial-Navcam runs.

**Repeatability (v0p22.1, three mixed Navcam + Mastcam-Z runs refining their own Navcam cameras from the shipped consensus).**

| scape | Navcam images (left) | stations, span | difference from the shipped rational camera, rms (centre / corners) | rotation absorbed | mean residual field |
|---|---|---|---|---|---|
| Bell Island 1451–1467 | 101 | 6, 10 m | 0.28 px (0.15 / 0.47) | 0.013° | 0.05 px |
| Airey Hill 961–991 | 60 | 3, 7.5 m | 0.41 px (0.15 / 0.82) | 0.055° | 0.06 px |
| Three Forks 684–693 | 35 | 3, 3.3 m | 8.5 px (4.3 / 14.8); c_y −45 px, f +10 px, rig yaw −146 mdeg | 0.89° | 0.21 px |

Reading: the residual field inside a scape (0.05–0.08 px) is far below ε while the cameras of two good scapes differ by 0.3–0.4 px and the Three Forks camera is 8 px off with its block fitting to 0.13 px median — the between-scape scatter is the unobservable part of the intrinsics, not lens shape. The three refined stereo geometries differ from the consensus by 0.1–0.6 % in range at 10 m. This is why v0p22.2 holds the Navcam intrinsics on weak networks (`navcam_intrinsics="auto"`, 4 stations / 5 m) and why the consensus should come from a joint calibration bundle.

**Attitude prior (v0p20).** Without it the Three Forks rational-camera block tilted 0.59° about East (0.17° with the polynomial).

**Stereo baseline (v0p22).** Rig pose freed on the refined Three Forks, Bell Island and Rockytop rational solutions with 2 mm, 10 mm and no prior on the translation (all three the same to 0.01 mm):

| | baseline change | centre direction change | rig rotation change | cost change | median residual change |
|---|---|---|---|---|---|
| Three Forks (52 images) | −0.07 mm | 0.20 mm | 5 mdeg | −0.22 % | −0.0003 px |
| Bell Island (145 images) | +0.03 mm | 0.63 mm | 15 mdeg | −0.28 % | −0.0007 px |
| Rockytop (168 images) | −0.01 mm | 0.47 mm | 12 mdeg | −0.21 % | −0.0004 px |

The CAHV pairs agree to 10 µm (Belva, 100 pairs: 1.3 × 10⁻⁴ ° and 10 µm; baseline 0.42436 m). 0.1 mm of baseline is 0.02 % of range.

**Mastcam-Z priors (v0p13–14).** The label CAHVOR models move the principal point with focus (ZR034 at Three Forks ≈ 160 px); rotating each prior attitude to the COLMAP principal point brought the left/right relative rotations of the Three Forks pairs from a 1.29° spread to 0.08° (≈ 7 px at f = 4700 px, still too much for a rigid rig). Focus bins of one or two images scattered by about ±100 px in f before they were held (v0p22).

## 3. Alignments (v0p9–v0p22.1)

- Belva Navcam (478 images, sols 748–815, v0p9): all registered, 297 k points, median residual 0.22 native px. Health (v0p13, prior-pair matching): 55 of 478 frames held at their prior, 0.13° rotation of the refined right camera relative to the CAHV rig (principal points moved 2–4 px at the same time), residuals 0.18 px centre → 0.31 px outer quarter, 6 of 13 stations outside the main tied block.
- Outlier frames (v0p22 defaults on the three rational solutions): 0, 1 and 1 of 28, 87 and 102 frames flagged; both flagged frames were weakly tied (26 and 0 observations). v0p22.1 separates these as "unconstrained".
- Two-view tracks (v0p22): about half of all points; on the v0p20 solutions the DOF-corrected ε agreed between all tracks and tracks of ≥ 3 (0.284 vs 0.288 px).

## 4. Error-model inputs, v0p22 sites (notebook 04, tracks ≥ 3; Airey Hill 961–991, Bell Island 1451–1467, Three Forks 684–693, all Navcam + Mastcam-Z 34)

**ε (native px, all cameras).** intra / cross: Airey Hill 0.31 / 0.34, Bell Island 0.24 / 0.33, Three Forks 0.20 / 0.32; DOF-corrected 0.29–0.41.

**ε against convergence angle (`eps_by_angle`, `eps_vs_theta.png`).** Bell Island cross-station: 0.28 px at 0–2° of largest ray angle rising to 0.37–0.39 px at 20–45°; same-station flat at 0.23 px to 10°. At equal largest angle (3–5°): intra 0.237, cross 0.301 px. Precision degrades slowly (~+35 % over 40°) while completeness falls with an e-fold of ~5° (the gate): the completeness loss, not the precision loss, is what the convergence gate should carry, and the effective ε of a cross-station point is the ε at its angle from this table, not one number.

**Gate (`fit_gate`, all parameters fitted, power form, |ΔLMST| covariate, θ ≤ 30°).** Bell Island: A 1.41, θ̄ 6.3°, CV 0.71, τ 2.99 h. Earlier (τ held at 2.3 h): Airey Hill θ̄ 4.0°, CV 0.47; Bell Island 4.9°, 0.80; Three Forks flat with angle (three stations; unconstrained).

**Gate forms (`compare_gate_forms`, ΔAIC from the best on the same binned trials).** The power / gamma-mixture form (1 + θ/θ_c)^−k is preferred everywhere over exponential, stretched exponential and logistic (Bell Island: ΔAIC 297 vs stretched, 1530 vs exponential; logistic worst → no hard θ_max cut-off). Illumination covariate: |ΔLMST| best at Bell Island (τ 3.0 h) and Three Forks (0.84 h, confounded with station identity); the sun-vector angle best at Airey Hill (e-fold 51°); the shadow-tip distance (|Δ(−cot e (sin a, cos a))|) clearly worst everywhere because cot e blows up at low sun; "none" worse by thousands of AIC. corr(|ΔLMST|, Δshadow-tip) over the binned cells 0.67–0.99, so the two covariates cannot be separated on these sites; the choice needs a site with a wide LMST spread at similar geometry (Taylor Fjellet, 6.7–21.6 h).

**ρ(θ).** Section 1.

**Camera repeatability.** Section 2.

## 5. Convergence and the matcher (v0p22; Three Forks 684–693 test set)

- Triangulation (fixed refined poses, then a full adjustment): transitivity 3 and 5, completion transitivity 8, 4° instead of 2° ray-angle tolerance — all within 1 % of the 34,758 points above 10° and within 4 % of the 4,809 above 20°. The tracks are limited by the matches, not by how they are chained.
- Matching on 30 cross-station pairs sampled across the convergence range, counting verified matches whose rays meet within 2 mrad: ratio test 0.9 gave 18 % more matches above 10° (1,256 → 1,486); a 12 px verification threshold gave nothing; no variant gave matches above 20° on pairs whose median convergence is 20–47°.
- Whole pipeline (28-image Sol 684–690 Navcam + Mastcam-Z set):

| variant | tie points | > 10° | > 20° | cross-station > 20° | median residual | extraction (CPU) |
|---|---|---|---|---|---|---|
| default (ratio 0.8) | 59,143 | 17,259 | 3,159 | 555 | 0.131 px | 103 s |
| ratio 0.9 | 60,525 | 18,657 | 3,679 | 578 | 0.136 px | 103 s |
| DSP-SIFT | 60,277 | 16,524 | 2,479 | 565 | 0.133 px | 173 s |
| affine shape | 53,445 | 15,538 | 2,660 | 616 | 0.133 px | 142 s |
| affine + DSP | 52,778 | 15,273 | 2,727 | 581 | 0.134 px | 201 s |
| affine + DSP + ratio 0.9 | 54,335 | 16,317 | 3,071 | 616 | 0.138 px | 201 s |

No tie point in any variant exceeds 40°. What this shows is the best that upright SIFT with COLMAP verification does on this set; whether the station geometry or the descriptor limits the convergence is not decided by it (section 1).

## 6. Single-pair COLMAP test (v0p22.1, `scripts/pair_experiment.py`; 14 stereo pairs of sols 684–690: 9 Navcam half-resolution, 5 Mastcam-Z 34)

| variant | pairs reconstructed | Navcam | median reprojection |
|---|---|---|---|
| COLMAP defaults (intrinsics unknown, SIMPLE_RADIAL, focal prior 1.2 × width) | 4 / 14 | 0 / 9 | — |
| defaults, no planar rejection, init angle 1° | 7 / 14 | | |
| label as perspective CAHVOR (pinhole + R1, R2), fixed | 14 / 14 | 9 / 9 | 0.60 px, 44 % fewer points |
| MPPP camera fixed | 14 / 14 | 9 / 9 | 0.11 px |

Context: Li et al. (2025, Martian World Models) report COLMAP usable on 71.8 % of their PDS Navcam/Mastcam stereo pairs. Here the failures are a camera-model problem specific to the 96° × 73° Navcam (the label is CAHVORE type 2, a fisheye model), not texture or illumination: they disappear when the calibration the PDS label carries is used in the right model class. The perspective reading "works" while being wrong by up to hundreds of pixels in the corners (section 2).

## 7. Sites and scapes

See `docs/results/sites.md` (tables by scape and by site: sol ranges, images, stations, span, LMST spread; `sites.csv`, `sites_by_site.csv`) and `sites_lmst.png` (per site, each camera normalised to unit area).

## 8. Rockytop Navcam worked example (v0p22.2, 27 Sep 2026)

Block: 168 Navcam images, site 26, sols 461–509, 9 stations over 21.6 m; re-adjusted with 0.22.2 (consensus start cameras and rig, outlier exclusion, attitude priors, intrinsics refined: strong network) from the v0p15 features and exhaustive matches (extraction at 3200 px, ratio 0.8). 165 of 168 registered (1 outlier at 0.69 px, 1 without tie points, 1 right image without its left). 426,472 points (50 % two-view, 7.9 % cross-station); median / rms / p95 residual 0.164 / 0.348 / 0.718 px; health PASS. Station shifts from telemetry 0.17–0.41 m; block vs telemetry similarity: scale 1.0045, rotation 1.27°. Camera change from the consensus: f 0.013 %, pp 0.5 px (Rockytop is one of the consensus members, so this is not a validation).

Tracks ≥ 3 (215,166 points): ε all 0.279, intra 0.268, cross 0.320 px (DOF 0.333 / 0.364). ε vs largest ray angle: cross flat 0.31–0.34 px from 0 to 90°; intra 0.22–0.26 px to 5°, 0.28–0.30 px at 5–15° (near-field, confounded with range). At 3–5°: intra 0.236, cross 0.314 px.

Gate (power / |ΔLMST| best; ΔAIC stretched 106, sun-angle 158, exp 225, logistic 827, no covariate ≥ 1,911): A 1.19 (1.11–1.25), θ̄ 4.7° (4.3–5.2), CV 0.41 (0.34–0.46), τ 3.9 h (3.6–4.2), θ_half 4.3°, s_intra 0.48, χ²/dof 14, 232,627 cross trials over |ΔLMST| 0–8.2 h.

ρ (disjoint pairs): 0.068 ± 0.003 at 0–0.1°, ≈ 0.03 from 0.1 to 2°, 0.01–0.02 to 15°, 0 at 30°. Gaussian + floor: ρ_0 0.053, θ_c 0.107°, ρ_∞ 0.021 ± 0.001, χ²/dof 30 (two scales; the single Gaussian does not describe it).

## 9. Lens terms (v0p22.2, `scripts/lens_terms_experiment.py`)

Final adjustment repeated from the converged block with the same observations (TF 431,118 / BI 1,166,323 / RT 1,412,950), both eyes:

| variant | Δcost TF / BI / RT | ΔBIC TF / BI / RT | corner median px TF / BI / RT | residual-field rms px TF / BI / RT |
|---|---|---|---|---|
| k4, p1, p2 | 0 / 0 / 0 | 0 / 0 / 0 | 0.255 / 0.228 / 0.243 | 0.038 / 0.023 / 0.021 |
| + k5 | 0 / −11 / −50 | +27 / +7 / −70 | 0.255 / 0.227 / 0.243 | 0.038 / 0.023 / 0.021 |
| + k5, k6 | −8 / −11 / −58 | +39 / +38 / −57 | 0.256 / 0.227 / 0.243 | 0.038 / 0.023 / 0.021 |
| p1 = p2 = 0 | +27,570 / +49,175 / +51,825 | +55,086 / +98,292 / +103,591 | 0.418 / 0.409 / 0.398 | 0.127 / 0.115 / 0.090 |
| + k5, k6, p = 0 | +27,479 / +49,056 / +51,559 | | 0.420 / 0.406 / 0.396 | 0.127 / 0.114 / 0.090 |

k5, k6: no residual statistic changes in the third decimal; values not repeatable (RT k5 +0.13 NL, −0.06 NR; BI +0.06 / +0.03; TF 0) while k4 moves 0.44–0.97 to compensate; the monotonic range shrinks from 2.0–2.7× the corner radius to 1.07–1.5× (0.99 in one p = 0 case). Keep held at zero.
p1, p2: removing them raises the median residual 16–31 %, the corner residual 60–80 %, the residual field 4–5× (corner cells 0.24–0.32 px); the camera moves 0.5–2.3 px rms (1.3–4.4 px in the corners). Values repeat across sites: NL p1 (1.3–1.8)e-4, p2 (1.7–1.9)e-4; NR p1 (−0.1–0.4)e-4, p2 (−1.04 to −0.96)e-4 → a physical decentering/tilt per eye; keep refined (or hold at the consensus).

## 10. Revisits within 5 m (`NEARBY_M`, `stations_near`)

Waypoint stations of other sols within 5 m of a scape's stations: Three Forks 684–693: S032D1214 (sol 693, 0.01 m); at 10 m also S024D3076 (sol 433, 5.8 m). Bell Island: S072D0542 (sol 1477, 4.1 m); at 10 m also S073D0000 (6.7 m). Belva: S038D2102 (sol 766, 2.4 m). Rockytop, Taylor Fjellet, Airey Hill: none within 10 m.

## 11. Lens model: rational vs θ-polynomial fisheye (v0p30, `scripts/lens_model_experiment.py`; Three Forks 684–693 and Bell Island 1451–1467, Navcam both eyes)

Final adjustment repeated from the converged rational block with the same observations (TF 431,118 / BI 1,166,323), the cameras re-initialised in each model and refined with the poses and points; rig, attitude and position priors as in the pipeline. Models: `FULL_OPENCV` rational (k1–k4 + p1, p2; 8 lens dof), `OPENCV_FISHEYE` (θ-polynomial k1–k4, no tangential; 4 dof), `THIN_PRISM_FISHEYE` with sx1, sy1 held (k1–k4 + p1, p2; 8 dof, "fisheye + tangential") and with sx1, sy1 free (10 dof).

| model | cost TF / BI | median px TF / BI | corner median px TF / BI | residual field rms px TF / BI (corner cells) | TF-vs-BI camera difference, shift removed, rms px NL / NR |
|---|---|---|---|---|---|
| rational | 71,724 / 196,977 | 0.152 / 0.151 | 0.255 / 0.228 | 0.038 (0.078) / 0.023 (0.062) | 0.42 / 0.71 |
| fisheye, no tangential | 99,258 / 246,087 | 0.19 / 0.190 | 0.41 / 0.408 | 0.12 (0.32) / 0.114 (0.321) | 1.94 / 2.09 |
| fisheye + tangential | 70,628 / 195,659 | 0.150 / 0.149 | 0.210 / 0.208 | 0.025 (0.025) / 0.019 (0.054) | 0.16 / 0.45 |
| + thin prism free | 70,344 / 194,767 | 0.149 / 0.148 | 0.20 / 0.200 | 0.02 / 0.012 (0.026) | 0.64 / 0.58 |

Reading: (i) the tangential terms carry the same information in either radial family — dropping them costs +38 % / +25 % as in §9, whichever radial polynomial is used. (ii) At equal parameter count the θ-polynomial beats the rational: cost −1.5 % / −0.7 %, corner residual −18 % / −9 %, residual-field corner cells −68 % / −13 %, and the two sites agree better (0.16 / 0.45 px rms after the common shift, versus 0.42 / 0.71 for the rational). Fitted values repeat across sites (NR k1 0.0480 / 0.0472, k2 −0.0177 / −0.0144, p1 −0.85e-4 / −0.82e-4, p2 −2.03e-4 / −2.04e-4); k3, k4 trade against each other as k4/k5 do in the rational. (iii) Freeing the thin-prism terms lowers the cost another 0.4 % but the camera is no longer repeatable (the sx1, sy1 terms trade against the principal point: 4–8 px common shift between the sites, 0.6 px rms after removing it) — hold them at zero, as k5, k6 in §9.

Decision for v0.30: the rational model stays the pipeline default. The gain of the fisheye + tangential model is real but small (a few hundredths of a pixel in the corners, none in the centre), the shipped consensus, the CAHVORE conversion, the export and the notebook-05 analysis are all written for the rational parameters, and the error model of notebook 04 does not depend on the lens family. A `navcam_model` switch (start cameras, consensus and CAHVORE conversion for `THIN_PRISM_FISHEYE` with sx1, sy1 held) is the candidate change for a later version, to be judged on the five-site repeatability rather than on the two sites here.

## 12. Nine-site review of the 0.22.4 alignments (28 Sep 2026; Navcam only, 16,000 features, three-round schedule, attitude prior 5°)

| site | stations | images reg. / project | median / rms px | ε intra / cross px | block rotation vs labels, about E, N, U (deg) | attitude about it, median / p95 (deg) | layout scale, rotation vs waypoints | scale as displacement |
|---|---|---|---|---|---|---|---|---|
| Three Forks | 4 | 112 (16 excluded) | 0.175 / 0.341 | 0.227 / 0.254 | 0.04 (+0.02, +0.01, −0.04) | 0.10 / 0.69 | 0.41 %, 0.82° | 0.02 m |
| Rockytop | 9 | 143 / 144 | 0.180 / 0.388 | 0.262 / 0.330 | 0.46 (−0.19, +0.13, +0.40) | 0.22 / 0.50 | 0.45 %, 0.96° | 0.10 m |
| Belva Crater | 5 | 80 / 82 | 0.142 / 0.329 | 0.206 / 0.310 | 0.08 (0.00, −0.06, −0.05) | 0.18 / 0.26 | 2.13 %, 2.29° | 0.22 m |
| Pearce Canyon | 11 | 160 / 176 | 0.214 / 0.441 | 0.218 / 0.364 | 0.32 (−0.09, +0.10, −0.29) | 0.24 / 0.35 | 0.87 %, 1.8° | 0.26 m |
| South Arm | 6 | 104 | 0.159 / 0.325 | 0.181 / 0.286 | 0.07 (−0.02, +0.03, −0.06) | 0.35 / 0.42 | 1.89 %, 3.15° | 0.14 m |
| Bell Island | 8 | 190 | 0.167 / 0.372 | 0.221 / 0.333 | 0.26 (+0.05, +0.25, −0.02) | 0.25 / 0.48 | 0.15 %, 0.98° | 0.02 m |
| Taylorfjellet | 11 | 253 | 0.197 / 0.418 | 0.256 / 0.380 | 0.28 (+0.03, −0.01, +0.28) | 0.23 / 0.46 | 0.08 %, 0.23° | 0.03 m |
| Olifants | 15 | 210 | 0.172 / 0.384 | 0.231 / 0.362 | 0.19 (+0.08, +0.10, −0.15) | 0.20 / 0.28 | 0.26 %, 0.87° | 0.13 m |
| Marble Mountain | 4 | 64 | 0.175 / 0.379 | 0.250 / 0.337 | 0.06 (−0.05, −0.02, −0.04) | 0.13 / 0.23 | 1.94 %, 1.21° | 0.33 m |

Block rotation: the median world-frame rotation taking each frame's label attitude to its refined one (from `sparse/cahv_initial` and `sparse/cahv_ba` frames; signs as the corrected v0.30 check). Every block stays within 0.46° of the East–North–Up frame the labels define; the larger ones are turns in azimuth (Rockytop +0.40°, Pearce −0.29°, Taylorfjellet +0.28°). Per-frame attitude scatter about it, median 0.10–0.35°, p95 0.23–0.69°, is the label pointing knowledge; the undivided p95 (0.30–0.77°) had warned at seven sites.

Layout against the waypoints: the similarity between waypoint and refined station positions has scale errors of 0.1–2.1 % and rotations of 0.2–3.2°, largest in the smallest blocks. As displacements at the stations (|s − 1| × √Σr²) they are 0.02–0.33 m, the size of waypoint errors, and the rotations 0.07–0.95 m. The solution's attitude follows the labels, so these are errors of the waypoint layout, not rotations of the solution.

Pearce Canyon: the 16 unregistered images were right-only exposures (their left partners are not in the archive selection); fixed in v0.30 (one-sensor rig).

Navcam intrinsics, nine-scape consensus (notebook 05, 28 Sep): repeatability against the consensus 0.16–0.50 px rms per camera and scape, except Taylorfjellet 0.71 px in both eyes with 0.54 px at the centre. Its refined focal lengths are 1.43 / 1.41 px (0.048 %) above the start in both eyes; the other scapes spread from −0.65 to +0.92 px, also in both eyes together. The label focal length does not change with the interpolation temperature (2958.48 ± 0.01 px in every scape), so the label model does not predict this; notebook 05 section 2a now tests it against the camera temperature. The nine-scape consensus differs from the five-scape one shipped in 0.30 by less than its scatter; it is not re-shipped until the temperature question is answered. Stereo: the consensus geometry gives a mean disparity bias of −0.09 px (worst −0.47), the label geometry −0.48 px (worst −1.94), i.e. −0.38 % of range at 10 m with the labels. Rig: the refined right-from-left rotation differs from the CAHV pairs by 5–24 mdeg in yaw and 12–23 mdeg in roll.

## 13. Navcam temperature bins and the sources of the residuals (v0p31, 28 Sep 2026; the nine 0.22.4 Navcam alignments of §12)

**Temperature bins** (`scripts/temperature_bins_experiment.py`, 10 °C bins, ≥ 8 images per bin; temperatures from 342 labels, the rest interpolated in SCLK). Reference: one camera per eye, fx, fy free, rest and rig held; then one camera per eye and bin.

| scape | images | T range °C | bins | cost | median px | rms px |
|---|---|---|---|---|---|---|
| Taylorfjellet | 245 | −32.0 … −6.4 | 4 | −1.04 % | 0.1967 → 0.1946 | 0.4173 → 0.4153 |
| Pearce Canyon | 160 | −49.8 … −13.5 | 3 | −1.95 % | 0.2140 → 0.2102 | 0.4414 → 0.4377 |
| Belva Crater | 80 | −38.4 … −18.7 | 2 | −0.35 % | 0.1424 → 0.1419 | 0.3291 → 0.3285 |
| South Arm | 104 | −47.9 … −19.6 | 2 | −0.44 % | 0.1592 → 0.1587 | 0.3252 → 0.3246 |
| Bell Island | 187 | −46.0 … −17.1 | 3 | −0.44 % | 0.1670 → 0.1660 | 0.3717 → 0.3740 |
| Rockytop | 136 | −35.5 … −13.8 | 2 | −0.32 % | 0.1682 → 0.1672 | 0.3628 → 0.3623 |
| Olifants | 210 | −34.8 … −10.7 | 3 | −1.15 % | 0.1718 → 0.1698 | 0.3843 → 0.3823 |
| Marble Mountain | 63 | −42.5 … −12.6 | 1 | 0 | — | — |
| Three Forks | 56 | −20.8 … −11.6 | 1 | 0 | — | — |

Fit f = a_scape + b T over the 21 bins (weights √obs): **NL fx +0.094 ± 0.006 px/°C (31.9 ± 2.1 ppm/°C), NR fx +0.086 ± 0.007 px/°C (29.2 ± 2.3 ppm/°C)**, residual 0.23 px; fy +0.067 ± 0.016 / +0.057 ± 0.017 px/°C, residual 0.6 px. Every scape with more than one bin has fx rising with temperature in both eyes (Taylorfjellet NL 2955.45 / 2956.44 / 2957.58 / 2957.94 px at −31.5 / −21.9 / −10.7 / −7.9 °C). Bins colder than −30 °C lie up to 0.5 px above the line. Across scapes (one camera per scape against its median temperature): +0.160 / +0.180 px/°C (r 0.90 / 0.93). The scape offsets a_scape at 0 °C still rise with scape temperature (NL 2957.5 Belva Crater … 2958.6 Taylorfjellet).

Consensus scatter (rms over the frame from the consensus, rotation removed; mean over both eyes and nine scapes): no thermal model 0.39 px (median 0.30); within-block slope 0.27 px (0.27); across-scape slope 0.27 px (0.25). Taylorfjellet NL 0.68 → 0.29 (within) / 0.24 (across); Three Forks stays the largest (NL 0.51 / 0.46, NR 0.63 / 0.44): its difference is at the centre (0.38 px), not in f. Decision: the consensus is formed at T0 with the within-block slope (`THERMAL_SOURCE = "auto"`), the direct measurement; the across-scape slope fits the same data and gains nothing measurable.

**Sources of the residuals** (native px, all observations of the final blocks; `share` = share of the summed squared residual):

- **The tail.** 1.2–4.5 % of the observations are above 1 px and carry 24–45 % of the squared residual; the top 1 % carry 16–24 %. Pearce Canyon (4.5 %, 45 %) and Taylorfjellet (3.9 %, 43 %) have the heaviest tails, Three Forks the lightest (1.2 %, 24 %).
- **Track length** is the strongest single factor: rms 0.17–0.22 px for two-view tracks (the stereo pair), 0.26–0.37 for 3–4 views, 0.35–0.49 for 5–8, 0.34–0.56 for 9–16, up to 0.63 above 16. Long tracks join exposures of different stations, sols and illumination; their keypoints are the least consistent. (Two-view tracks also fit more easily, having no redundancy beyond the pair.)
- **Resolution**: full-resolution frames have rms 0.31–0.58 native px against 0.29–0.41 for half resolution (Pearce 0.58, Taylorfjellet 0.49, Bell Island 0.48). In full-frame pixels the half-resolution frames are still the noisier, so σ in native px is not constant across scales.
- **Range**: points nearer than 3 m (ground in front of the rover, 28–54 % of observations) have the highest rms at six scapes (Pearce 0.49, Taylorfjellet 0.47); beyond 50 m it rises again (0.39–0.52). 6–25 m is the best-fitted band.
- **Local time**: frames before 10:00 or after 16:00 LMST have rms 0.43–0.80 but carry < 10 % of the observations; within 11–15 h the rms is flat.
- **Image radius**: flat to 0.85 of the half-diagonal; the outer corners (1–3 % of observations) are 10–25 % higher. The mean radial residual is ≤ 0.02 px in every radius band: no lens-model signature.
- **Eye and temperature**: NL and NR within 0.01 px of each other; no dependence on the camera temperature.
- **Images**: the worst images are whole stereo pairs (NL and NR of one exposure at 0.6–0.7 px: Bell Island sol 1463, Taylorfjellet 1622, Pearce 1196, Rockytop 509 and 471, Olifants 1807); the worst 5 % of images carry 10–17 % of the squared residual. Marble Mountain's three worst (sol 1979, 0.8–0.9 px) have < 220 observations.
- **Alignment (prior offsets)**: median camera-centre shift from the waypoint prior 0.01 m (Three Forks, Marble Mountain) to 0.27 m (Pearce Canyon), p95 0.04–0.41 m; attitude change median 0.10–0.47°, p95 0.28–0.68°. The largest shifts are at stations with one exposure (2 images: Pearce S054D0762 0.67 m, Marble S091D0606 0.56 m, Belva S039D0858 0.47 m), which the tie points hold only weakly, and at Rockytop S026D1004 (0.36 m median over 31 images, most likely an error of the waypoint itself).

Reading: the reprojection error is set by the matching of long, cross-station tracks and by a heavy tail of bad observations, not by the camera model (no radial signature, no eye or temperature dependence) or by the temperature (≤ 2 % of the cost). The next gains are in the tail: per-scale σ, a tighter final residual cut for long tracks, and the learned-matcher comparison of methods §13.

## 14. What the thermal model changes, and what the error model's parameters depend on (v0p31.1, 28 Sep 2026)

**Leave-one-scape-out test of the thermal slope** (nine scapes of §13; for each scape, the consensus and both slopes are formed from the other eight and used to predict its camera at its own temperature):

| thermal model | fx prediction error, rms / max over 18 cameras | frame difference from the predicted camera, mean / median / max rms px |
|---|---|---|
| none | 0.78 / 1.43 px | 0.44 / 0.34 / 0.89 |
| within-block slope (~30 ppm/°C) | 0.45 / 0.84 px | 0.31 / 0.29 / 0.66 |
| across-scape slope (~55–60 ppm/°C) | 0.33 / 0.57 px | 0.32 / 0.30 / 0.50 |

Both models cut the prediction error of a new scape's focal length by 40–60 %; the across-scape slope predicts fx better (it absorbs the part of the scape-to-scape spread that follows temperature but is not seen inside a block), the two are equal over the whole frame. For starting a block's camera from the consensus (`SfmProject.create`, held-intrinsics blocks) the across-scape slope is the better predictor; for the bins inside one block the within-block slope is the measured one.

**Geometric effect of the bins inside a block** (one camera per eye vs one per eye and 10 °C bin, same observations; `sparse` difference of all tie points matched by their first observation, after the similarity of the 3–25 m points):

| scape | bins | block rotation mdeg | scale ppm | shape change, median mm (ppm of range): < 3 m | 3–6 m | 6–12 m | 12–25 m | 25–50 m |
|---|---|---|---|---|---|---|---|---|
| South Arm | 2 | 0.6 | +12 | 0.13 (60) | 0.15 (38) | 0.26 (32) | 1.1 (69) | 2.2 (63) |
| Bell Island | 3 | 0.6 | −7 | 0.18 (77) | 0.34 (87) | 1.4 (161) | 3.9 (234) | 12 (367) |
| Rockytop | 2 | 1.0 | 0 | 0.28 (114) | 0.40 (104) | 1.7 (204) | 4.5 (289) | 27 (708) |
| Belva Crater | 2 | 1.7 | +52 | 0.58 (283) | 0.83 (202) | 1.2 (139) | 2.2 (134) | 8.5 (255) |
| Olifants | 3 | 3.3 | +62 | 0.67 (274) | 0.83 (215) | 1.9 (228) | 8.0 (496) | 42 (1248) |
| Taylorfjellet | 4 | 50.6 | −1 | 0.49 (196) | 0.66 (171) | 1.9 (223) | 6.8 (413) | 26 (731) |
| Pearce Canyon | 3 | 19.3 | −132 | 1.5 (715) | 2.7 (670) | 5.8 (728) | 28 (1615) | 74 (2165) |

The temperature bins reshape the block by 0.1–3 mm within 6 m, 1–28 mm at 12–25 m (60–1600 ppm of range), with the largest change at Pearce Canyon (a 13-image bin at −36 °C) and a 0.05° turn of the whole Taylorfjellet block. For scale: a single Navcam stereo pair at 20 m has a range precision of about 0.13–0.26 m (full- and half-resolution frames, ε ≈ 0.28 px), so the thermal change is a few per cent of it — but it is systematic and does not average down over many observations.

**Error-model inputs of fifteen Navcam scapes** (`scripts/error_analysis_batch.py`, notebook 05's measurements one alignment at a time; tracks ≥ 3; gate: power form, |ΔLMST| covariate, 5000 points, 50 bootstrap resamples; Rochette, Sid, Rockytop, Three Forks, Tuxedo Park, Airey Hill, Bunsen Peak and Rio Chiquito are the current alignments on D:\scapes\colmap, the other seven the 0.22.4 alignments of §12):

| site | images | stations | ε same / cross station px | same-station rate | cross-station rate at θ = 0 | θ½ deg | τ h | ρ first bin | cross-station fraction | sun el. deg | LMST p10–p90 h | band contrast σ 2 px | SIFT / Mpx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rochette | 37 | 3 | 0.285 / 0.295 | 0.56 | 0.85 | 3.7 | 6.7 | 0.105 | 0.03 | 52 | 2.8 | 0.0314 | 3356 |
| Sid | 81 | 8 | 0.248 / 0.324 | 0.56 | 0.60 | 10.7 | 4.0 | 0.096 | 0.31 | 49 | 2.9 | 0.0320 | 2543 |
| Rockytop | 136 | 9 | 0.285 / 0.320 | 0.46 | 0.52 | 5.2 | 9.0 | 0.071 | 0.08 | 33 | 2.6 | 0.0398 | 9786 |
| Three Forks | 56 | 4 | 0.188 / 0.237 | 0.56 | 0.65 | 14.8 | 1.4 | 0.023 | 0.43 | 47 | 2.2 | 0.0339 | 4977 |
| Belva Crater | 80 | 5 | 0.245 / 0.314 | 0.33 | 0.79 | 8.5 | 1.7 | 0.136 | 0.13 | 53 | 4.7 | 0.0234 | 1904 |
| Tuxedo Park | 100 | 8 | 0.235 / 0.262 | 0.62 | 0.80 | 4.3 | 2.0 | 0.039 | 0.22 | 40 | 1.7 | 0.0488 | 6479 |
| Airey Hill | 48 | 3 | 0.327 / 0.282 | 0.29 | 0.78 | 3.6 | 4.0 | 0.077 | 0.06 | 50 | 4.6 | 0.0250 | 2413 |
| Bunsen Peak | 108 | 6 | 0.297 / 0.381 | 0.44 | 0.61 | 8.1 | 4.9 | 0.038 | 0.35 | 55 | 3.0 | 0.0349 | 4625 |
| Pearce Canyon | 160 | 11 | 0.266 / 0.368 | 0.45 | 0.61 | 8.9 | 4.0 | 0.052 | 0.36 | 46 | 4.2 | 0.0353 | 9439 |
| Rio Chiquito | 38 | 2 | 0.231 / 0.240 | 0.55 | 0.78 | 5.6 | — | 0.063 | 0.07 | 71 | 1.1 | 0.0205 | 2140 |
| South Arm | 104 | 6 | 0.203 / 0.288 | 0.50 | 0.74 | 6.5 | 2.9 | 0.038 | 0.22 | 71 | 2.3 | 0.0188 | 1342 |
| Bell Island | 187 | 8 | 0.263 / 0.337 | 0.47 | 0.66 | 4.6 | 3.6 | 0.080 | 0.18 | 66 | 4.2 | 0.0247 | 3922 |
| Taylorfjellet | 245 | 11 | 0.294 / 0.385 | 0.44 | 0.58 | 4.2 | 4.4 | 0.068 | 0.18 | 61 | 3.5 | 0.0254 | 5276 |
| Olifants | 210 | 15 | 0.273 / 0.367 | 0.45 | 0.53 | 4.7 | 8.6 | 0.052 | 0.15 | 45 | 4.3 | 0.0335 | 8919 |
| Marble Mountain | 63 | 4 | 0.284 / 0.340 | 0.52 | 0.47 | 4.6 | — | 0.081 | 0.11 | 56 | 2.3 | 0.0302 | 4530 |
| pooled (Navcam-Navcam) | | | | 0.47 | 0.63 | 6.7 | 3.6 | | | | | | |

— τ unconstrained (Rio Chiquito: LMST range 1.1 h; Marble Mountain: no decline over 5.6 h). Pooled gate: θ̄ 8.0° (7.8–8.3), CV 0.25 (0.20–0.29), τ 3.6 h (3.5–3.7), A 1.36 relative / 0.63 absolute; the stretched exponential (β 0.95) beats the power form by ΔAIC 24, |ΔLMST| beats the sun-vector angle by ΔAIC 48 and no covariate by 88,000; χ²/dof 30 (the sites are not one population).

**Constrained (common to all sites):** the image precision, ε 0.27 px same-station and 0.32 px cross-station, between-site CV 0.14–0.15 — what spread there is follows the resolution mix (ε same-station vs the fraction of full-resolution frames, ρ +0.61), not the terrain; the cross-station rate at zero angle and zero ΔLMST, 0.63 (p10–p90 0.53–0.79, CV 0.17); the same-station rate, 0.47 (0.37–0.56); the illumination covariate and the gate form (pooled, well determined).

**Not constrained per site, or site-dependent:** θ½ 3.6–14.8° (CV 0.46) and the gate shape — the CV runs to its limit at 7 of 15 sites, so a single site cannot fix the shape; the pooled fit is what the model should take. θ½ follows the network, not the appearance: ρ +0.70 with the cross-station fraction (Three Forks and Sid, with the largest cross-station fractions, 0.43 and 0.31, are the widest). τ 1.4–9 h (CV 0.53, χ²/dof 47 against the bootstrap errors). The residual decorrelation ρ at the smallest separation, 0.02–0.14, non-zero at every site (se 0.002–0.007). The registration offsets from the waypoints, 0.03–0.42 m.

**Appearance.** Texture and contrast were measured on five left images per site (radiometric 16-bit and pipeline 8-bit, inside the terrain mask and below −3° elevation, i.e. the ground within ~35 m; `scripts/site_appearance.py`): rms contrast; band-pass contrast at the SIFT octaves σ 1, 2, 4, 8 px (DoG / local mean); power-spectrum slope; structure-tensor coherence; SIFT keypoint density and median response; repetitiveness (keypoints with a look-alike in the same image); shadow fraction; dynamic range. Between-site differences exceed the within-site scatter by 1.2–2.4×, but **single-image contrast is mostly illumination**: band contrast at σ 2 px against the median sun elevation ρ −0.84, rms contrast −0.74, shadow fraction −0.61 (per image, 1/sin(sun elevation) explains 15–43 % of the log contrast). After removing the sun-elevation term per image, the only appearance measure related to an error-model parameter is the SIFT keypoint strength: the cross-station rate at zero angle falls with the sun-corrected keypoint response (ρ −0.71, p < 0.01) and density (ρ −0.51) — rock- and pebble-strewn ground rich in small, high-contrast features (Rockytop, Pearce Canyon, Olifants) gives many keypoints that do not survive a change of station; sand and smooth regolith (South Arm, Belva Crater, Rio Chiquito) give fewer but more durable ones. ε, θ½ and ρ show no relation to any texture measure (|ρ| < 0.5). With 12 measures × 7 parameters tested at n = 15, a single ρ −0.71 is suggestive, not established.

Reading: the error model can take ε, the absolute match rate at zero angle, the gate form and the illumination covariate as common constants (pooled values); θ½ (or θ̄) and τ must be site parameters, and θ½ is set by the network geometry the site allows. Appearance, measured per image, is dominated by the sun; the measure of terrain texture that matters for matching is keypoint strength corrected for sun elevation, and it should be measured on the tie points themselves (the matched fraction of keypoints, and the descriptor distance of cross-station matches against Δθ and ΔLMST) rather than on whole images.

## 15. Rig stability, joint calibration and the lens model across fifteen Navcam blocks (v0p35, 29 Sep 2026)

Blocks: the fifteen of §14 (Rochette, Sid, Rockytop, Three Forks, Tuxedo Park, Airey Hill, Bunsen Peak, Rio Chiquito: current alignments; Belva Crater, Pearce Canyon, South Arm, Bell Island, Taylorfjellet, Olifants, Marble Mountain: 0.22.4). Camera temperatures: project records for seven blocks, 427 label samples (85 new, from the IMGs under `D:\data\m2020\datadrive`) interpolated in spacecraft clock for the other eight. `scripts/navcam_calibration_study.py all`; tables, tests and figure in `docs/results/v0p35/` (`navcam_calibration_report.py`). Rig angles in mdeg relative to the label (CAHV) rig; formal σ are rescaled by the variance factor (0.24–0.36).

**Rig per block** (at most 120,000 points per block; "pipeline rig": intrinsics and rig rotation free as in `reconstruct`; "common pp": both principal points held at the median over the blocks, NL (2591.17, 1944.22), NR (2574.16, 1950.15) px):

| block | sol | stations, span | T median °C | T left − right °C | yaw, pipeline rig | yaw, common pp (± σ) | pitch | roll | per-bin yaw (T °C: mdeg) | baseline free, m (± σ mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Rochette | 180 | 3, 11 m | -20.7 | +0.55 | -8.0 | -6.4 (± 0.28) | -2.9 | +20.2 | -21: -7.9; -17: -11.4 | 0.42426 (± 12.0) |
| Sid | 360 | 8, 21 m | -15.7 | -0.09 | -19.8 | -13.2 (± 0.25) | +0.9 | +22.4 | -23: -8.8; -12: -20.5; -8: -22.6 | 0.42438 (± 4.7) |
| Rockytop | 477 | 9, 22 m | -19.2 | +2.14 | -25.4 | -7.2 (± 0.16) | +0.9 | +24.7 | -21: -21.6; -17: -27.5 | 0.42439 (± 3.7) |
| Three Forks | 686 | 4, 5 m | -14.6 | +0.10 | -0.0 | -9.5 (± 0.16) | +1.2 | +20.6 | — | 0.42433 (± 24.5) |
| Belva Crater | 795 | 5, 14 m | -21.1 | +0.87 | -11.5 | -5.5 (± 0.15) | +2.3 | +17.4 | -37: -0.2; -21: -13.3 | 0.42453 (± 10.9) |
| Tuxedo Park | 904 | 8, 46 m | -16.1 | +0.56 | -10.3 | -10.3 (± 0.15) | +2.3 | +21.0 | — | 0.42439 (± 1.8) |
| Airey Hill | 990 | 3, 6 m | -13.1 | +1.32 | -1.4 | -15.1 (± 0.24) | +3.0 | +19.5 | — | 0.42427 (± 19.9) |
| Bunsen Peak | 1084 | 6, 9 m | -15.0 | +1.83 | +9.7 | -1.8 (± 0.11) | +3.0 | +15.7 | -38: +25.2; -26: +17.5; -13: +2.1; -7: -4.1 | 0.42436 (± 22.3) |
| Pearce Canyon | 1199 | 11, 28 m | -20.8 | +0.93 | +6.7 | +3.3 (± 0.10) | +3.2 | +12.3 | -36: +20.6; -21: +3.4; -15: -4.6 | 0.42445 (± 4.0) |
| Rio Chiquito | 1333 | 2, 4 m | -21.4 | +0.34 | -4.7 | -6.6 (± 0.20) | +4.4 | +10.7 | -23: +0.5; -16: -10.0 | 0.42427 (± 25.6) |
| South Arm | 1407 | 6, 8 m | -22.2 | +0.55 | -8.2 | +2.5 (± 0.09) | +3.6 | +12.2 | -47: +19.8; -22: -9.4 | 0.42433 (± 11.3) |
| Bell Island | 1461 | 8, 15 m | -22.4 | +1.17 | -7.6 | +1.9 (± 0.13) | +4.6 | +13.8 | -33: +8.5; -22: -8.9; -18: -10.6 | 0.42439 (± 4.2) |
| Taylorfjellet | 1627 | 11, 38 m | -11.0 | +1.07 | -7.6 | -11.0 (± 0.12) | +5.0 | +13.7 | -32: +22.7; -22: +6.2; -11: -7.1; -8: -11.9 | 0.42440 (± 2.0) |
| Olifants | 1784 | 15, 39 m | -18.1 | +2.46 | +5.0 | -0.5 (± 0.15) | +7.1 | +17.7 | -29: +17.1; -23: +10.4; -16: +1.7 | 0.42428 (± 2.2) |
| Marble Mountain | 1972 | 4, 21 m | -16.2 | +1.51 | -12.3 | -7.3 (± 0.18) | +8.6 | +14.0 | — | 0.42445 (± 12.6) |

- **The yaw trades with the principal points.** With the principal points free, the yaw's formal σ is 0.6–0.9 mdeg and the pipeline yaws scatter with sd 9.5 mdeg (Rockytop −25.4 against −7.2 with common principal points); with common principal points σ is 0.09–0.28 mdeg. Only the combination (disparity at infinity) is what stereo sees; the common-pp yaw carries it.
- **Between blocks the rig is not one rig.** Common-pp yaw: mean −5.8, scatter 5.8 mdeg against a median formal σ of 0.15 (Q = 24,300 on 14 dof, τ 6.0 mdeg); pitch scatter 2.8 (τ 2.7), roll 4.2 (τ 4.3).
- **Yaw follows the camera temperature.** Within blocks (30 bins, 11 blocks, one offset per block): **−1.13 ± 0.04 mdeg/°C** (p ≈ 10⁻²⁰³, residual τ 1.6 mdeg); every one of the eleven blocks with more than one bin has a negative slope (−0.8 to −1.6 mdeg/°C). Between blocks the common-pp yaw against the block's median temperature: −1.11 ± 0.31 mdeg/°C (ρ −0.75, p 0.001) — the same slope, so the between-block and within-block responses are one effect. The left–right temperature difference (−0.1 to +2.5 °C) explains nothing (p 0.34).
- **Pitch and roll drift with sol.** Pitch **+0.0050 ± 0.0004 mdeg/sol** (ρ +0.99, τ falls from 2.7 to 0.9 mdeg), roll **−0.0060 ± 0.0014 mdeg/sol** (p < 10⁻⁴); yaw +0.0045 ± 0.0019 mdeg/sol with the temperature term (p 0.02). Over sols 180–1972: 9 mdeg in pitch (0.46 px of vertical parallax at infinity), 11 in roll. No temperature term in pitch (within blocks +0.055 ± 0.025 mdeg/°C).
- **Network strength does not set the rig.** Stations, span and cross-station fraction: p > 0.2 for yaw and pitch. The yaw residual after temperature and sol still correlates with log(observations) (+32 mdeg per decade, p < 0.001; τ 3.8 → 2.0 mdeg) — unexplained, and it is why blocks keep a rig of their own.
- **The baseline is not measurable from these blocks.** With the right camera's position free, the baseline length is 0.42426–0.42453 m with σ 1.8–25.6 mm (set by the 1 m position priors); the label pairs agree to 10 µm. Hold it.

In pixels (f ≈ 2956 px): the yaw slope is **0.053 px of disparity at infinity per °C**; a block spanning 25 °C carries ±0.66 px about its median temperature — ±0.5 % of range at 10 m, ±1.1 % at 20 m — against ±0.05 % for the focal-length term. The drift over the mission is 0.46 px in vertical parallax and 0.42 px in disparity.

**Joint calibration** (15 blocks, 1,653 images, 15,000 points per block, 1,009,322 observations; T0 −19.1 °C; one NL, one NR camera, one rig; `joint_rational.json`, `joint_fisheye_t.json`):

- **Focal slope** (profile over 0–90 ppm/°C; cost 250,589 at 0, 244,734 at 30, 244,639 at 45, 246,425 at 60, 255,073 at 90): **38.6 ± 0.2 ppm/°C** (formal), between the within-block (30 ± 2) and across-block (55–60) estimates of §13–14. This is the slope for a camera shared by blocks, i.e. for start cameras; the formal σ ignores the between-block scatter (§13's two estimates are the realistic bracket).
- **Rig yaw slope** (profile at 38.6 ppm/°C; cost 247,802 at 0, 244,445 at −1.13): **−1.03 ± 0.01 mdeg/°C** (within-block −1.13). The rig term lowers the joint cost by 1.4 %, the focal term by 2.4 %.
- **Cameras at T0** (rational, σ rescaled): NL fx 2956.275 ± 0.019, fy 2955.935 ± 0.034, cx 2591.08 ± 0.04, cy 1943.50 ± 0.04 px; NR fx 2948.919 ± 0.018, fy 2948.543 ± 0.034, cx 2574.41 ± 0.04, cy 1949.53 ± 0.04 px; rig yaw −11.57 ± 0.47, pitch +4.97 ± 0.30, roll +15.35 ± 0.06 mdeg; baseline held at 0.42436 m. Against the averaged consensus of 28 Sep: NL 0.10 px rms (corners 0.16), NR 0.23 px (0.35), disparity at infinity +0.12 px.
- The fisheye + tangential joint fits the same observations with 0.59 % lower cost (242,984 against 244,415) at the same number of parameters.

**Leave one block out** (the joint calibration of the other 14 predicts the block; costs relative to the block's own calibration at the same thermal slopes; Δd∞, Δv∞: predicted minus own disparity and vertical parallax at infinity):

| block | cameras held: cost % (rational / fisheye+t) | fisheye+t − rational, cameras held, % | + rig held, rig(T) % | + rig held, one rig % | prediction vs own, rms px (rational / fisheye+t) | Δfx NL, NR px (rational) | Δd∞ px | Δv∞ px |
|---|---|---|---|---|---|---|---|---|
| Rochette | 0.35 / 0.41 | -0.52 | 9.11 | 9.65 | 0.253 / 0.195 | +0.19, +0.14 | -0.168 | +0.325 |
| Sid | 0.31 / 0.29 | -0.56 | 4.42 | 4.82 | 0.400 / 0.418 | +0.19, -0.05 | -0.034 | +0.177 |
| Rockytop | 0.67 / 0.67 | -0.52 | 6.69 | 7.35 | 0.198 / 0.223 | -0.16, +0.08 | -0.210 | +0.133 |
| Three Forks | 2.73 / 3.84 | -1.35 | 7.18 | 5.96 | 0.459 / 0.640 | -0.38, -0.67 | -0.092 | +0.108 |
| Belva Crater | 0.45 / 0.32 | -0.54 | 0.93 | 0.79 | 0.455 / 0.435 | +0.64, +0.85 | -0.171 | +0.048 |
| Tuxedo Park | 0.51 / 0.36 | -0.71 | 1.50 | 2.12 | 0.248 / 0.259 | -0.90, -0.65 | -0.130 | +0.039 |
| Airey Hill | 0.19 / 0.13 | -0.60 | 0.51 | 1.48 | 0.213 / 0.147 | +0.24, +0.17 | -0.147 | +0.014 |
| Bunsen Peak | 0.57 / 0.65 | -0.34 | 0.65 | 4.20 | 0.168 / 0.361 | +0.06, +0.12 | +0.073 | -0.017 |
| Pearce Canyon | 0.29 / 0.30 | -0.37 | 0.90 | 3.85 | 0.199 / 0.258 | -0.25, -0.21 | +0.084 | -0.022 |
| Rio Chiquito | 0.53 / 0.43 | -0.63 | 4.38 | 5.97 | 0.136 / 0.231 | -0.37, -0.07 | +0.004 | -0.076 |
| South Arm | 0.20 / 0.24 | -0.73 | 1.42 | 3.14 | 0.220 / 0.240 | -0.20, -0.09 | +0.150 | -0.026 |
| Bell Island | 0.42 / 0.41 | -0.58 | 1.22 | 3.03 | 0.218 / 0.199 | +0.16, +0.41 | +0.109 | -0.095 |
| Taylorfjellet | 0.50 / 0.44 | -0.61 | 1.89 | 4.05 | 0.144 / 0.157 | -0.27, -0.06 | +0.111 | -0.093 |
| Olifants | 0.21 / 0.21 | -0.45 | 2.31 | 3.47 | 0.158 / 0.111 | -0.36, +0.06 | +0.162 | -0.206 |
| Marble Mountain | 0.57 / 0.48 | -0.53 | 7.67 | 8.10 | 0.309 / 0.374 | +0.41, +0.69 | -0.025 | -0.271 |

- **The shared cameras transfer.** With the predicted cameras held (rig free), a new block's cost rises by 0.57 % on average (median 0.45 %; Three Forks 2.7 %), its median residual by 0.001 px (0.194 against 0.193 px) and its corner median by 0.005 px. The prediction is 0.25 px rms from the block's own calibration over the frame (median 0.23, corners 0.46), fx within 0.39 px rms.
- **The rig does not transfer as well.** Holding the predicted rig as well raises the cost by 3.4 % on average (median 1.9 %); without the rig's temperature term 4.5 % (median 4.1 %); the temperature term helps in 13 of 15 blocks (not at Three Forks and Belva Crater). The early and late blocks cost most (Rochette 9.1 %, Marble Mountain 7.7 %): the drift. As disparity and vertical parallax at infinity, the predicted rig is 0.125 and 0.144 px rms from the block's own; adding the drift (fitted without the block) brings them to 0.106 and 0.041 px, the roll from 4.5 to 3.4 mdeg.
- **Lens model.** With the predicted cameras held, the fisheye + tangential model fits every one of the 15 held-out blocks better than the rational model: cost −0.60 % (median −0.56 %, Wilcoxon p 6·10⁻⁵), corner median 0.239 against 0.259 px (−8 %), median residual 0.193 against 0.194 px. Its transfer penalty is the same (0.61 against 0.57 %), and its prediction is as far from the block's own calibration (0.28 against 0.25 px rms, fisheye closer in 5 of 15, p 0.23). The rig results are the same for both.

Reading: the Navcam intrinsics are stable — one camera per eye with the 38.6 ppm/°C focal term predicts a new block to within 0.5 % in cost and a quarter of a pixel over the frame. The rig is not: its yaw turns by −1.0 mdeg/°C (0.05 px of disparity per °C, the largest thermal effect on stereo range, ten times the focal term), its pitch and roll drift over the mission, and a block-to-block residual of 2–4 mdeg (0.1–0.2 px) remains. Decision: a temperature-dependent rig with a drift for the start and for held rigs (weak networks, the thermal stage's bins), and the rig rotation refined in every block whose network supports it (`refine_rig="auto"`). The lens model is fisheye + tangential (better on unseen blocks, especially the corners; same transfer); the rational model stays available.


## 16. Is the rig's drift robust? Six more blocks, the 24 sites, and the resolutions (v0p35.1, 29 Sep 2026)

**Sites.** Notebook 03's `SITES` now lists 23 sites (plus `threeforks_large` on disk). State on `D:\scapes\colmap` (29 Sep, 18:00):

| site | sols | state | flag |
|---|---|---|---|
| butler_landing | 1–14 (images 9–14) | 0.35 alignment, health WARN, 76 images, 7 stations | **weak network by span** (4.9 m < 5 m): notebook 03 held cameras and rig, residuals 0.35 px (2.5 × the others), edge/centre 1.8. 67 % full resolution; labels without CAHV temperature (11 label samples). Rig study: used. |
| van_zyl | 49–71 | 0.35, PASS, 116 images, 9 stations | 64 % quarter resolution, 20 left-only frames, early labels (26 label samples). Used. |
| rochette | 179–190 | v0p35 block | early label rig (below) |
| seitah_north | 238–279 | 0.35, **FAIL** | waypoint layout 8.6 % off in scale (1.9 m over 10 stations; station shift 3.1 m, block rotation 0.42°): a pose-prior problem; project recorded as `seitah_colmap` (folder renamed after the run); sols 237–242 without CAHV temperature (6 samples). Used. |
| sid, rockytop | | v0p35 blocks | — |
| whale_mountain | 606–610 | 0.35, WARN, 31 images, **2 stations** | weak network (cameras and rig held → the notebook 03 `IndexError`, fixed); nearby waypoints reach sols 592–639, `KEEP_ONLY_REMAINING` kept 31 of 78. Shown, **not fitted**. |
| threeforks_south | 652–683 | 0.30-era alignment, 306 images, 28 stations | **contains all 56 Three Forks images** (sols 413–693): a replicate, not an independent block. Fitted; fits repeated without it. |
| pico_turquino | 1307–1322 | 0.35, PASS, 81 images, 7 stations | — (added to `SITES` as 1307–1322, the sols of its images) |
| origny | 1775–1813 | the v0p35 "Olifants" block, in `olifants_colmap` | **rename the folder** to `origny_colmap`; notebook 03 now warns when a WORK folder holds no image of the site's sol range |
| olifants | 1880–1889 | 0.35 run found **no PDS products** (datadrive ends at sol 1819, `D:\data\m2020` starts at 1900); its manifest landed in the Origny folder | **needs data** |
| groloy | 1922–1934 | products on disk, **not run** | — |
| overlook_mountain | ? | folder appeared during this work, not in `SITES`, not reviewed | — |

In this section "Olifants" is still the v0p35 block of sols 1775–1813 (the site **origny**).

**One label reference.** `rig_study` measures the rig against each block's label (CAHV) rig. These are identical after sol ~250; Butler Landing, Van Zyl, Rochette and Seitah North carry an earlier label calibration that differs by 0.23 mdeg in pitch and 0.58 mdeg in roll. From 0.35.1 all angles are referred to one label rig (`navcal.common_reference`); for the 15 v0p35 blocks this moves only Rochette (rates change < 0.15 mdeg / 1000 sol).

**New blocks** (`navcam_calibration_study.py rig --common-pp <v0p35>`; blocks with thermal bins read from `sparse/cahv_ba_single`, fisheye-solved blocks refitted to the rational model, 0.11–0.20 px rms). Common principal points, one label reference, mdeg, σ rescaled:

| block | sols (median) | stations | images (full-res %) | T °C | yaw | pitch | roll | σ yaw / pitch / roll |
|---|---|---|---|---|---|---|---|---|
| **Butler Landing** | 9–14 (12) | 7 | 76 (67) | -20.9 | -8.45 | -6.65 | +24.77 | 0.43 / 0.08 / 0.12 |
| **Van Zyl** | 49–71 (63) | 9 | 116 (11) | -21.2 | +4.57 | -8.15 | +18.89 | 0.19 / 0.04 / 0.07 |
| Rochette | 178–341 (180) | 3 | 37 (19) | -20.7 | -6.44 | -2.68 | +20.76 | 0.28 / 0.05 / 0.09 |
| **Seitah North** | 237–278 (242) | 10 | 81 (10) | -16.9 | -11.85 | -1.50 | +22.96 | 0.15 / 0.04 / 0.07 |
| **Whale Mountain** (not fitted) | 606–609 (609) | 2 | 31 (6) | -18.2 | -7.75 | +0.94 | +23.81 | 0.20 / 0.05 / 0.09 |
| **Three Forks South** | 413–693 (670) | 28 | 306 (40) | -14.5 | -10.84 | +1.12 | +17.23 | 0.11 / 0.03 / 0.05 |
| Three Forks | 433–693 (686) | 4 | 56 (12) | -14.6 | -9.51 | +1.20 | +20.61 | 0.16 / 0.04 / 0.06 |
| **Pico Turquino** | 1307–1322 (1309) | 7 | 81 (10) | -17.9 | +3.05 | +3.79 | +12.09 | 0.16 / 0.05 / 0.08 |

(the other v0p35 blocks as §15). Table and figure: `docs/results/v0p35p1/rig_drift_robustness.json`, `navcam_rig_drift.png`.

**The replicate.** Three Forks South contains Three Forks at the same epoch and temperature: the rigs agree to 0.08 mdeg in pitch (0.004 px of vertical parallax), 1.3 mdeg in yaw (0.07 px of disparity) and 3.4 mdeg in roll. Pitch is reproducible to its formal σ; yaw and roll carry a block-dependent part of 1–3 mdeg — the floor under their between-block τ.

**Out of sample.** The v0p35 drift (15 blocks, linear in sol with temperature) predicting the new blocks (residual mdeg, z with σ² = fit ⊕ τ ⊕ block; in brackets the residual against one fixed rig):

| block | pitch | yaw | roll |
|---|---|---|---|
| Butler Landing (12) | **-4.39, z -4.4** (-9.8) | -1.36, z -0.3 (-2.7) | +2.26, z +0.7 (+7.7) |
| Van Zyl (63) | **-6.13, z -6.2** (-11.3) | +11.08, z +2.5 (+10.4) | -3.20, z -1.0 (+1.8) |
| Seitah North (242) | -0.58, z -0.6 (-4.7) | -1.23, z -0.3 (-6.1) | +0.35, z +0.1 (+5.9) |
| Whale Mountain (609) | +0.12, z +0.1 (-2.2) | -0.29, z -0.1 (-2.0) | +3.95, z +1.4 (+6.7) |
| Three Forks South (670) | -0.18, z -0.2 (-2.1) | +0.65, z +0.2 (-5.1) | -3.68, z -1.2 (+0.1) |
| Pico Turquino (1309) | -0.48, z -0.6 (+0.6) | +7.72, z +2.0 (+8.8) | -3.59, z -1.3 (-5.0) |

Every block after sol 240 is predicted within 2 σ on every angle, and the drift removes most of the offset from a fixed rig. The two blocks before sol 100 — Butler Landing, mostly full resolution, and Van Zyl, mostly quarter resolution, so not one block's artefact — have their pitch 4–6 mdeg (0.2–0.3 px of vertical parallax) below the linear line.

**Refit** (`rig_drift`, weights 1/(σ² + τ²); mdeg / 1000 sol):

| fit | pitch | yaw (p) | roll | τ pitch / yaw / roll |
|---|---|---|---|---|
| v0p35, 15 blocks | +4.92 ± 0.43 | +4.51 ± 1.93 (0.02) | -6.14 ± 1.39 | 0.84 / 3.80 / 2.74 |
| **20 blocks** (without Whale Mountain) | +6.29 ± 0.62 | +3.89 ± 1.60 (0.015) | -6.10 ± 1.09 | 1.56 / 4.02 / 2.76 |
| 19 without Butler Landing | +5.93 ± 0.65 | +3.38 ± 1.70 (0.05) | -5.68 ± 1.16 | 1.53 / 4.03 / 2.75 |
| 18 without Butler Landing and Van Zyl | +5.00 ± 0.34 | +5.12 ± 1.69 (0.003) | -6.14 ± 1.27 | 0.75 / 3.69 / 2.76 |
| 19 without Three Forks South | +6.32 ± 0.66 | +3.90 ± 1.70 (0.02) | -6.29 ± 1.09 | 1.65 / 4.23 / 2.71 |
| 19 without Seitah North | +6.40 ± 0.66 | +3.59 ± 1.68 (0.03) | -5.99 ± 1.17 | 1.60 / 4.05 / 2.82 |
| 21 with Whale Mountain | +6.24 ± 0.60 | +3.95 ± 1.56 (0.01) | -6.30 ± 1.12 | 1.54 / 3.97 / 2.85 |
| **20, pitch hinge at sol 300** | **+4.46 ± 0.41** after, **+25.8 ± 2.8** before | +3.89 ± 1.60 | -6.10 ± 1.09 | **0.79** / 4.02 / 2.76 |

Leave-one-block-out rates (20 blocks): pitch +5.7…+6.6, yaw +3.4…+5.3, roll −7.0…−5.7 — every sign stable. Leave-one-block-out **prediction** rms (mdeg):

| model | pitch | yaw | roll |
|---|---|---|---|
| one rig | 4.19 | 6.30 | 4.54 |
| temperature | 4.22 | 5.05 | 4.76 |
| temperature + sol | 1.78 | **4.72** | **2.96** |
| + pitch hinge at sol 300 | **1.10** | (5.44) | (3.31) |

- **Pitch drift: robust, faster early.** It holds out of sample after sol 240, its jackknife range is narrow, and it cuts the prediction error from 4.2 to 1.8 mdeg; with a faster rate before sol ~300 to 1.1 mdeg (0.06 px of vertical parallax). The early part is set by four blocks (Butler Landing −6.7, Van Zyl −8.2, Rochette −2.7, Seitah North −1.5 mdeg): the pitch rose by ~9 mdeg in the first 300 sols, then +4.5 mdeg / 1000 sol to Marble Mountain (+8.6). A hinge at sol 250–325 fits best (AIC 54–55 against 80 for the line, τ 1.56 → 0.79); an exponential settling from landing fits worse (AIC ≥ 60) because Butler Landing (sol 12) lies above Van Zyl (sol 63) — the shape before sol 100 is not resolved. Over the mission the pitch changes by ~16 mdeg, 0.8 px of vertical parallax at infinity.
- **Roll drift: holds.** −6.1 ± 1.1 mdeg / 1000 sol, sign stable, prediction 4.5 → 3.0 mdeg; τ stays 2.8 mdeg, which the Three Forks replicate (3.4 mdeg) shows is mostly estimation. Roll moves the image edges only (1 mdeg = 0.045 px at 2560 px from the centre).
- **Yaw drift: marginal, kept.** +3.9 ± 1.6 mdeg / 1000 sol with 20 blocks (p 0.015; it was +2.9, p 0.08, with the first 18 and rises to +5.1 without the two earliest). It improves the prediction slightly (5.05 → 4.72 mdeg) and its sign is stable. It is the weakest term; Pico Turquino (+7.7) and Van Zyl (+11.1) are the largest yaw residuals.
- **Yaw follows temperature: robust.** Within blocks −1.05 ± 0.03 mdeg/°C (40 bins in 15 blocks; v0p35 −1.13 ± 0.04), between blocks −1.17 ± 0.31 (−1.28 with sol); the joint slope −1.03 stands. The yaw residual still correlates with log(observations) (p 0.03). Pitch and roll within-block slopes +0.08 ± 0.02 and +0.26 ± 0.07 mdeg/°C.

**Decision.** The rig file's drift is `navcal.rig_drift_model` on the 20 blocks: pitch +4.46 mdeg / 1000 sol after sol 300 and +25.8 before (hinge), yaw +3.89, roll −6.10; `sol0` and the hinge reference are the means over the 15 joint blocks, so the joint rig is unchanged. Blocks with a strong network still refine the rig rotation (`refine_rig="auto"`); the drift matters for the start and for held rigs (Butler Landing, Whale Mountain, the thermal bins). Written to `D:\scapes\colmap\camera_analysis\navcal_v0p35p1\navcam_joint` (and `_rational`); cameras as v0p35. A block between sols 20 and 170 would pin down the early shape.

**Resolutions: is there a pixel offset between full, half and quarter resolution?** MPPP maps native keypoints to full-frame pixels by a pure scale in corner-origin pixels, x_full = x / s (pixel centres: (x + 0.5)/s − 0.5); sub-frames are placed with the label's FIRST_LINE(_SAMPLE). This is exact for binning that starts at the detector corner. Sub-sampling instead would put half- and quarter-resolution images 0.5 and 1.5 full-res px off, in x and y alike and in both eyes.

- *Flight calibration.* The label CAHVORE of 1,760 Navcam products (all manifests at hand), mapped to full-frame pixels the same way, per eye and sol against the half-resolution products: full − half resolution Δhc +0.002 / +0.001 px, Δvc −0.001 / 0.000 px (NL / NR, 80 sols; interquartile ±0.02 px); quarter − half −0.03 / +0.02 px (48 / 33 sols); focal lengths equal to 0.001 px. Sub-frames (1280 × 224, 5120 × 960, 3840 × 2880, placed with FIRST_LINE(_SAMPLE)) agree with the full frames of their sol to a median 0.005 px. JPL's models of the three resolutions are one camera under exactly MPPP's convention (`navcal_report.label_scale_consistency`, `label_scale_consistency.json`).
- *Images.* Each block re-adjusted with one camera per eye and scale (`navcal.scale_offsets`: principal point free — "pp" — or principal point and focal length — "ppf"; everything else and the rig held; covariance). The method recovers an injected (1.0, −0.5) px shift to 0.15 px on the synthetic block (test). Over 20 blocks (`scale_offsets.json`), offsets from the half-resolution camera, full-res px:

| scale pair | cameras (blocks) | pp: weighted mean x, y | ppf: weighted mean x, y | scatter over blocks x, y (ppf) | median formal σ |
|---|---|---|---|---|---|
| full − half | 26–28 (15–16) | +0.36, −0.10 | −0.17, −0.33 | 0.50, 0.93 | 0.07–0.11 |
| quarter − half | 15 (10) | +0.06, −0.06 | +0.02, +0.07 | 0.54, 0.48 | 0.14–0.17 |

  Per block the offset of a scale subset reaches ±1 px (Belva Crater full-res +1.2, +0.9; Bunsen Peak −1.1, −1.6), ten times its formal σ, but it changes sign from block to block, is not equal in x and y, differs between the pp and ppf fits (the principal point trades with the focal length and with the attitude of those frames), and — decisive — is **the same in both eyes**: left minus right, the offsets average −0.05 / 0.00 px (x / y) with rms 0.13–0.14 / 0.05–0.06 px over 18–19 block/scale pairs. A common shift of both eyes is an attitude change of those frames (1 px = 0.02°), which the pose absorbs; it does not enter stereo. The quarter − half offset, where a convention error would show at 1.0 px, is +0.02 ± 0.14 px.

Reading: there is **no systematic pixel offset between the Navcam resolutions** at the 0.1–0.2 px level: MPPP's mapping agrees with the flight calibration to 0.03 px, and the blocks show no common offset. The block-specific common-mode shifts of a scale subset (0.5–1 px) are frame-attitude effects of particular image sequences (different pointing and terrain), not a camera property; what stereo sees (left − right) is ≤ 0.14 px rms and averages zero. No correction is needed; `scripts/navcam_calibration_study.py scale` repeats the check on new blocks.

## 17. Waypoint priors of short stops (v0p35.2, 29 Sep 2026)

Seitah North failed its health check on the waypoint layout: the layout was off by 8.6 % in scale, 1.9 m over 10 stations. Its station shifts, refined minus waypoint, are:

| station | images | shift |
|---|---|---|
| S007D2246, S007D2326, S007D2440, S008D0000, S008D0064 | 10–24 | 0.06–0.63 m |
| S007D2280, S007D2298 (sol 238), S007D2378, S007D2406 (sol 239) | 2 | **3.0–3.1 m** |
| S008D0012 (sol 278) | 2 | 0.33 m |

The two-image stations are single stereo pairs taken during drives. The four on sols 238–239 carry waypoints about 3 m off. The tie points had already moved them to where they belong. Their 1 m position priors could not hold them, but those four priors made the waypoint layout look 8.6 % too large.

Notebook 03's `LOCALIZE_MIN_IMAGES = 4` leaves out the position prior of every station with fewer than 4 images. Before triangulation these stations are placed on the block from their tie points, and their attitude priors stay. On the solved Seitah North block (one camera per eye, re-adjusted with and without the five two-image stations' priors):

- the prior scale error drops to 1.5 % (0.27 m, pass) over the five stations that keep a prior;
- the waypoint layout is still turned by 2.3° against the solution;
- the station shifts are the same to 1 mm, so the solution itself was not being bent;
- the health verdict goes from FAIL to WARN; `block_rotation_deg` 0.42° remains, mostly azimuth (−0.38°).

On the synthetic block (tests), a two-image station 1.8 m off its prior dragged the whole block by 14 cm with its prior, and 4.5 cm (the noise) without it. Where tie points are fewer than at Seitah North, dropping these priors matters for the geometry, not only for the check.

## 18. Closing the Navcam analysis; Mastcam-Z focus states (v0p40, 29 Sep 2026)

**Consensus.** The joint calibration used 23 blocks, from Butler Landing (sol 9) to Marble Mountain (sol 1979), with 8,000 points per block. The focal slope is 38.1 ± 0.2 ppm/°C and the rig yaw slope −1.018 ± 0.007 mdeg/°C. Cameras and rig are written at −20 °C. Against v0p35.1 at the same temperature the cameras differ by 0.5–0.6 px rms, most of it a common +0.5 px shift of cx that trades with a −3.6 mdeg rig yaw. Ten blocks were still aligned with the rational cameras (older runs). A joint of only the 13 fisheye-aligned blocks differs by 0.3–0.4 px, so the consensus should be regenerated once those sites are re-run.

**Rig epochs.** Three blocks hold images from two parts of the mission, brought in by the nearby waypoints: Sid (sols 91–101 and 360–371), Three Forks South (413–433 and 652–693) and South Arm (1359–1360 and 1407–1411). Rochette's sol-341 epoch has only 3 frames and is not split. Solving one rig per epoch shows:

| block, epoch | T (°C) | yaw | pitch | roll (mdeg) |
|---|---|---|---|---|
| Sid 91–101 | −22.2 | −1.6 | −3.8 | +26.7 |
| Sid 360–371 | −8.6 | −15.9 | +1.2 | +21.8 |
| Three Forks South 413–433 | −17.1 | −10.1 | −0.8 | +17.1 |
| Three Forks South 652–693 | −14.2 | −11.1 | +1.5 | +17.1 |
| South Arm 1359–1360 | −22.3 | +1.3 | +3.5 | +11.0 |
| South Arm 1407–1411 | −21.6 | +3.1 | +3.8 | +13.9 |

Sid's pitch step (+5.0 mdeg) is what the hinge drift predicts (+5.5), and its yaw step (−14.3) is the thermal term for 13.5 °C (−14). As one block at sol 360, Sid had mixed the two sides of the knee. With the epochs the drift barely changes: pitch late rate 4.81 → 4.91, early 25.2 → 22.5 mdeg/1000 sol.

**Site flags.**

| action | sites | reason |
|---|---|---|
| re-run (rational Navcam model) | Rochette, Belva Crater, Tuxedo Park, Airey Hill, Bunsen Peak, Rio Chiquito, South Arm, Marble Mountain | aligned with the old rational cameras |
| re-running | Rockytop, Pearce Canyon | |
| re-run (rig held on a strong network) | Sid, Origny | `NAVCAM_RIG_REFINE = "refine"` bug |
| remove candidate | Whale Mountain | 2 stations |
| keep, cameras and rig held | Butler Landing, Groloy, Overlook Mountain | weak networks |
| check sol ranges | threeforks_south / threeforks_north, pico_turquino | overlap through the nearby waypoints (sols 413–693 in both); Pico Turquino holds sols to 1322 |

**Mastcam-Z focus states.** The per-image test on two older Nav+Zcam blocks, Airey Hill (152 Zcam images) and Three Forks (342), fitted each image's own focal length with its centre held near its Navcam-shifted label position. Almost all focus groups sit at 1.009–1.013 × the label f, which is the backlash state. A few one- and two-image bins at 0.96–0.99 are failed bins rather than a second state, so they are marked implausible. Those blocks were solved with bins started in the backlash state, and a regular-state image in such a bin shows up only if its centre cannot absorb the difference. The first new Zcam runs will show how many regular groups there are.
