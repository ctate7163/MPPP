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
