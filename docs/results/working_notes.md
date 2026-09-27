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

See `docs/results/sites.md` (table of the scapes used, sol ranges, images, stations, span, LMST spread) and `sites_lmst.png`.
