# Camera models and calibration

`mppp.calib` defines one frozen model form per camera family and ships the promoted models in
`src/mppp/data/cmods/`. `mppp cmods list` prints the models in use; `MPPP_CMODS` points at another folder, whose
missing files fall back to the package copies.

## Model forms

**Navcam** (eyes NL, NR). COLMAP `THIN_PRISM_FISHEYE` at the full 5120 x 3840 frame, with sx1 = sy1 = 0 held. The
focal lengths, the principal point and k1 are linear in the camera temperature about a reference temperature T0;
fx, fy, cx, cy carry offsets for the images before the early-mission sol. NL is the rig reference. The stereo rig
(NR from NL) has one yaw for every temperature and a rotation drift linear in sol, with a hinge in the pitch rate.

**Hazcam** (front FL, FR and rear RL, RR). The same form as the Navcam: fx, fy and k1 linear in temperature, an
early-mission offset on fx, fy and a rig rotation drifting linearly with sol (`mppp calib hazcam`, below; no
`M2020_FL.json` ... promoted yet). Without a model each image
starts from the `THIN_PRISM_FISHEYE` fitted by least squares to its own label CAHVORE over the whole frame
(`mppp.calib.labelfit`, 0.93), in the label model's own camera frame so the pose prior is unchanged. The front
Hazcams are central (E about 1e-7 m): 0.13-0.19 px rms. The rear Hazcams carry entrance-pupil terms of 3-9 mm; a
central camera cannot follow them, so the fit uses points at 2, 5 and 20 m and is good to 2-4 px rms (full-frame
pixels; the rear products are usually quarter resolution, so 0.5-1 px in the image). In a block each eye starts from
the median of its images' fits and refines fx, fy, k1, k2 (`hazcam_intrinsics = "radial"`; "focal", "all",
"hold"); the rig rotation of each pair is refined (`hazcam_rig_refine = "rotation"`, 0.93.1; a pair whose eyes are
both weak stays at the CAHV rig; or "hold"). The front
baseline is 0.25 m, the rear 0.93 m. The generic start model of 0.92 (f = 2130 px, k1 = -0.25, k2 = +0.08) was
330 px rms from the flight calibration and is replaced (f = 2148 px, k1 = +0.0145, k2 = -0.0083; used only
without a label model).

**Mastcam-Z** (zooms 34, 48, 63, 79, 110 mm). One calibration per zoom, `FULL_OPENCV` at 1648 x 1200, a function of
the focus count and the focus-motor backlash state only: fx and fy are lines in focus count per eye and state, k1
and k2 are constant per eye, and there is no temperature term. 34 mm is calibrated; the other zooms start from the
label CAHVOR of each image.

The two backlash states are "backlash" (the default, about 0.95 % above the label focal length) and "regular" (low
backlash, near the label; the images on the low-backlash list and those a block's focal fits find). The regular
state scales the focal length only: its fx and fy are the backlash lines divided by one ratio per eye (checked to lie
in 1.004–1.016), and it has its own principal point. **110 mm has no state** ("single"): the low-backlash lists never
apply to it, no regular ratio is derived or checked for it, and its model has the one set of lines. Its frames tie
to the other zooms through the block's points without a state of their own, so they anchor the other zooms'
per-state corrections.

## The calibration loop

The block alignments refine the cameras (see [Pipeline](pipeline.md)). The joint adjustments below combine many
finished blocks into one candidate model, check it against the model in use, and promote it on request. A promotion
changes the run key, so the next batch run realigns every block with the new cameras.

```text
finished blocks  ->  mppp calib navcam   ->  Navcam candidate        ->  check  ->  promote
                 ->  mppp calib zcam     ->  Mastcam-Z focus model   ->  check  ->  promote
                 ->  mppp calib nz       ->  report only (never promoted)
```

The blocks come from a site group of the site definitions (`--group`, default `ecam_consensus` for Navcam and
`zcam34_consensus` for Mastcam-Z) under the scapes root, or are listed with `--blocks NAME=WORK ...`. Each joint
needs at least two finished blocks. Results go to `<root>/camera_analysis/<name>_<date>/` unless `--out` is given.

### Navcam tied joint: `mppp calib navcam`

Module [`mppp.calib.tied`](api.md#mppp.calib.tied). Every term of the Navcam form is a parameter of one bundle
adjustment over the finished Navcam blocks:

- each eye gets one COLMAP camera per (temperature bin, era), where the era is before or after the early-mission
  sol. The cameras are tied by constraint residuals to global parameters:
  `fx = G.fx + S.fx (T - T0) + early * E.fx` (likewise fy, cx, cy), `k1 = G.k1 + S.k1 (T - T0)`, and one set of
  k2, p1, p2, k3, k4 for all bins;
- each block gets one stereo rig per (rig epoch, bin, era). The rig rotations are tied to one global rig and its
  drift in sol, with the early-mission hinge in pitch. The yaw of the global rig and the baseline are held.

The global parameters and their covariance go into a candidate folder (`navcam_joint/`). A diagnostic pass then
holds the global law and frees one camera per (block, eye, bin, era) and one rig per (block, epoch), to show the
per-block, per-bin and per-sol values about the global solution. The candidate is checked against the models in use
(`promotion_check.json`): convergence, residuals, camera and rig changes, thermal slopes, the early-mission fit and
leave-one-block-out limits.

```bash
mppp calib navcam                          # blocks of the ecam_consensus group under MPPP_SCAPES
mppp calib navcam --promote-if-ok --note "tied joint of the consensus blocks"
```

### Mastcam-Z joint: `mppp calib zcam`

Module [`mppp.calib.ztied`](api.md#mppp.calib.ztied). One bundle adjustment over the finished Navcam + Mastcam-Z
blocks of one zoom (`--zoom`, default 34):

- **Navcam held** at the promoted law (camera per eye, temperature bin and era; rig on its sol trend). The Navcam
  fixes the scale and the directions that the Mastcam-Z is fitted against. Frame poses are free, with the same
  priors as the block runs.
- **Mastcam-Z fitted**: every block's focus-bin camera is tied to the global focus law of its eye (zcam-v2 form):
  fx and fy lines in focus count per backlash state with one slope per eye; k1, k2, p1, p2 one value per eye; one
  Mastcam-Z rig for all blocks, rotation refined, baseline held.
- **Diagnostic pass**: the global law held, each block's bins and rig free.

The result is a candidate focus model, checked against the one in use (`report.md`). `--no-k-tests` skips the
k1/k2 likelihood-ratio tests.

**Backlash states: each its own camera, the pose shared** (0.93.2). The two focus states of an eye are two
cameras: the regular (low-backlash) state has its own focal length (the regular ratio) and its own principal point,
`regular_pp_offset_px` = (d_cx_regular, d_cy_regular) from the backlash state's, stored in the model as the regular
state's cx, cy lines. Both states use the same exposure pose and the same Mastcam-Z rig, so the state cannot move
the frame rotation or the rig. The state is per eye: an exposure can have ZL regular and ZR backlash (Three Forks:
96 ZL images regular, their ZR partners backlash), and those mixed exposures make the offset observable - the
backlash eye and the rig fix the pose. The joint fits the offset per eye when it has at least 5 such exposures
(`ztied.MIN_MIXED_EXPOSURES`), else holds it; the blocks start the regular cameras at it and hold it. Before 0.93.2
both states shared pp0 and a boresight change of the regular state went into the rig: in the 22 Oct Z34 joint
chal_rocks, the one block with regular images (8 % of the observations), was the outlier of the per-block rigs
(pitch -29, yaw -12 mdeg; the others within +-7) and anchored the rig's yaw trend. In a synthetic joint, a (+6, -4) px
regular offset in a third of the ZL images moved the rig by (-18, -22, +9) mdeg with the old form and by < 0.1
mdeg with the offset fitted.

**Fixed terms per zoom** (0.93.1, `mppp.calib.forms.ZCAM_FIXED_TERMS`): at 34 mm k2 is held at 0.5 (ZL) and 0.6
(ZR), with p1 and p2 free; at 48 mm k2, p1 and p2 are free; at 63, 79 and 110 mm k2 = p1 = p2 = 0. The block runs
hold these values in their bin cameras and the joint holds the global terms at them; the likelihood-ratio tests then
include "the fixed terms free" against the fixed values. The per-image prior camera keeps the promoted model's own
values (its k1 was fitted with its k2). The 22 Oct Z34 joint gave k2 = 0.484 +- 0.010 (ZL) and 0.615 +- 0.009 (ZR)
with k1 and k2 correlated 0.98, so fixing k2 costs little.

**A zoom without a model** starts from the blocks: per eye the median refined bin camera with one focus slope, and
the median Mastcam-Z rig (`ztied.start_model_from_blocks`). A first calibration may be promoted from 2 blocks
(`min_blocks_start`); later ones need 3. The joint accepts blocks aligned with an older Mastcam-Z model: the check
is that they were aligned with the Navcam models in use. `--group` takes comma-separated groups (their union),
filtered to the sites whose `zcam` list has the zoom.

```bash
mppp calib zcam --zoom 34
mppp calib zcam --zoom 34 --promote-if-ok
mppp calib zcam --zoom 63 --group zcam_consensus --promote-if-ok
```

`scripts/windows/run_sites_zcam.bat` aligns the `zcam_consensus` blocks (every zoom of each site) and then runs
`zcam_joints.bat`: one joint per zoom (34 over `zcam34_joint,zcam_consensus`, the others over `zcam_consensus`),
each promoted if its checks pass, the Z34 figures, and the combined check below.

### Combined check: `mppp calib nz`

Module [`mppp.calib.nzjoint`](api.md#mppp.calib.nzjoint). One adjustment of the finished Navcam + Mastcam-Z blocks
in which the Navcam law and the Mastcam-Z focus law are both free, starting from the promoted laws. It compares the
result with the Navcam-only law: how far each Navcam term moved in units of its standard deviation, the Navcam and
Mastcam-Z residuals before and after, and the rig.

The two solutions agree when no Navcam term moves by more than 3 sd, no focal length by more than 100 ppm, no
principal point by more than 0.3 px, and the Navcam residual median rises by no more than 5 %. If they disagree, the
Mastcam-Z frames pull on the Navcam scale or pointing.

`calib nz` is a check, not a calibration: it writes `nz_joint.json` and `report.md` and never promotes anything. It
exits with 0 whether or not the solutions agree.

`--navcam-group` (0.93.1) adds Navcam-only blocks (those not already in `--group`, 3000 points each) so that the
Navcam law is held by more than the Mastcam-Z blocks; compare the result with the Navcam-only joint. The 22 Oct
check over the six Z34 blocks alone moved NL cx by -1.3 px and both focal lengths by about +250 ppm while the Navcam
residual changed by -0.7 %: a principal-point / attitude degeneracy that the extra blocks should pin.

```bash
mppp calib nz --zoom 34
mppp calib nz --zoom 34 --group zcam34_joint,zcam_consensus --navcam-group ecam_consensus
```

### Hazcam fit: `mppp calib hazcam`

Module [`mppp.calib.hazfit`](api.md#mppp.calib.hazfit) (0.93.1). The Hazcam law in the Navcam form, fitted from the
finished blocks' refined Hazcam cameras and rigs (`sparse/cahv_ba` of each block's project; `--variant` picks the
project folder, `hazcam` for the Hazcam-only alignments):

- per eye, fx and fy with their own intercepts and one shared slope in ppm per degC about T0 = -20 degC (fitted when
  the blocks' median temperatures span at least 5 degC), an early-mission offset on fx and fy (fitted with at least
  two blocks on each side of sol 380), k1 linear in temperature and k2 the weighted mean; cx, cy, p1, p2, k3, k4 the
  median of the label fits (the blocks hold them). Blocks are weighted by their tie-point observations; an eye held
  as weak in a block is left out of that eye's fit;
- per pair, the rig rotation at sol 0 and a linear drift in sol (pitch, yaw, roll, mdeg per sol) from at least three
  blocks with a refined rig, otherwise the median rotation; the translation stays the CAHV one.

Each eye and pair is checked on its own (`hazfit.HAZCAM_LIMITS`: at least 3 blocks, |slope| <= 100 ppm/degC, f0
within 1 % of the label fits, aspect within 1 %, block scatter <= 600 ppm, rig within 0.5 deg of CAHV); the passing
files go to `candidate/` and `--promote-if-ok` promotes them. The block temperature is the median of the eye's
images (the rear Hazcam labels carry stale temperature sensors, so their slope may come out unfitted).

```bash
mppp calib hazcam --group hazcam_consensus --variant hazcam
mppp calib hazcam --group hazcam_consensus --variant hazcam --promote-if-ok
```

Since 0.94.4 `run_pipeline.bat` fits the Hazcams on the `ecam_consensus` blocks (Navcam with both Hazcam pairs)
right after the Navcam joint, both promoted if they pass (`run_hazcam.bat` and `hazcam_calibration.bat`, the
separate `hazcam_consensus` alignment with the Navcam held, were removed in 0.94.4; `run_sites.py --group
hazcam_consensus --variant hazcam --set NAVCAM_INTRINSICS=hold` then the commands above do the same).

## Promotion: `mppp cmods`

A candidate becomes the model in use when it is promoted, either by `--promote-if-ok` above (only when every check
passes) or by hand:

```bash
mppp cmods list                                         # the models in use and their form check
mppp cmods promote <candidate folder or files> --note "why" --dry-run   # check and show, copy nothing
mppp cmods promote <candidate folder or files> --note "why"
```

`promote` validates each file against the form, copies it into the models folder and appends the note to
`cmods/CHANGES.md`; the replaced files stay in the git history. `scripts/promote_cmods.py` does the same from a
script. Commit `src/mppp/data/cmods` after a promotion.

Camera models and site definitions are the author's decisions; open an issue before proposing a change to them.
