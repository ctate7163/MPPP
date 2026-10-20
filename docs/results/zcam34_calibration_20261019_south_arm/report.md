# Mastcam-Z 34 mm calibration (calibrated (v0p74))

MPPP v0p74 Mastcam-Z 34 mm calibration from 1 blocks: mars2020_sol_1408_south_arm_zcam

## Focus lines (p = a + b (focus - reference) [+ c (T - T0)])

| eye | state | param | a | sd a | tau a | chi2/dof | b (per count) | sd b | T slope (per degC) | z | applied | blocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZL034 | backlash | fx | 4720.5838 | 0.6695 | 0.0000 | 0.00 | +0.032031 | 0.001638 | +nan | +nan | False | 1 |
| ZL034 | backlash | fy | 4731.3492 | 1.1848 | 0.0000 | 0.00 | +0.032031 | 0.001638 | +nan | +nan | False | 1 |
| ZL034 | backlash | k1 | -0.4589 | 0.0110 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZL034 | backlash | k2 | 0.6776 | 0.1634 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZL034 | regular | fx | 4676.1602 | 0.6632 | 0.0000 | 0.00 | +0.031729 | 0.001623 | +nan | +nan | False | 1 |
| ZL034 | regular | fy | 4686.8244 | 1.1737 | 0.0000 | 0.00 | +0.031729 | 0.001623 | +nan | +nan | False | 1 |
| ZL034 | regular | k1 | -0.4589 | 0.0110 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZL034 | regular | k2 | 0.6776 | 0.1634 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | backlash | cx | 743.1541 | 0.0000 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | backlash | cy | 663.9622 | 0.0000 | 0.0000 | 0.00 | +0.001222 | 0.000289 | +nan | +nan | False | 1 |
| ZR034 | backlash | fx | 4726.3012 | 1.1175 | 0.0000 | 0.00 | +0.048536 | 0.001921 | +nan | +nan | False | 1 |
| ZR034 | backlash | fy | 4737.0237 | 1.2729 | 0.0000 | 0.00 | +0.048536 | 0.001921 | +nan | +nan | False | 1 |
| ZR034 | backlash | k1 | -0.4732 | 0.0082 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | backlash | k2 | 1.0504 | 0.1453 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | regular | cx | 743.1541 | 0.0000 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | regular | cy | 663.9622 | 0.0000 | 0.0000 | 0.00 | +0.001222 | 0.000289 | +nan | +nan | False | 1 |
| ZR034 | regular | fx | 4681.8239 | 1.1070 | 0.0000 | 0.00 | +0.048079 | 0.001903 | +nan | +nan | False | 1 |
| ZR034 | regular | fy | 4692.4455 | 1.2610 | 0.0000 | 0.00 | +0.048079 | 0.001903 | +nan | +nan | False | 1 |
| ZR034 | regular | k1 | -0.4732 | 0.0082 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |
| ZR034 | regular | k2 | 1.0504 | 0.1453 | 0.0000 | 0.00 | +0.000000 | 0.000000 | +nan | +nan | False | 1 |

Regular (low-backlash) state derived from the backlash lines (v0p74.9): fx, fy divided by ZL034 1.00950 (default, 0 blocks), ZR034 1.00950 (default, 0 blocks)

## Stereo rig (right from left)

- rotation vector +1131.42, +3960.55, -610.25 mdeg
- baseline 243.56 mm
- blocks: mars2020_sol_1408_south_arm_zcam

| block | sol | T optics | rotation vector (mdeg) | baseline (mm) | ZR pp offset moved into the rig (px) |
|---|---|---|---|---|---|
| mars2020_sol_1408_south_arm_zcam | 1412.0 | -19.3 | +1131.4, +3960.6, -610.2 | 243.56 | +0.00, +0.00 |

- ZR principal point at the reference focus fixed at pp0 {'ZR034': [743.1541318639494, 663.9621648425003]} px; each block's offset moved into its rig rotation before averaging (the two are degenerate in a block's BA)

## Checks

- FAIL blocks: 1 blocks
- ok ZL034 lines: fx, fy lines fitted
- ok ZL034 f0: 4721.2 -> 4726.0 px (0.10 %)
- ok ZL034 aspect: 1.00228
- ok ZL034 k1: -0.4589
- ok ZR034 lines: fx, fy lines fitted
- ok ZR034 f0: 4728.8 -> 4731.7 px (0.06 %)
- ok ZR034 aspect: 1.00227
- ok ZR034 k1: -0.4732
- ok ZL034 backlash fx block scatter: between-block sd 0.0 px (chi2/dof 0.0)
- ok ZL034 backlash fy block scatter: between-block sd 0.0 px (chi2/dof 0.0)
- ok ZR034 backlash fx block scatter: between-block sd 0.0 px (chi2/dof 0.0)
- ok ZR034 backlash fy block scatter: between-block sd 0.0 px (chi2/dof 0.0)
- ok rig baseline: 0.2436 m
- ok rig change: 0.038 deg from the rig in use

Promoted on 19 Oct 2026 (v0p87) at Christian's request although the blocks check (>= 2 blocks) fails: one good block (South Arm, sol 1408, ZR-fix diagnostic run) is better than the v0p75 start model for the next run.
