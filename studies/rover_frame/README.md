# Rover frame (study 3A)

What the finished blocks say about the rover body, the mast and the camera mounts — no new adjustment.

    python studies/rover_frame/run_rover_frame.py              # every finished ecam_consensus and zcam_consensus block
    python studies/rover_frame/run_rover_frame.py --sites chal_rocks_tiny seitah_overlook_tiny

`run_pipeline.py` runs it as its `rover_frame` stage, after the error analysis.

- **Labels** (`rover_frame/labels.py`): the rover attitude (`ROVER_COORDINATE_SYSTEM`), the rover motion counter
  (SITE, DRIVE, POSE, ARM, ...), the RSM angles (encoder, resolver, requested) and head frame
  (`RSM_COORDINATE_SYSTEM`), the Navcam plate temperatures, the bogie, differential and arm joint angles, and the
  camera's CAHV pose in the rover frame. Cached in `<root>/camera_analysis/rover_frame_labels.json`.
- **Body poses**: the Hazcams are fixed to the body, so a Hazcam frame's adjusted pose is the body pose of its
  sequence and rover pose (RMC POSE). Its residual against the telemetry attitude is the IMU error plus the block
  frame.
- **Mast head**: a Navcam or Mastcam-Z frame at the same rover pose, relative to that body and in the label's head
  frame (the label kinematics removed), is a constant mount plus the RSM pointing error and flexure. It is fitted
  against cos/sin el, cos/sin az and T. A frame at another rover pose of the station gets the station's measured
  body error carried over to its own telemetry attitude ("propagated").
- **Positions**: the camera centres in the body frame (Hazcams, relative to the front-left Hazcam's origin) and in
  the head frame (mast cameras) are constants. Their scatter is the measurement.
- **Body changes**: consecutive Hazcam poses of one station, giving the adjusted rotation and origin move, the
  telemetry change, and what the telemetry missed, with the POSE / ARM counters and the bogie and arm angle changes
  (wheel settling, arm-induced motion).

Output: `<root>/camera_analysis/rover_frame_<date>/` (`report.md`, `images.csv`, `body_poses.csv`,
`body_changes.csv`, `stations.csv`, `rover_frame.json`, figures), with the figures also in `Claude outputs/`.

## Rover kernels (SPICE, spiceypy)

The study also estimates the rover attitude and position per rover pose and saves them as SPICE kernels
(`rover_frame/estimate.py`, `rover_frame/spice.py`; `--no-kernels` leaves them out). It needs spiceypy
(`pip install spiceypy`, or the `rover` extra).

- **Estimate** (no new adjustment): the end-of-drive pose of a station (its first SITE / DRIVE / POSE / ARM) from
  both Hazcam pairs (`F+R`); later poses from the front pair (`front`), else the rear (`rear`); a pose without
  Hazcam frames gets the station's measured body error carried over by the telemetry change (`propagated`).
  - Global terms over all blocks: the rear-to-front Hazcam correction and origin offset, and the rover frame
    alignment. `M2020_ROVER` is the IMU frame on average: zero mean tilt residual of the end-of-drive bodies.
  - The sigma of a propagated pose is the telemetry scatter after the alignment.
- **Kernels** in `<root>/camera_analysis/rover_kernels_<date>/`:
  - `m2020_surf_rover_mppp_<block>_v01.bc`: CK type 3, `M2020_ROVER` relative to `M2020_LOCAL_LEVEL`. The attitude is
    constant per interval and is not interpolated across a pose change. The segment ID names the source.
  - `m2020_surf_rover_mppp_<block>_v01.bsp`: SPK type 9, the rover relative to the landing site in
    `M2020_LOCAL_LEVEL`. It needs NAIF's clock and leapseconds kernels.
  - `m2020_mppp_hazcams_v01.tf`: TK frames of the four Hazcams relative to `M2020_ROVER`, IDs -168401 to -168404.
    These are unused in `m2020_v04.tf`; to be confirmed with NAIF.
  - `rover_intervals.csv` and `rover_kernels.json`: the sigma, source and RMC of every interval, plus the global terms
    and the read-back check. A CK holds no covariance.
  - A meta-kernel `mppp_rover_<date>.tm`.
- **NAIF base kernels**: the leapseconds, Mars PCK, `m2020_v04.tf` and the newest `m2020_168_sclkscet` are fetched once
  into `%LOCALAPPDATA%\mppp\naif` (`MPPP_NAIF`, `--naif`). Without them, the CK and FK are written against a stub of
  NAIF's three surface frames (same IDs and angles), and the SPK is left out.
- **Times**: encoded SCLK = `round(SPACECRAFT_CLOCK_START_COUNT x 65536)`. The label count is in seconds, with a decimal
  millisecond fraction.
- **Caveat**: the block frame is the landing site's ENU, with site frames taken as north-aligned. North at a later site
  is rotated by about 5–6 mdeg per km of east–west travel.
