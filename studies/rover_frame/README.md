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
