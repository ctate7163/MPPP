# The rover model

MPPP ships a 3D model of Perseverance for the station products: with the rover shown, `mppp mesh` adds it
to the station mesh at the rover's pose and the station's images are projected onto it like onto the terrain
(see [Station products](station-products.md)). Other users are welcome to use it for image-based rendering.

| file | what |
|---|---|
| [`perseverance_rmech.glb`](rover/perseverance_rmech.glb) | glTF binary, one mesh per part, 197,539 triangles, 3.6 MB |
| [`perseverance_rmech.npz`](https://github.com/ctate7163/MPPP/raw/main/src/mppp/data/rover/perseverance_rmech.npz) | what `mppp.ibr.rover_model` loads: `vertices`, `faces`, `part`, `meta` |
| [`scripts/build_rover_model.py`](https://github.com/ctate7163/MPPP/blob/main/scripts/build_rover_model.py) | builds the `.npz` from NASA's file; every number below is a named constant in it. `scripts/export_rover_glb.py` writes the `.glb` from the `.npz` |

## Source and credit

The model is derived from NASA's **"Mars Perseverance Rover, 3D Model"** (10 June 2020, file
`25042_Perseverance.glb`), <https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/>.

**Credit: NASA/JPL-Caltech.** The geometry is NASA's; the changes listed here are MPPP's. NASA has not reviewed or
endorsed this version. NASA's material is made available under the
NASA Images and Media Usage Guidelines (see nasa.gov); please credit
NASA/JPL-Caltech when you use the model, and say that it was modified.

## Frame and parts

Metres, in the rover mechanical frame **RMECH**: x forward, y right, z down, z = 0 at the deck top. (glTF viewers
assume y up, so the file opens lying on its side; that is the frame, not an error.) The rover navigation frame's
origin is at (0.09002, 0, −1.13338) m in it. Parts: `body`, `arm`, `arm_base`, `suspension`, `wheels`, `antenna` (the
high-gain antenna) - these go into the station mesh - and `mast` (the fixed mast tube with the two wind-sensor
booms) and `head` (the camera head: Navcams, Mastcam-Z, SuperCam), which do not: the head turns with every image, and
a panorama made from the mast cannot show the mast. Both are used for masking and for shadows.

## What was changed, and why

The aim was a model whose outline matches the rover in the Navcam images when it is drawn through each frame's label
camera model.

1. **Not simplified.** NASA's triangles are kept as they are (197,539). Versions 0.96.1 to 0.97.6 decimated every
   part (71,331 triangles in all); the arm's plate and the wheels lost their shape, so that was undone.
2. **Left out:** the parts under the belly. The camera head is in the file at NASA's pose: looking forward, level.
3. **Placed by the deck fiducials.** Seven of the fiducial discs modelled on NASA's deck (TDF1–4, LDIFF, RDIFF, AFT1)
   were fitted rigidly to their surveyed positions; they agree to 2.2 mm rms. (RPFA1, RPFA2 and AFT2 are 8–18 mm off
   in NASA's model and were not used.) Drawn through a Navcam label camera, the model then lies 2.8 px from the
   image (8.8 px with the earlier fit to a point cloud).
4. **Arm stowed.** NASA's file has the arm deployed. Its five joints (shoulder azimuth, shoulder elevation, elbow,
   wrist, turret) were turned to the stowed pose given by the joint angles in the image labels (below).
5. **Rear deck block** turned 30° about one of its ends to where the images show it.
6. **High-gain antenna** in one neutral pose (the dish to the right of its gimbal, tilted 45° forward and up).
7. **UHF antenna:** the can's wall raised 10 % to the real rim, and NASA's domed cap replaced by a flat floor 5 cm
   inside the rim (the real can is open; the 5 cm is not measured). Its width is NASA's.

## Articulation from the image labels

Every PDS image label records the rover's joint angles when the image was taken.

- **Mast (pan and tilt).** Only the head turns; the mast tube and its booms are fixed to the deck. The label's
  azimuth az and elevation el **resolver** angles place a point x of the head (in the head's own frame) at

      X = P + Rot(G) Rz(az) (h + Ry(-el) x)        (rover navigation frame, metres)

  with P = (0.8012, 0.5569, 0) m where the azimuth axis meets z = 0, h = (-0.0004, 0, -1.9177) m the elevation
  axis from P (0.784 m above the deck top), G the axis' tilt (0.16 degrees) and azimuth zero, Rz and Ry
  right-handed rotations about z (down) and y. In the labels az = 180 degrees looks forward and el = 90 degrees
  is level (el = 0 looks straight down). These numbers come from the mast chain fitted to the solved camera poses
  of 32 stations: the resolver angles fit them to 9 mdeg, the encoder angles (which equal the commanded angles)
  only to 57 mdeg, so the resolvers are used. NASA's model has the head at az = 180, el = 90; at that pose its
  Navcam housings sit within 2 cm of the fitted camera centres and its mast tube 5 mm from the fitted azimuth
  axis. `rover_model.head_in_mech(az, el)` gives the head at any angles; `rover_model.head_angles_deg(R)` gives
  the angles back from a camera's rotation, which is how the shadow maps place the head for each image without
  reading the label again.
- **Arm.** The label's five arm joint angles map to the model's joint angles as
  model = sign × label + offset, with signs (+, −, −, −, −) and offsets (0, 0, −90, +90, 0) degrees
  (`mppp.ibr.rover_model.arm_model_angles_deg`). The stowed pose is the label angles
  (90.09, −16.37, −161.66, 178.18, 278.52)°; over 443 stows it repeats to 7–29 mdeg, so one stowed model serves
  every station where the arm is stowed (`arm_is_stowed`).
- **High-gain antenna.** The label azimuth and elevation of a checked frame (65°, −45°) match the model's neutral
  pose.
- **Wheels and suspension.** The labels carry four steering angles, two bogie angles and two differential angles.

**What is not done yet:** apart from the head, the shipped model is one pose. A deployed arm, the antenna's pointing on a given sol and
the suspension on rough ground are not followed at run time; where the real rover differs from the model, the images
are projected onto the wrong surface there.

## Use

```python
from mppp.ibr import rover_model
V, F, meta = rover_model.load()                  # vertices (m, RMECH), triangles, metadata (source, parts, frame)
V, F, meta = rover_model.load(parts=["body", "arm", "antenna"])     # some parts only
Vh, Fh = rover_model.head_in_mech(az_deg=250.0, el_deg=70.0)       # the camera head at the label's mast angles
```

or open the `.glb` in any glTF tool. To rebuild from NASA's file:
`python scripts/build_rover_model.py 25042_Perseverance.glb`.
