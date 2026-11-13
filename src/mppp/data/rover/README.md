# perseverance_rmech.npz

A model of the Perseverance rover in the rover mechanical frame (RMECH: x forward, y right, z down,
z = 0 at the deck top; metres), loaded by `mppp.ibr.rover_model`.

Derived from NASA's "Mars Perseverance Rover, 3D Model" (file `25042_Perseverance.glb`),
https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/ . **Credit: NASA/JPL-Caltech.**

Modified by MPPP (`scripts/build_rover_model.py`): NASA's triangles as they are (197,539; not simplified since 0.97.7); placed by seven deck fiducial discs; the arm turned
to its stowed pose from the joint angles in the image labels; the rear deck block, the high-gain antenna and the UHF
antenna can adjusted to the Navcam images; the camera head kept as a part of its own (turned by the mast angles for shadows), the parts under the belly left out. NASA has not
reviewed or endorsed this version. The changes are described in `docs/rover-model.md`
(https://ctate7163.github.io/MPPP/rover-model/).
