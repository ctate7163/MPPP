"""
Write the rover model of mppp.ibr.rover_model as a binary glTF file, one mesh per part, for use outside MPPP
(0.97.6, Christian 13 Nov 2026: "host the rover 3d model on github for other MPPP users").

    python scripts/export_rover_glb.py [--out docs/rover/perseverance_rmech.glb]

The file is in the rover mechanical frame RMECH (x forward, y right, z down, z 0 = the deck top), metres - not in
glTF's y-up convention: a glTF viewer shows it lying on its side.  Needs trimesh.  The model is derived from NASA's
"Mars Perseverance Rover, 3D Model" (credit NASA/JPL-Caltech); see docs/rover-model.md for the changes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CREDIT = "NASA/JPL-Caltech (Mars Perseverance Rover, 3D Model); modified by MPPP"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(ROOT / "docs" / "rover" / "perseverance_rmech.glb"))
    a = ap.parse_args()
    import trimesh
    from mppp.ibr import rover_model
    scene = trimesh.Scene()
    V, _, meta = rover_model.load()
    for part in meta["parts"]:
        _, F, _ = rover_model.load(parts=[part])
        used, inv = np.unique(F, return_inverse=True)
        scene.add_geometry(trimesh.Trimesh(V[used], inv.reshape(F.shape), process=False), geom_name=part, node_name=part)
    scene.metadata.update({"credit": CREDIT, "source": meta.get("url"), "frame": meta.get("frame"),
                           "changes": "https://ctate7163.github.io/MPPP/rover-model/"})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(trimesh.exchange.gltf.export_glb(scene))
    print(f"{out}: {len(meta['parts'])} parts, {sum(len(g.faces) for g in scene.geometry.values())} triangles, {out.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
