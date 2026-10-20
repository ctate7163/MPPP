"""v0p86: does this Python's pycolmap match SIFT features with LightGlue, on the GPU, fast enough?
Run by install_lightglue.bat in the new environment:  python lightglue_check.py <MPPP src folder>
Exit 0: yes; 2: LightGlue works but too slow for MPPP's blocks (CPU ONNX?); 1: no."""
import sys

sys.path.insert(0, sys.argv[1])
import pycolmap                                                   # noqa: E402
from mppp.sfm.matching import LIGHTGLUE_MAX_HOURS_PER_BLOCK, lightglue_probe   # noqa: E402

cuda = bool(getattr(pycolmap, "has_cuda", False))
print(f"pycolmap {pycolmap.__version__}, CUDA {cuda}, {sys.executable}")
if not cuda:
    print("FAIL: this pycolmap has no CUDA")
    sys.exit(1)
print("LightGlue test on synthetic images (1-3 min; the first run downloads the LightGlue model) ...", flush=True)
r = lightglue_probe(use_gpu=True, refresh=True, timeout_s=1800)
print(r)
if not r.get("ok"):
    print("FAIL: LightGlue does not run here:", r.get("error"))
    sys.exit(1)
ms = float(r["ms_per_pair"])
for name, pairs in (("Rochette Z34 (240 images)", 28680), ("a 400-image Navcam block", 79800)):
    print(f"  {name}: {pairs} pairs x {ms:.0f} ms = {pairs * ms / 3.6e6:.1f} h of LightGlue matching")
h = 28680 * ms / 3.6e6
if h > LIGHTGLUE_MAX_HOURS_PER_BLOCK:
    print(f"SLOW: {ms:.0f} ms per pair with {r.get('keypoints')} keypoints per image - MPPP would fall back to SIFT on "
          f"blocks like Rochette (limit {LIGHTGLUE_MAX_HOURS_PER_BLOCK:g} h per block)")
    sys.exit(2)
print(f"PASS: SIFT + LightGlue on the GPU, {ms:.0f} ms per pair with {r.get('keypoints')} keypoints per image")
sys.exit(0)
