"""Rational (FULL_OPENCV, k4 denominator) vs OPENCV_FISHEYE for the Navcam: same block, same observations, each
camera model started from a least-squares fit to the converged rational camera, then the final adjustment repeated."""
# Usage: python scripts/lens_model_experiment.py <sfm project root> <out.json>   (v0p30; results in
# docs/results/working_notes.md section 11).  Variants: rational, fisheye (OPENCV_FISHEYE), fisheye_tangential
# (THIN_PRISM_FISHEYE with sx1, sy1 held), thin_prism_fisheye (sx1, sy1 free).
import sys, copy, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pycolmap
from scipy.optimize import least_squares
from mppp.sfm.project import SfmProject
from mppp.sfm.reconstruction import bundle_adjust
from lens_terms_experiment import observations, residual_stats
root = sys.argv[1]
proj = SfmProject.load(root); rec0 = pycolmap.Reconstruction(str(root + '/sparse/cahv_ba'))
key_of = {int(v): k for k, v in proj.settings['database']['cameras'].items()}

def fit_fisheye(cam):
    # rays of a pixel grid through the rational camera, then fisheye params that reproject them onto the grid
    w, h = cam.width, cam.height
    u, v = np.meshgrid(np.linspace(1, w - 1, 64), np.linspace(1, h - 1, 48))
    px = np.stack([u.ravel(), v.ravel()], 1)
    rays = np.array([cam.cam_from_img(p) for p in px])          # normalised x, y (z = 1)
    fx, fy, cx, cy = cam.params[:4]
    def model(q):
        fx, fy, cx, cy, k1, k2, k3, k4 = q
        r = np.hypot(rays[:, 0], rays[:, 1]); th = np.arctan(r)
        thd = th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return np.stack([fx * rays[:, 0] * s + cx, fy * rays[:, 1] * s + cy], 1)
    q0 = [fx, fy, cx, cy, 0, 0, 0, 0]
    res = least_squares(lambda q: (model(q) - px).ravel(), q0, method='lm')
    fit_rms = float(np.sqrt(np.mean(np.sum((model(res.x) - px) ** 2, 1))))
    return res.x, fit_rms

out = {}
for variant in ('rational', 'fisheye', 'fisheye_tangential', 'thin_prism_fisheye'):
    rec = copy.deepcopy(rec0); p = copy.deepcopy(proj)
    fits = {}
    if variant != 'rational':
        for cid, cam in rec.cameras.items():
            if key_of.get(int(cid)) not in ('NL', 'NR'):
                continue
            q, fit_rms = fit_fisheye(cam)
            if variant == 'fisheye':
                cam.model = pycolmap.CameraModelId.OPENCV_FISHEYE
                cam.params = q
                p.cameras[key_of[int(cid)]]['model'] = 'OPENCV_FISHEYE'
            else:   # fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1: the fisheye radial with tangential and thin-prism terms
                cam.model = pycolmap.CameraModelId.THIN_PRISM_FISHEYE
                cam.params = np.array([q[0], q[1], q[2], q[3], q[4], q[5], 0.0, 0.0, q[6], q[7], 0.0, 0.0])
                p.cameras[key_of[int(cid)]]['model'] = 'THIN_PRISM_FISHEYE'
            fits[key_of[int(cid)]] = {'fit_rms_px': fit_rms, 'params': q.tolist()}
            p.cameras[key_of[int(cid)]]['free_params'] = ['sx1', 'sy1'] if variant == 'thin_prism_fisheye' else []
        print('fisheye start fit to the rational camera (grid rms px):', {k: round(v['fit_rms_px'], 4) for k, v in fits.items()}, flush=True)
    t = time.time()
    # fisheye: k1-k4 must be refined; MPPP's fixed_camera_params holds "extra params" for unknown models -> pass through refine_tangential
    ba = bundle_adjust(rec, p, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=True, refine_rig='rotation',
                       max_iterations=200, attitude_prior_deg=1.0)
    o = observations(rec, p); st = residual_stats(o)
    out[variant] = {'cost': ba['final_cost'], 'iterations': ba['iterations'], **st, 'fits': fits,
                    'params': {key_of[int(c)]: list(map(float, rec.cameras[c].params)) for c in rec.cameras if key_of.get(int(c)) in ('NL', 'NR')}}
    print(f"{variant:9s} cost {ba['final_cost']:.1f} it {ba['iterations']}  median {st['median_px']:.4f}  rms {st['rms_px']:.4f}  "
          f"centre/edge/corner median {st['centre_median_px']:.4f}/{st['edge_median_px']:.4f}/{st['corner_median_px']:.4f}  "
          f"field {st['residual_field_rms_px']:.4f} (corner cells {st['residual_field_corner_rms_px']:.4f})  {time.time()-t:.0f} s", flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
