"""Per focus bin and per stereo pair data from the solved Nav+Zcam blocks (v0.22 native models)."""
import sys, json, re, collections
sys.path.insert(0, '/home/claude/MPPP_repo/src')
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from mppp.sfm.thermal import interpolate_temperatures
BLOCKS = {"Rockytop": "/mnt/user-data/uploads/scapes/rockytop_colmap_nav_zcam34",
          "Three Forks": "/mnt/user-data/uploads/threeforks_colmap_nav_zcam34",
          "Airey Hill": "/mnt/user-data/uploads/scapes/aireyhill_colmap_nav_zcam34"}
samples = json.load(open('/home/claude/calib/label_temps_all.json'))
bins, pairs, images = [], [], []
for blk, root in BLOCKS.items():
    col = Path(root) / 'colmap'
    proj = json.load(open(col / 'project.json'))
    summ = json.load(open(col / 'error_input' / 'summary.json'))
    nat = col / 'error_input' / 'native'
    # native image id -> name, camera id (header lines only)
    name_of, ncam = {}, {}
    with open(nat / 'images.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            t = line.split()
            if len(t) == 10 and t[9].endswith('.png'):
                name_of[int(t[0])] = t[9]
                next(f, None)
    frames = {}
    for line in open(nat / 'frames.txt'):
        if line.startswith('#') or not line.strip():
            continue
        t = line.split()
        q = np.array(t[2:6], float)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        frames[int(t[12])] = (R, -R.T @ np.array(t[6:9], float))
    res = np.load(col / 'error_input' / 'residuals.npz')
    nobs = collections.Counter(res['image_id'].tolist())
    rows = {r['name']: r for r in proj['images']}
    zst = [r['stem'] for r in proj['images'] if r['instrument'].startswith('Z')]
    T = interpolate_temperatures(samples, zst, max_gap_s=3600)
    by_name = {}
    for iid, nm in name_of.items():
        r = rows.get(nm)
        if r is None or not r['instrument'].startswith('Z'):
            continue
        t = T.get(r['stem'])
        R, C = frames.get(iid, (None, None))
        d = {"block": blk, "name": nm, "stem": r['stem'], "bin": r['instrument'], "group": r.get('camera_group'), "eye": r['eye'],
             "sol": r['sol'], "sclk": r['sclk_key'], "sequence": r.get('sequence'), "focus": r.get('focus_count'),
             "label_f": r.get('label_f_px'), "label_c": r.get('label_c_px'), "T": (0.5 * (t['NL'] + t['NR']) if t else None),
             "obs": nobs.get(iid, 0), "R": R, "C": C}
        by_name[nm] = d
        images.append({k: v for k, v in d.items() if k not in ('R', 'C')})
    # bins
    ci, cr = summ['cameras_initial'], summ['cameras_refined']
    for k, c in ci.items():
        if not k.startswith('Z') or k not in cr:
            continue
        mem = [d for d in by_name.values() if d['bin'] == k]
        if not mem:
            continue
        p = cr[k]['params']
        Ts = [d['T'] for d in mem if d['T'] is not None]
        bins.append({"block": blk, "bin": k, "group": c.get('group'), "focus": c.get('focus_count_median'), "images": len(mem),
                     "obs": int(sum(d['obs'] for d in mem)), "f_held": 'fx' in (c.get('fixed_params') or []),
                     "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3], "f0_start": 0.5 * (c['params'][0] + c['params'][1]),
                     "label_f_median": float(np.median([d['label_f'] for d in mem if d['label_f']])),
                     "label_cx_median": float(np.median([d['label_c'][0] for d in mem if d['label_c']])),
                     "label_cy_median": float(np.median([d['label_c'][1] for d in mem if d['label_c']])),
                     "T": float(np.median(Ts)) if Ts else None, "nT": len(Ts),
                     "sol": float(np.median([d['sol'] for d in mem])), "sequences": sorted({d['sequence'] for d in mem})})
    # stereo pairs: same sclk, both eyes
    bys = collections.defaultdict(dict)
    for d in by_name.values():
        bys[d['sclk']][d['eye']] = d
    for s, e in bys.items():
        if 'L' in e and 'R' in e and e['L']['R'] is not None and e['R']['R'] is not None:
            L, Rr = e['L'], e['R']
            M = Rr['R'] @ L['R'].T                       # right-from-left rotation (camera frames)
            rv = Rotation.from_matrix(M).as_rotvec()
            pairs.append({"block": blk, "sclk": s, "sol": L['sol'], "focus_L": L['focus'], "focus_R": Rr['focus'],
                          "bin_L": L['bin'], "bin_R": Rr['bin'], "obs_L": L['obs'], "obs_R": Rr['obs'],
                          "pitch_mdeg": float(np.degrees(rv[0]) * 1e3), "yaw_mdeg": float(np.degrees(rv[1]) * 1e3),
                          "roll_mdeg": float(np.degrees(rv[2]) * 1e3), "baseline_m": float(np.linalg.norm(Rr['C'] - L['C'])),
                          "T": L['T'], "label_cL": L['label_c'], "label_cR": Rr['label_c'], "fL": L['label_f'], "fR": Rr['label_f'],
                          "cam_cL": None, "cam_cR": None})
    print(blk, len(by_name), 'Zcam images,', sum(1 for b in bins if b['block'] == blk), 'bins,', sum(1 for p in pairs if p['block'] == blk), 'pairs,',
          sum(1 for d in by_name.values() if d['T'] is not None), 'with T', flush=True)
json.dump({"bins": bins, "pairs": pairs, "images": images}, open('/home/claude/zfocus/data.json', 'w'), indent=1, default=float)
