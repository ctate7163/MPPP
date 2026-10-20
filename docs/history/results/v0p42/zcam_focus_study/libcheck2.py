import sys, json, numpy as np
sys.path.insert(0, '/home/claude/MPPP_repo/src')
from mppp.sfm import calibration as CAL
B = {"Rockytop": "/mnt/user-data/uploads/scapes/rockytop_colmap_nav_zcam34",
     "Three Forks": "/mnt/user-data/uploads/threeforks_colmap_nav_zcam34",
     "Airey Hill": "/mnt/user-data/uploads/scapes/aireyhill_colmap_nav_zcam34"}
sols = {k: CAL.load_solution(v, k) for k, v in B.items()}
rows = CAL.zcam_boresight_table(sols, min_observations=0)
d = json.load(open('data.json')); P = {(p['block'], p['sclk']): p for p in d['pairs']}
lib = {(r['scape'], r['sclk']): r for r in rows}
common = set(P) & set(lib)
print(len(rows), len(d['pairs']), len(common))
# compare eqy per pair
bins = {(b['block'], b['bin']): b for b in d['bins']}
dd = []
for k in common:
    p = P[k]; r = lib[k]
    fR = 0.5 * (bins[(p['block'], p['bin_R'])]['fx'] + bins[(p['block'], p['bin_R'])]['fy'])
    eqy = bins[(p['block'], p['bin_R'])]['cy'] - bins[(p['block'], p['bin_L'])]['cy'] - fR * np.radians(p['pitch_mdeg'] * 1e-3)
    dd.append(r['eqy_px'] - eqy)
print('eqy lib - study: median', np.median(dd), 'max abs', np.max(np.abs(dd)))
obs = {k: (P[k]['obs_L'], P[k]['obs_R']) for k in common}
for mo in (0, 100, 300):
    sel = [lib[k] for k in common if min(obs[k]) >= mo]
    for passes in (1, 2):
        b = CAL.fit_zcam_boresight(sel, clip=3.5)
        print(mo, len(sel), {q: round(1e3 * b[q]['slope_per_count'], 2) for q in ('eqx_px', 'eqy_px')}, {q: b[q]['n'] for q in ('eqx_px','eqy_px')})
        break
