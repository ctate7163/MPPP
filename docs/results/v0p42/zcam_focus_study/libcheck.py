"""v0p42: reproduce the study through the library path (focus_table, fit_focus_model, zcam_boresight_table)."""
import sys, json
sys.path.insert(0, '/home/claude/MPPP_repo/src')
from mppp.sfm import calibration as CAL
B = {"Rockytop": "/mnt/user-data/uploads/scapes/rockytop_colmap_nav_zcam34",
     "Three Forks": "/mnt/user-data/uploads/threeforks_colmap_nav_zcam34",
     "Airey Hill": "/mnt/user-data/uploads/scapes/aireyhill_colmap_nav_zcam34"}
sols = {k: CAL.load_solution(v, k) for k, v in B.items()}
d = json.load(open('/home/claude/zfocus/data.json'))
T = {im['name']: im['T_fpa'] for im in d['images']}
for s in sols.values():                   # the interpolated HEAD_FPA temperatures into the project records
    for r in s.images.values():
        if r['name'] in T:
            r['camera_temperature_degC'] = T[r['name']]
            s.manifest.get(r['stem'], {}).pop('camera_temperature_degC', None) if s.manifest else None
FOCUS = CAL.focus_table(sols)
print({s: FOCUS[[r['scape'] for r in FOCUS].index(s)]['navcam_scale'] for s in sols})
# the study's backlash selection: f/label > 1.005
for r in FOCUS:
    if r.get('f_label_median_px') and r['f_refined_px'] / r['f_label_median_px'] <= 1.005:
        r['state'] = 'regular'
for g in ('ZL034', 'ZR034'):
    for kw in (dict(), dict(thermal=True), dict(trend=True), dict(thermal=True, trend=True)):
        f = CAL.fit_focus_model(FOCUS, g, min_observations=1000, min_focus=-1100, per_scape_offset=False,
                                navcam_normalise=True, reference_focus=600.0, **kw)
        print(g, kw, round(f['f0_px'], 1), round(f['slope_px_per_count'], 4), 'T', f.get('thermal', {}).get('f_px_per_degC'),
              'sol', f.get('trend', {}).get('f_px_per_sol'), 'rms', round(f['rms_px'], 2), f['n_bins'])
B = CAL.zcam_boresight_table(sols)
b = CAL.fit_zcam_boresight(B)
print(len(B), {q: (round(1e3 * b[q]['slope_per_count'], 3), round(1e3 * b[q]['sd'], 3), b[q]['n']) for q in ('eqx_px', 'eqy_px', 'roll_mdeg')})
bt = CAL.fit_zcam_boresight(B, thermal=True)
print('thermal', {q: (round(bt[q]['per_degC'], 3), round(bt[q]['sd_per_degC'], 3)) for q in ('eqx_px', 'eqy_px', 'roll_mdeg')})
