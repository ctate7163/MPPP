import sys, json
sys.path.insert(0, '/home/claude/MPPP_repo/src')
from mppp.sfm import calibration as CAL
exec(open('libcheck.py').read().split("for g in ('ZL034'")[0].split("sols = ")[0])
sols = {k: CAL.load_solution(v, k) for k, v in B.items()}
d = json.load(open('/home/claude/zfocus/data.json')); T = {im['name']: im['T_fpa'] for im in d['images']}
for s in sols.values():
    for r in s.images.values():
        if r['name'] in T: r['camera_temperature_degC'] = T[r['name']]
FOCUS = CAL.focus_table(sols)
for r in FOCUS:
    if r.get('f_label_median_px') and r['f_refined_px'] / r['f_label_median_px'] <= 1.005: r['state'] = 'regular'
for g in ('ZL034', 'ZR034'):
    for kw in (dict(trend=True), dict(thermal=True, trend=True), dict()):
        f = CAL.fit_focus_model(FOCUS, g, min_observations=1000, min_focus=-1100, per_scape_offset=False, navcam_normalise=True, reference_focus=600.0, **kw)
        print(g, kw, 'f0 %.1f slope %.4f±%.4f' % (f['f0_px'], f['slope_px_per_count'], f['slope_sd_px_per_count']),
              'T', ((round(f['thermal']['f_px_per_degC'], 3), round(f['thermal']['sd_px_per_degC'], 3)) if 'thermal' in f else '-'),
              'sol', ((round(f['trend']['f_px_per_sol'], 4), round(f['trend']['sd_px_per_sol'], 4)) if 'trend' in f else '-'), 'rms %.2f' % f['rms_px'])
    f = CAL.fit_focus_model(FOCUS, g, min_observations=1000, min_focus=-1100, per_scape_offset=True, navcam_normalise=True, reference_focus=600.0)
    print(g, 'per-scape offsets', {k: round(v, 1) for k, v in f['scape_offsets_px'].items()}, 'slope %.4f' % f['slope_px_per_count'])
