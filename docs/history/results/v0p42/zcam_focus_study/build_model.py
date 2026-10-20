"""v0p42: build m20_cmods/M2020_ZCAM034_focus_model.json through the library path (notebook 04 §5b functions)."""
import sys, json, copy
sys.path.insert(0, '/home/claude/MPPP_repo/src')
from mppp.sfm import calibration as CAL
B = {"Rockytop": "/mnt/user-data/uploads/scapes/rockytop_colmap_nav_zcam34",
     "Three Forks": "/mnt/user-data/uploads/threeforks_colmap_nav_zcam34",
     "Airey Hill": "/mnt/user-data/uploads/scapes/aireyhill_colmap_nav_zcam34"}
OUT = '/home/claude/MPPP_repo/src/mppp/data/m20_cmods/M2020_ZCAM034_focus_model.json'
old = json.load(open(OUT))
prev = old.get('previous') or {g: {k: old['cameras'][g][k] for k in ('f0_px', 'reference_focus', 'slope_px_per_count', 'aspect')}
                                for g in old['cameras']}
sols = {k: CAL.load_solution(v, k) for k, v in B.items()}
d = json.load(open('/home/claude/zfocus/data.json')); T = {im['name']: im['T_fpa'] for im in d['images']}
for s in sols.values():            # HEAD_FPA from 86 labels, interpolated in SCLK within sol and eye (addT.py)
    for r in s.images.values():
        if r['name'] in T:
            r['camera_temperature_degC'] = T[r['name']]
FOCUS = CAL.focus_table(sols)
for r in FOCUS:                    # backlash state: f / label > 1.005 (these 0.22 blocks have no <bin>_reg cameras)
    if r.get('f_label_median_px') and r['f_refined_px'] / r['f_label_median_px'] <= 1.005:
        r['state'] = 'regular'
kw = dict(min_observations=1000, min_focus=-1100, per_scape_offset=False, navcam_normalise=True, reference_focus=600.0)
fits, meas, rocky, offs = {}, {}, {}, {}
for g in ('ZL034', 'ZR034'):
    fits[g] = CAL.fit_focus_model(FOCUS, g, trend=True, **kw)
    meas[g] = CAL.fit_focus_model(FOCUS, g, trend=True, thermal=True, **kw)['thermal']
    rocky[g] = CAL.fit_focus_model([r for r in FOCUS if r['scape'] == 'Rockytop'], g, thermal=True, **kw)['thermal']
    offs[g] = CAL.fit_focus_model(FOCUS, g, **dict(kw, per_scape_offset=True))['scape_offsets_px']
bore_rows = CAL.zcam_boresight_table(sols)
bore = CAL.fit_zcam_boresight(bore_rows, bootstrap=300)
boreT = CAL.fit_zcam_boresight(bore_rows, thermal=True)
m = CAL.focus_model_json(fits, bore, focus_range={"ZL034": [-400, 1300], "ZR034": [-1100, 1300]})
for g, c in m['cameras'].items():
    f = fits[g]
    c['f0_sd_px'] = None
    c['slope_sd_px_per_count'] = f['slope_sd_px_per_count']
    c['n_bins'] = f['n_bins']
    c['thermal'] = {"T0_degC": -15.0, "sensor": "HEAD_FPA", "f_px_per_degC": 0.0,
                    "measured_px_per_degC": [meas[g]['f_px_per_degC'], meas[g]['sd_px_per_degC']],
                    "measured_within_rockytop_px_per_degC": [rocky[g]['f_px_per_degC'], rocky[g]['sd_px_per_degC']],
                    "T_range_degC": meas[g]['T_range_degC'],
                    "note": "not significant once each block's Navcam focal scale is divided out; held at 0"}
    c['trend'] = dict(f['trend'], sol_range=[461, 991],
                      note="3 blocks (sols 461-530, 684-692, 961-991); carried by the Airey Hill offset; "
                           "evaluated at the sol clamped to sol_range (no extrapolation); provisional")
    c['block_offsets_px'] = offs[g]
    if g == 'ZR034':
        c['pp'].update(sd_boot_px_per_count=[bore['eqx_px']['sd_boot'], bore['eqy_px']['sd_boot']],
                       note="right-minus-left equivalent boresight against focus (Huber, per-block offsets, "
                            f"{bore['pairs']} stereo pairs), put on ZR; about the eye's median focus in the block")
    else:
        c['pp']['note'] = "reference eye: the right-minus-left difference is carried by ZR"
m['stereo'] = {"roll_mdeg_per_count": bore['roll_mdeg']['slope_per_count'], "roll_sd_boot": bore['roll_mdeg']['sd_boot'],
               "eqx_px_block": bore['eqx_px']['offsets'], "eqy_px_block": bore['eqy_px']['offsets'],
               "roll_mdeg_block": bore['roll_mdeg']['offsets'],
               "per_degC": {q: [boreT[q]['per_degC'], boreT[q]['sd_per_degC']] for q in ('eqx_px', 'eqy_px', 'roll_mdeg')},
               "note": "information; Mastcam-Z has no rig by default (notebook 03 ZCAM_RIG)"}
m['units'] = ("full-frame pixels (1648 x 1200); f = f0_px + slope_px_per_count (focus - reference_focus) "
              "+ thermal.f_px_per_degC (T_FPA - thermal.T0_degC) + trend.f_px_per_sol (clamp(sol, trend.sol_range) - "
              "trend.sol0); fy/fx = aspect; cx, cy + pp.c*_px_per_count (bin focus - the eye's median focus in the block) "
              "(mppp.sfm.project.zcam_model_focal, zcam_model_pp_shift)")
m['source'] = ("MPPP v0p42 (docs/results/v0p42/zcam_focus_study/build_model.py): backlash-state focus bins (>= 1000 "
               "observations, focus >= -1100, weights = observations) of the Rockytop (sols 461-530), Three Forks "
               "(684-692) and Airey Hill (961-991) Navcam + Mastcam-Z blocks (MPPP 0.22), each block's Mastcam-Z f "
               "divided by its Navcam focal scale (refined / start: Rockytop 1.00025, Three Forks 1.00351, Airey Hill "
               "1.00003); HEAD_FPA temperatures from the labels. Replaces the v0.21.1 model (polynomial Navcam, focus "
               "states mixed).")
m['previous'] = prev
json.dump(m, open(OUT, 'w'), indent=1, default=float)
json.dump({"fits": fits, "thermal_measured": meas, "thermal_rockytop": rocky, "block_offsets": offs,
           "boresight": bore, "boresight_thermal": boreT}, open('/home/claude/zfocus/build_model_fits.json', 'w'), indent=1, default=float)
for g, c in m['cameras'].items():
    print(g, {k: c[k] for k in ('f0_px', 'slope_px_per_count', 'slope_sd_px_per_count', 'aspect', 'fit_rms_px', 'label_f0_px', 'label_slope_px_per_count', 'n_bins')},
          'trend', round(c['trend']['f_px_per_sol'], 4), '±', round(c['trend']['sd_px_per_sol'], 4), 'pp', c['pp'], 'offs', {k: round(v, 1) for k, v in c['block_offsets_px'].items()})
    print('   thermal measured', [round(x, 3) for x in c['thermal']['measured_px_per_degC']], 'rockytop', [round(x, 3) for x in c['thermal']['measured_within_rockytop_px_per_degC']], c['thermal']['T_range_degC'])
print('stereo', json.dumps(m['stereo'], default=float)[:800])
