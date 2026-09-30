import json, numpy as np
d = json.load(open('data.json'))
BL = ('Rockytop', 'Three Forks', 'Airey Hill')
F0, T0, S0 = 600.0, -15.0, 700.0
def fit(y, X, names, w=None):
    w = np.ones(len(y)) if w is None else w
    sw = np.sqrt(w); c, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None); r = y - X @ c
    s2 = (w * r * r).sum() / max(len(y) - X.shape[1], 1)
    cov = np.linalg.inv((X * w[:, None]).T @ X) * s2
    return dict(zip(names, zip(c, np.sqrt(np.diag(cov))))), r
res = {}
for g in ('ZL034', 'ZR034'):
    bb = [b for b in d['bins'] if b['group'] == g and not b['f_held'] and b['obs'] >= 1000]
    f = np.array([0.5 * (b['fx'] + b['fy']) for b in bb]); lab = np.array([b['label_f_median'] for b in bb])
    back = f / lab > 1.005
    bb = [b for b, k in zip(bb, back) if k]; f = f[back]
    F = np.array([b['focus'] for b in bb]) - F0; T = np.array([b['T_fpa'] for b in bb]) - T0
    S = np.array([b['sol'] for b in bb]) - S0; blk = np.array([b['block'] for b in bb]); one = np.ones(len(f))
    w = np.minimum(np.array([b['obs'] for b in bb]) / 5000.0, 1.0)   # down-weight thin bins
    print(f"\n{g}: {len(f)} backlash bins; T_fpa {T.min()+T0:.1f}..{T.max()+T0:.1f} C; corr(F,T) {np.corrcoef(F,T)[0,1]:+.2f}")
    models = {
        "focus": (np.c_[one, F], ['f0', 'a']),
        "focus+sol": (np.c_[one, F, S], ['f0', 'a', 'c']),
        "focus+T": (np.c_[one, F, T], ['f0', 'a', 'b']),
        "focus+T+sol": (np.c_[one, F, T, S], ['f0', 'a', 'b', 'c']),
        "blocks+focus": (np.c_[[(blk == x).astype(float) for x in BL]].T.__array__() if False else np.column_stack([(blk == x).astype(float) for x in BL] + [F]), list(BL) + ['a']),
        "blocks+focus+T": (np.column_stack([(blk == x).astype(float) for x in BL] + [F, T]), list(BL) + ['a', 'b']),
    }
    res[g] = {}
    for nm, (X, names) in models.items():
        keep = X.any(0); X = X[:, keep]; names = [n for n, k in zip(names, keep) if k]
        c, r = fit(f, X, names, w)
        rms = float(np.sqrt(np.average(r * r, weights=w)))
        n, k = len(f), X.shape[1]; bic = n * np.log(np.average(r * r, weights=w)) + k * np.log(n)
        s = "  ".join(f"{a} {v:.4g}±{e:.2g}" for a, (v, e) in c.items())
        print(f"  {nm:16s} rms {rms:5.2f}  BIC {bic:6.1f}  {s}")
        res[g][nm] = {"coef": {a: [float(v), float(e)] for a, (v, e) in c.items()}, "rms": rms, "bic": float(bic)}
    # within Rockytop only (the only block with a wide temperature range)
    s = blk == 'Rockytop'
    c, r = fit(f[s], np.c_[one[s], F[s], T[s]], ['f0', 'a', 'b'], w[s])
    print(f"  Rockytop only    rms {np.sqrt(np.average(r*r,weights=w[s])):5.2f}  " + "  ".join(f"{a} {v:.4g}±{e:.2g}" for a, (v, e) in c.items()))
    res[g]['rockytop_focus+T'] = {"coef": {a: [float(v), float(e)] for a, (v, e) in c.items()}}
json.dump(res, open('fitsT.json', 'w'), indent=1)

# --- R-L boresight vs focus, T, sol
bins = {(b['block'], b['bin']): b for b in d['bins']}
P = [p for p in d['pairs'] if p['obs_L'] >= 100 and p['obs_R'] >= 100 and p['focus_L'] is not None]
fR = np.array([0.5 * (bins[(p['block'], p['bin_R'])]['fx'] + bins[(p['block'], p['bin_R'])]['fy']) for p in P])
dcx = np.array([bins[(p['block'], p['bin_R'])]['cx'] - bins[(p['block'], p['bin_L'])]['cx'] for p in P])
dcy = np.array([bins[(p['block'], p['bin_R'])]['cy'] - bins[(p['block'], p['bin_L'])]['cy'] for p in P])
eqx = dcx + fR * np.radians(np.array([p['yaw_mdeg'] for p in P]) * 1e-3)
eqy = dcy - fR * np.radians(np.array([p['pitch_mdeg'] for p in P]) * 1e-3)
roll = np.array([p['roll_mdeg'] for p in P])
F = np.array([0.5 * (p['focus_L'] + p['focus_R']) for p in P]) - F0
T = np.array([0.5 * (p['T_fpa_L'] + p['T_fpa_R']) for p in P]) - T0
S = np.array([p['sol'] for p in P]) - S0; blk = np.array([p['block'] for p in P]); one = np.ones(len(P))
print(f"\nR-L boresight from {len(P)} stereo pairs; corr(F,T) {np.corrcoef(F,T)[0,1]:+.2f}")
bres = {}
for nm, y in (('eqx_px', eqx), ('eqy_px', eqy), ('roll_mdeg', roll)):
    bres[nm] = {}
    for mn, X, names in (("focus+T+sol", np.c_[one, F, T, S], ['c0', 'focus', 'T', 'sol']),
                         ("blocks+focus+T", np.column_stack([(blk == x).astype(float) for x in BL] + [F, T]), list(BL) + ['focus', 'T'])):
        # robust: iterate once with a 3.5-sigma clip
        c, r = fit(y, X, names); m = np.abs(r - np.median(r)) < 3.5 * 1.4826 * np.median(np.abs(r - np.median(r)))
        c, r = fit(y[m], X[m], names)
        print(f"  {nm:10s} {mn:15s} n {m.sum()}  scatter {1.4826*np.median(np.abs(r)):.2f}  " + "  ".join(
            f"{a} {v*(1000 if a in ('focus','sol') else 1):+.3g}±{e*(1000 if a in ('focus','sol') else 1):.2g}" for a, (v, e) in c.items()))
        bres[nm][mn] = {a: [float(v), float(e)] for a, (v, e) in c.items()}
json.dump(bres, open('boresightT.json', 'w'), indent=1)
print("(focus and sol slopes are per 1000 counts / 1000 sol; T per degC)")
