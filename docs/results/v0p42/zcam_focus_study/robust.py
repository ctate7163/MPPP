import sys, json, numpy as np
d = json.load(open('data.json')); bins = {(b['block'], b['bin']): b for b in d['bins']}
BL = ('Rockytop', 'Three Forks', 'Airey Hill')
P = [p for p in d['pairs'] if p['obs_L'] >= 100 and p['obs_R'] >= 100]
fR = np.array([0.5 * (bins[(p['block'], p['bin_R'])]['fx'] + bins[(p['block'], p['bin_R'])]['fy']) for p in P])
eqx = np.array([bins[(p['block'], p['bin_R'])]['cx'] - bins[(p['block'], p['bin_L'])]['cx'] for p in P]) + fR * np.radians(np.array([p['yaw_mdeg'] for p in P]) * 1e-3)
eqy = np.array([bins[(p['block'], p['bin_R'])]['cy'] - bins[(p['block'], p['bin_L'])]['cy'] for p in P]) - fR * np.radians(np.array([p['pitch_mdeg'] for p in P]) * 1e-3)
F = np.array([0.5 * (p['focus_L'] + p['focus_R']) for p in P]) - 600; blk = np.array([p['block'] for p in P])
X = np.column_stack([(blk == b).astype(float) for b in BL] + [F])
def huber(y, X, k=1.345, it=50):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b; s = 1.4826 * np.median(np.abs(r - np.median(r)))
        u = np.abs(r) / (k * s); w = np.where(u <= 1, 1, 1 / u)
        b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ b; s = 1.4826 * np.median(np.abs(r))
    cov = np.linalg.inv((X * w[:, None]).T @ X) * s * s
    return b, np.sqrt(np.diag(cov)), r
def boot(y, X, n=300):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); out.append(huber(y[i], X[i], it=20)[0][-1])
    return np.std(out)
for nm, y in (('eqx', eqx), ('eqy', eqy)):
    b, se, r = huber(y, X)
    print(nm, 'Huber slope %.3f ± %.3f (boot %.3f) px/1000' % (1e3 * b[-1], 1e3 * se[-1], 1e3 * boot(y, X)), 'offsets', np.round(b[:3], 2))
    # binned medians of residual + slope term
    yy = r + b[-1] * F
    for lo in range(-1200, 1300, 300):
        s = (F + 600 >= lo) & (F + 600 < lo + 300)
        if s.sum() > 3: print(f'   focus {lo:5d}..{lo+300:5d} n {s.sum():3d} median {np.median(yy[s]):+.2f}  (line {1e3*b[-1]*(lo+150-600)/1e3:+.2f})')
