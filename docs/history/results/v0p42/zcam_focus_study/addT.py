import json, numpy as np, collections
d = json.load(open('data.json')); zt = json.load(open('ztemps.json'))
S = collections.defaultdict(list)          # (sol, eye) -> [(sclk, T_fpa)]
for stem, r in zt.items():
    S[(int(stem[4:8]), stem[1])].append((float(r['sclk'].strip('"')), r['HEAD_FPA']))
for k in S: S[k].sort()
def T_of(sol, eye, sclk, gap=5400):
    s = S.get((sol, eye)) or []
    if not s: return None, None
    t = np.array([a for a, _ in s]); v = np.array([b for _, b in s])
    if t[0] <= sclk <= t[-1]:
        return float(np.interp(sclk, t, v)), 0.0
    j = int(np.argmin(np.abs(t - sclk))); dt = abs(t[j] - sclk)
    return (float(v[j]), dt) if dt <= gap else (None, dt)
imgs = {}
n = 0
for im in d['images']:
    sclk = float(im['sclk'].replace('_', '.'))
    im['T_fpa'], im['T_gap_s'] = T_of(im['sol'], im['eye'], sclk)
    n += im['T_fpa'] is not None
    imgs[(im['block'], im['name'])] = im
print(n, 'of', len(d['images']), 'Zcam images have an FPA temperature')
by_bin = collections.defaultdict(list)
for im in d['images']:
    if im['T_fpa'] is not None: by_bin[(im['block'], im['bin'])].append((im['T_fpa'], max(im['obs'], 1)))
for b in d['bins']:
    v = by_bin.get((b['block'], b['bin']))
    b['T_fpa'] = float(np.average([a for a, _ in v], weights=[w for _, w in v])) if v else None
    b['T_fpa_spread'] = float(np.ptp([a for a, _ in v])) if v else None
byS = {(im['block'], im['sclk'], im['eye']): im for im in d['images']}
for p in d['pairs']:
    L = byS.get((p['block'], p['sclk'], 'L')); R = byS.get((p['block'], p['sclk'], 'R'))
    p['T_fpa_L'] = L and L['T_fpa']; p['T_fpa_R'] = R and R['T_fpa']
json.dump(d, open('data.json', 'w'), indent=1)
bb = [b for b in d['bins'] if b['T_fpa'] is not None]
print(len(bb), 'of', len(d['bins']), 'bins with T; spread within bins median', np.median([b['T_fpa_spread'] for b in bb]))
for blk in ('Rockytop', 'Three Forks', 'Airey Hill'):
    t = [im['T_fpa'] for im in d['images'] if im['block'] == blk and im['T_fpa'] is not None]
    print(blk, len(t), 'T range', min(t), max(t))
