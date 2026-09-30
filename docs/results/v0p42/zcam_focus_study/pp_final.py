import json, numpy as np
src = open('fitT.py').read()
exec(src.split("res = {}")[0])
exec("bins = {(b['block'], b['bin']): b for b in d['bins']}\n" + src.split("bins = {(b['block'], b['bin']): b for b in d['bins']}")[1].split("print(f\"\\nR-L")[0])
out = {}
for nm, y in (('eqx_px', eqx), ('eqy_px', eqy), ('roll_mdeg', roll)):
    X = np.column_stack([(blk == x).astype(float) for x in BL] + [F]); names = list(BL) + ['focus']
    c, r = fit(y, X, names); m = np.abs(r - np.median(r)) < 3.5 * 1.4826 * np.median(np.abs(r - np.median(r)))
    c, r = fit(y[m], X[m], names)
    out[nm] = {k: [float(v), float(e)] for k, (v, e) in c.items()}; out[nm]['n'] = int(m.sum()); out[nm]['scatter'] = float(1.4826*np.median(np.abs(r)))
    print(nm, {k: (round(v[0], 5), round(v[1], 5)) if isinstance(v, list) else v for k, v in out[nm].items()})
json.dump(out, open('pp_final.json', 'w'), indent=1)
