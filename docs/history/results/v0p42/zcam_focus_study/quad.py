import json, numpy as np
exec(open('fitT.py').read().split("res = {}")[0])
ns=json.load(open('navscale.json'))
for g in ('ZL034','ZR034'):
    bb=[b for b in d['bins'] if b['group']==g and not b['f_held'] and b['obs']>=1000]
    f=np.array([0.5*(b['fx']+b['fy']) for b in bb]); lab=np.array([b['label_f_median'] for b in bb]); back=f/lab>1.005
    bb=[b for b,k in zip(bb,back) if k]; f=f[back]/np.array([ns[b['block']] for b in bb])
    F=np.array([b['focus'] for b in bb])-F0; blk=np.array([b['block'] for b in bb]); w=np.minimum(np.array([b['obs'] for b in bb])/5000,1)
    print(g,'focus range',F.min()+F0,F.max()+F0, 'n<0:',(F+F0<0).sum())
    Xb=[(blk==x).astype(float) for x in BL]
    c,r=fit(f,np.column_stack(Xb+[F,F**2/1e3]),list(BL)+['a','q'],w); print('  quadratic', {k:(round(v,5),round(e,5)) for k,(v,e) in c.items() if k in 'aq'})
    s=F+F0>=0; c,r=fit(f[s],np.column_stack([x[s] for x in Xb]+[F[s]]),list(BL)+['a'],w[s]); print('  focus>=0 slope', c['a'])
    c,r=fit(f,np.column_stack(Xb+[F]),list(BL)+['a'],w)
    for x,y,z,b in sorted(zip(F+F0,r,f,blk)): print(f"    F {x:7.0f} {b[:5]} f {z:8.1f} res {y:+6.1f}")
