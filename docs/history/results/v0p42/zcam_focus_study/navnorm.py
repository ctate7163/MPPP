import json, numpy as np
exec(open('fitT.py').read().split("res = {}")[0])
B={"Rockytop": "/mnt/user-data/uploads/scapes/rockytop_colmap_nav_zcam34",
   "Three Forks": "/mnt/user-data/uploads/threeforks_colmap_nav_zcam34",
   "Airey Hill": "/mnt/user-data/uploads/scapes/aireyhill_colmap_nav_zcam34"}
nscale={}
for b,r in B.items():
    s=json.load(open(r+'/colmap/error_input/summary.json')); ci,cr=s['cameras_initial'],s['cameras_refined']
    rat=[ (cr[k]['params'][0]+cr[k]['params'][1])/(ci[k]['params'][0]+ci[k]['params'][1]) for k in ('NL','NR')]
    nscale[b]=float(np.mean(rat)); print(b,'Navcam f refined/initial',[round((x-1)*1e6) for x in rat],'ppm')
json.dump(nscale,open('navscale.json','w'))
for g in ('ZL034','ZR034'):
    bb = [b for b in d['bins'] if b['group'] == g and not b['f_held'] and b['obs'] >= 1000]
    f = np.array([0.5 * (b['fx'] + b['fy']) for b in bb]); lab = np.array([b['label_f_median'] for b in bb])
    back = f / lab > 1.005; bb=[b for b,k in zip(bb,back) if k]; f=f[back]
    fn = f/np.array([nscale[b['block']] for b in bb])
    F = np.array([b['focus'] for b in bb]) - F0; T = np.array([b['T_fpa'] for b in bb]) - T0
    S = np.array([b['sol'] for b in bb]) - S0; blk = np.array([b['block'] for b in bb]); one = np.ones(len(f))
    w = np.minimum(np.array([b['obs'] for b in bb]) / 5000.0, 1.0)
    print(g)
    for nm,X,names in (("focus",np.c_[one,F],['f0','a']),("focus+T",np.c_[one,F,T],['f0','a','b']),("focus+T+sol",np.c_[one,F,T,S],['f0','a','b','c']),
                       ("blocks+focus",np.column_stack([(blk==x).astype(float) for x in BL]+[F]),list(BL)+['a']),
                       ("blocks+focus+T",np.column_stack([(blk==x).astype(float) for x in BL]+[F,T]),list(BL)+['a','b'])):
        c,r=fit(fn,X,names,w); n,k=len(f),X.shape[1]; bic=n*np.log(np.average(r*r,weights=w))+k*np.log(n)
        print(f"  Navcam-normalised {nm:15s} rms {np.sqrt(np.average(r*r,weights=w)):5.2f} BIC {bic:6.1f} "+"  ".join(f"{a} {v:.4g}±{e:.2g}" for a,(v,e) in c.items()))
    # label slope for comparison
    L=np.array([b['label_f_median'] for b in bb]); c,_=fit(L,np.c_[one,F],['f0','a']); print('  label (CAHVOR) f vs focus:',{a:round(v,4) for a,(v,e) in c.items()})
