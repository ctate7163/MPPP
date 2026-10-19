import json, math, collections, numpy as np
from refdata import PANOS, SAMPLES, SAMPLE_NOTES
d=json.load(open('M20_waypoints.json'))
P=[x['properties'] for x in d['features']]
# chronological order check
order=sorted(range(len(P)),key=lambda i:(P[i]['sol'],P[i]['site'],P[i]['drive']))
print('file already chronological:',order==list(range(len(P))))
P=[P[i] for i in order]
for p in P:
    im=p['images'][0]; c=int(im['columns'] or 0)
    p['res']={18584:'full',18585:'full',9292:'med',4646:'low'}.get(c,'none' if c==0 else 'partial')
X=np.array([p['easting'] for p in P]); Yn=np.array([p['northing'] for p in P])
E=[i for i,p in enumerate(P) if p['final']=='y']
EX=X[E]; EY=Yn[E]
R=10.0
# position at a sol = last waypoint with sol<=s
sols_all=np.array([p['sol'] for p in P])
def pos_at(s):
    idx=np.where(sols_all<=s)[0]
    i=idx[-1] if len(idx) else 0
    return X[i],Yn[i],i
# ---- clusters (from cluster.py logic) ----
cl=json.load(open('clusters.json'))
# map EOD index -> cluster id for clusters with n>=4
Eidx={P[i]['RMC']:k for k,i in enumerate(E)}
big=[c for c in cl if c['n']>=4]
for ci,c in enumerate(big):
    c['id']=ci+1
    mem=[Eidx[r] for r in c['rmc']]
    c['mem']=mem
    mx,my=EX[mem],EY[mem]
    c['panos']=[];c['samples']=[]
    for num,ss,name,zoom in PANOS:
        hit=False
        for s in range(ss[0],ss[-1]+1):
            x,y,_=pos_at(s)
            if np.min(np.hypot(mx-x,my-y))<=R and s in ([*ss] if len(ss)<3 else ss) or (np.min(np.hypot(mx-x,my-y))<=R and s==ss[0]):
                hit=True
        # simpler: check first and last acquisition sol positions
        hits=[s for s in ss if np.min(np.hypot(mx-pos_at(s)[0],my-pos_at(s)[1]))<=R]
        if hits: c['panos'].append(f"#{num} {name} ({'-'.join(map(str,ss))}, {zoom} mm)")
    for num,name,loc,s in SAMPLES:
        x,y,_=pos_at(s)
        if np.min(np.hypot(mx-x,my-y))<=R: c['samples'].append(f"{name} ({loc}, sol {s})")
    if any(np.min(np.hypot(mx-pos_at(s)[0],my-pos_at(s)[1]))<=R for s in range(653,691)): c['depot']=True
json.dump(big,open('clusters_big.json','w'),indent=1)
for c in big:
    print(c['id'],c['n'],c['core'],c['sols'][0],c['sols'][-1],c['panos'],c['samples'],c.get('depot'))
# unmatched panos / samples: where were they?
print('\nPanos not in any >=4 cluster:')
for num,ss,name,zoom in PANOS:
    if not any(f"#{num} " in s for c in big for s in c['panos']):
        x,y,i=pos_at(ss[0]); nn=np.sum(np.hypot(EX-x,EY-y)<=R)
        print(f"  #{num} {name} sols {ss}: EOD 360s within 10 m of rover position = {nn}")
print('\nSamples not in any >=4 cluster:')
for num,name,loc,s in SAMPLES:
    if not any(name in t for c in big for t in c['samples']):
        x,y,i=pos_at(s); nn=np.sum(np.hypot(EX-x,EY-y)<=R)
        print(f"  {num} {name} sol {s}: EOD 360s within 10 m = {nn}")
