import json, math, itertools, collections
import numpy as np
d=json.load(open('M20_waypoints.json'))
P=[x['properties'] for x in d['features']]
for p in P:
    im=p['images'][0]; c=int(im['columns'] or 0)
    full = abs(float(im['azmax'])-float(im['azmin'])-360)<1e-3
    p['res']={18584:'full',18585:'full',9292:'med',4646:'low'}.get(c,'none' if c==0 else 'partial')
    p['pan360']=full
Y=[p for p in P if p['final']=='y']
M=[p for p in P if p['final']!='y']
xy=np.array([[p['easting'],p['northing']] for p in Y])
n=len(Y)
D=np.hypot(xy[:,None,0]-xy[None,:,0], xy[:,None,1]-xy[None,:,1])
R=10.0
# single linkage components
parent=list(range(n))
def find(a):
    while parent[a]!=a:
        parent[a]=parent[parent[a]]; a=parent[a]
    return a
for i in range(n):
    for j in range(i+1,n):
        if D[i,j]<=R: parent[find(i)]=find(j)
# manual merges (user-confirmed site membership): OEB landing waypoints sol 13/14 join the OEB cluster (sol 13 is 10.8 m from sol 33)
MANUAL_MERGE=[('3_0','3_1374'),('3_110','3_1374')]
rmc_idx={p['RMC']:k for k,p in enumerate(Y)}
for a,b in MANUAL_MERGE:
    if a in rmc_idx and b in rmc_idx: parent[find(rmc_idx[a])]=find(rmc_idx[b])
comp=collections.defaultdict(list)
for i in range(n): comp[find(i)].append(i)
def maxclique(idx):
    # largest subset with all pairwise <= R (brute force w/ simple BK)
    idx=list(idx); best=[]
    adj={i:set(j for j in idx if j!=i and D[i,j]<=R) for i in idx}
    def bk(Rs,Ps,Xs):
        nonlocal best
        if not Ps and not Xs:
            if len(Rs)>len(best): best=list(Rs)
            return
        if len(Rs)+len(Ps)<=len(best): return
        for v in list(Ps):
            bk(Rs+[v],Ps&adj[v],Xs&adj[v]); Ps=Ps-{v}; Xs=Xs|{v}
    bk([],set(idx),set())
    return best
def ranges(sols):
    sols=sorted(sols); out=[]; 
    return sols
out=[]
for root,idx in comp.items():
    if len(idx)<2: continue
    pts=[Y[i] for i in idx]
    cl=maxclique(idx)
    e=[p['easting'] for p in pts]; nn=[p['northing'] for p in pts]; z=[p['elev_geoid'] for p in pts]
    cx,cy=np.mean(e),np.mean(nn)
    # mid-drive / interpolated points within 10 m of any member
    mids=[m for m in M if min(math.hypot(m['easting']-p['easting'],m['northing']-p['northing']) for p in pts)<=R]
    diam=max(D[i,j] for i in idx for j in idx)
    out.append(dict(n=len(idx),core=len(cl),core_sols=sorted(Y[i]['sol'] for i in cl),
        sols=sorted(p['sol'] for p in pts), rmc=[p['RMC'] for p in sorted(pts,key=lambda p:p['sol'])],
        sites=sorted(set(p['site'] for p in pts)),
        lon=float(np.mean([p['lon'] for p in pts])),lat=float(np.mean([p['lat'] for p in pts])),elev=float(np.mean(z)),
        ns=max(nn)-min(nn),ew=max(e)-min(e),dz=max(z)-min(z),diam=float(diam),
        res=dict(collections.Counter(p['res'] for p in pts)),
        nmid=len(mids),mid_sols=sorted(set(m['sol'] for m in mids)),
        e=float(cx),nrt=float(cy)))
out.sort(key=lambda c:c['sols'][0])
json.dump(out,open('clusters.json','w'),indent=1)
print('end-of-drive (360) waypoints:',n,' clusters>=2:',len(out))
print('chain>=4:',sum(c['n']>=4 for c in out),' strict clique>=4:',sum(c['core']>=4 for c in out))
for c in out:
    if c['n']>=3:
        print(f"n={c['n']:2d} core={c['core']:2d} sols={c['sols']} NS={c['ns']:.1f} EW={c['ew']:.1f} dz={c['dz']:.1f} diam={c['diam']:.1f} res={c['res']} mid={c['nmid']}{c['mid_sols']} lon={c['lon']:.5f} lat={c['lat']:.5f} z={c['elev']:.1f}")
