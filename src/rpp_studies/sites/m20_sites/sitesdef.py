import json, numpy as np
P=[f['properties'] for f in json.load(open('M20_waypoints.json'))['features']]
X=np.array([p['easting'] for p in P]); Y=np.array([p['northing'] for p in P]); S=np.array([p['sol'] for p in P]); F=np.array([p['final'] for p in P])
sites=json.load(open('sites.json'))['sites']
rows={r['id']:r for r in json.load(open('table_rows.json'))}
big=json.load(open('clusters_big.json')); rmc={p['RMC']:i for i,p in enumerate(P)}
HALF=30.0
from refdata import PANOS
def _pos(s):
    w=np.where(S<=s)[0]; return w[-1] if len(w) else 0
CLUSTER_NAMES={1:'OEB Landing Site',2:'Van Zyl Overlook',3:'Artuby Ridge',4:'Issole, South Seitah',5:'Brac, Seitah North',
 6:'Enchanted Lake',7:'Delta front ascent',8:'Rockytop',9:'Knob Mountain',10:'Three Forks South',
 11:'Belva Crater',12:'Tuxedo Park',13:'Bunsen Peak',14:'Cheyava Falls',15:'Pico Turquino',16:'South Arm',
 17:'Witch Hazel Hill',18:'Bell Island',23:'Hippo Pools',19:'Taylorfjellet',21:'Origny',22:'Vingeanne'}
EXTRA_WINDOWS={'wild_basin_lake':(838,853),'plateau_point':(1252,1263),
 'observation_mountain':(592,600),'whale_mountain':(606,610)}   # user-defined sol windows (17 Oct: Observation and Whale Mountain, Hidden Harbor)
REMOVED_CLUSTERS={20}      # 20 unnamed 1756-1761 dropped
RUN_EXCLUDE={'threeforks_north':(413,)}   # single-sol pass-throughs dropped from the listed sol ranges
CENTRE_OFFSETS={'cluster12':(-20.0,0.0)}   # Tuxedo Park shifted 20 m (west: the sol 905 and 907 stops lie 33-36 m west of the cluster mean) so they fall inside the square
ADDED={'tenby_berea':'Tenby / Berea','wild_basin_lake':'Wild Basin Lake','plateau_point':'Plateau Point','rochette':'Rochette','threeforks_north':'Three Forks North','chal_rocks_large':'Chal Rocks','airey_hill':'Airey Hill',
 'overlook_mountain':'Overlook Mountain','bright_angle':'Bright Angel','rio_chiquito':'Rio Chiquito','groloy':'Groloy',
'marble_mountain':'Marble Mountain','observation_mountain':'Observation Mountain','whale_mountain':'Whale Mountain'}
# 17 Oct (Christian's edits of blk08 / blk09 in sites.json): these visits replace the computed residency runs
VISIT_EDITS={'cluster10':[(413,414),(652,683)],'threeforks_north':[(413,413),(433,435),(663,666),(670,673),(678,692)]}
def _first_range(sols):
    return tuple(sols[0]) if isinstance(sols[0],list) else tuple(sols)
E=np.where(F=='y')[0]
def residency(cx,cy,half=HALF):
    """Sols on which the rover ended the sol inside the square, as discrete continuous ranges."""
    sols=np.arange(S.min(),S.max()+1); inside=[]
    for s in sols:
        i=np.where(S<=s)[0][-1]
        inside.append(abs(X[i]-cx)<=half and abs(Y[i]-cy)<=half)
    runs=[];st=None
    for s,v in zip(sols,inside):
        if v and st is None: st=s
        if not v and st is not None: runs.append((st,s-1)); st=None
    if st is not None: runs.append((st,sols[-1]))
    return runs
def fmt_counts(o, pad=False):
    n = f"{o['n360']:>2}" if pad else f"{o['n360']}"          # pad=True right-aligns one-digit counts under N×10
    return f"N×{n}" + (f", Z×{o['nz']}" if o.get('nz') else '')
def fmt_runs(runs): return ', '.join(f"{a}" if a==b else f"{a}–{b}" for a,b in runs)
def build():
    out=[]
    for c in big:
        if c['id'] in REMOVED_CLUSTERS: continue
        m=[rmc[r] for r in c['rmc']]
        dx,dy=CENTRE_OFFSETS.get(f"cluster{c['id']}",(0.0,0.0))
        out.append(dict(key=f"cluster{c['id']}",old_id=c['id'],name=CLUSTER_NAMES[c['id']],cx=float(X[m].mean()+dx),cy=float(Y[m].mean()+dy),kind='cluster'))
    for k,nm in ADDED.items():
        a,b=EXTRA_WINDOWS.get(k) or _first_range(sites[k]['sols']); idx=[i for i in E if a<=S[i]<=b]
        xs,ys=X[idx],Y[idx]
        if max(np.hypot(xs[:,None]-xs,ys[:,None]-ys).ravel())>40:   # spread too wide for one 60 m panel: centre on the densest stop
            dens=[np.sum(np.hypot(X[E]-X[i],Y[E]-Y[i])<=30) for i in idx]; i0=idx[int(np.argmax(dens))]
            sel=[i for i in idx if np.hypot(X[i]-X[i0],Y[i]-Y[i0])<=30]; xs,ys=X[sel],Y[sel]
        out.append(dict(key=k,old_id=None,name=nm,cx=float(xs.mean()),cy=float(ys.mean()),kind='site'))
    for o in out:
        allruns=[(int(a),int(b)) for a,b in residency(o['cx'],o['cy'])]
        o['first_all']=allruns[0][0]          # numbering keeps using every sol, so dropping a listed sol does not renumber blocks
        o['runs']=[r for r in allruns if not (r[0]==r[1] and r[0] in RUN_EXCLUDE.get(o['key'],()))]
        if o['key'] in VISIT_EDITS: o['runs']=[tuple(r) for r in VISIT_EDITS[o['key']]]; o['visits_edited']=True
        o['first']=o['first_all']
        inbox=[i for i in range(len(P)) if abs(X[i]-o['cx'])<=HALF and abs(Y[i]-o['cy'])<=HALF]
        o['n360']=int(sum(F[i]=='y' for i in inbox))
        o['zpanos']=[num for num,ss,_,_ in PANOS if any(abs(X[_pos(t)]-o['cx'])<=HALF and abs(Y[_pos(t)]-o['cy'])<=HALF for t in ss)]
        o['nz']=len(o['zpanos']); o['npart']=int(sum(F[i]=='m' for i in inbox)); o['ninterp']=int(sum(F[i]=='i' for i in inbox))
    out.sort(key=lambda o:o['first'])
    for k,o in enumerate(out,1): o['id']=k
    return out
if __name__=='__main__':
    out=build(); json.dump(out,open('sites_zoom.json','w'),indent=1)
    for o in out: print(o['id'],o['name'],o['key'],'| sols',fmt_runs(o['runs']),'| 360s',o['n360'],'partial',o['npart'],'interp',o['ninterp'])
