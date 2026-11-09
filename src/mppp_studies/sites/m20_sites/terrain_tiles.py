import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from adjustText import adjust_text
import qmesh
from refdata import PANOS, SAMPLES
d=json.load(open('M20_waypoints.json')); P=[x['properties'] for x in d['features']]
lon=np.array([p['lon'] for p in P]); lat=np.array([p['lat'] for p in P]); S=np.array([p['sol'] for p in P])
isE=np.array([p['final']=='y' for p in P])
big=json.load(open('clusters_big.json')); rows={r['id']:r for r in json.load(open('table_rows.json'))}
def tiles(levels,box):
    U=[];V=[];H=[]
    for z in levels:
        deg=180/2**z
        for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'terrain', f'{z}_*.terrain'))):   # sorted: the same grid on every OS
            _,x,y=map(int,os.path.basename(fn)[:-8].split('_'))
            u,v,h=qmesh.read(fn)
            lo=-180+x*deg+u*deg; la=-90+y*deg+v*deg
            m=(lo>box[0]-deg)&(lo<box[1]+deg)&(la>box[2]-deg)&(la<box[3]+deg)
            U.append(lo[m]);V.append(la[m]);H.append(h[m])
    U=np.concatenate(U);V=np.concatenate(V);H=np.concatenate(H)
    # de-duplicate shared tile edges
    k=np.round(U*1e5).astype(np.int64)*10**8+np.round(V*1e5).astype(np.int64)
    _,i=np.unique(k,return_index=True); return U[i],V[i],H[i]
def dem(levels,box,step):
    U,V,H=tiles(levels,box)
    gx=np.arange(box[0],box[1],step); gy=np.arange(box[2],box[3],step)
    GX,GY=np.meshgrid(gx,gy)
    Z=griddata((U,V),H,(GX,GY),method='linear')
    return gx,gy,Z
def shade(gx,gy,Z,ax,vmin=None,vmax=None):
    dx=np.deg2rad(gx[1]-gx[0])*3389500*np.cos(np.deg2rad(gy.mean())); dy=np.deg2rad(gy[1]-gy[0])*3389500
    Zf=np.where(np.isnan(Z),np.nanmean(Z),Z)
    ls=LightSource(azdeg=315,altdeg=40)
    hs=ls.hillshade(Zf[::-1],vert_exag=3,dx=dx,dy=dy)[::-1]
    cmap=plt.get_cmap('gist_earth')
    norm=plt.Normalize(vmin if vmin is not None else np.nanpercentile(Z,1),vmax if vmax is not None else np.nanpercentile(Z,99))
    rgb=cmap(norm(Zf))[...,:3]
    rgb=0.55*rgb+0.45*np.dstack([np.full_like(Zf,0.86)]*3)   # mute the tint
    out=rgb*(0.35+0.65*hs[...,None])
    out=np.where(np.isnan(Z)[...,None],0.93,out)
    ax.imshow(out,extent=[gx[0],gx[-1],gy[0],gy[-1]],origin='lower',interpolation='bilinear')
    cs=ax.contour(gx,gy,Zf,levels=np.arange(-2700,-1000,100),colors='k',linewidths=0.3,alpha=0.25)
    return norm,cmap
