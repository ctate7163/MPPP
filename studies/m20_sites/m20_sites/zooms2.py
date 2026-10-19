import json, textwrap, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from refdata import PANOS, SAMPLES
from sitesdef import P, X, Y, S, F, HALF, fmt_runs, fmt_counts
from dem import LocalFrame, sample_dem, hillshade, dem_paths
out_png = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs/m20_cluster_zooms.png'
sites = json.load(open('sites_zoom.json'))
LF = LocalFrame(P)
C1,C2,C3,C4='#2a78d6','#eb6834','#1baf7a','#eda100'; INK='#0b0b0b'; INK2='#52514e'; MUT='#898781'
last_sol = int(S.max())
def posi(s):
    w = np.where(S <= s)[0]; return w[-1] if len(w) else 0
ncol = 5; nrow = -(-(len(sites) + 1) // ncol)   # 5 columns: the blocks + the legend (34 blocks: 7 rows, 36: 8)
assert len(sites) + 1 <= ncol*nrow
fig, axs = plt.subplots(nrow, ncol, figsize=(3.45*ncol, 3.6*nrow), facecolor='#fcfcfb')
dem_used = set()
for k, (ax, o) in enumerate(zip(axs.flat, sites)):
    cx, cy = o['cx'], o['cy']
    # ---- DEM background (local metres, via lon/lat) ----
    g = np.linspace(-HALF, HALF, 121)
    GE, GN = np.meshgrid(g, g)
    lon, lat = LF.to_lonlat(GE + cx, GN + cy)
    z, used = sample_dem(lon, lat)
    if np.isfinite(z).any():
        dem_used.update(used)
        hs = hillshade(z, g[1]-g[0], g[1]-g[0])
        zr = z - np.nanmedian(z)
        ax.imshow(hs, extent=[-HALF, HALF, -HALF, HALF], origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.9, zorder=0)
        lev = np.arange(np.floor(np.nanmin(zr)), np.ceil(np.nanmax(zr)) + 0.5, 0.5)
        if len(lev) > 1:
            cs = ax.contour(g, g, zr, levels=lev, colors='#7a5c3a', linewidths=0.4, alpha=0.6, zorder=1)
    # ---- waypoints in the square ----
    inb = np.where((np.abs(X - cx) <= HALF) & (np.abs(Y - cy) <= HALF))[0]
    # traverse polyline (all waypoints, clipped by axes)
    ax.plot(X - cx, Y - cy, color=MUT, lw=0.8, zorder=2)
    for i in inb:
        if F[i] == 'y':
            ax.add_patch(plt.Circle((X[i]-cx, Y[i]-cy), 10, fill=False, ec=C2, lw=0.45, alpha=0.3, zorder=2))
    e = inb[F[inb] == 'y']; m = inb[F[inb] == 'm']; it = inb[F[inb] == 'i']
    ax.scatter(X[e]-cx, Y[e]-cy, s=42, color=C1, ec='white', lw=0.8, zorder=4)
    ax.scatter(X[m]-cx, Y[m]-cy, marker='s', s=20, facecolor='white', ec=C1, lw=1.0, zorder=4)
    ax.scatter(X[it]-cx, Y[it]-cy, marker='x', s=18, color=INK2, lw=0.9, zorder=4)
    for i in inb:
        ax.annotate(str(S[i]), (X[i]-cx, Y[i]-cy), xytext=(3, 2.5), textcoords='offset points', fontsize=7.2,
                    color=INK if F[i] == 'y' else INK2, zorder=6)
    for num, ss, _, _ in PANOS:
        i = posi(ss[0])
        if abs(X[i]-cx) <= HALF and abs(Y[i]-cy) <= HALF:
            ax.scatter(X[i]-cx, Y[i]-cy, marker='^', s=74, facecolor='none', ec=C4, lw=1.6, zorder=5)
    for _, name, loc, s in SAMPLES:
        i = posi(s)
        if abs(X[i]-cx) <= HALF and abs(Y[i]-cy) <= HALF:
            ax.scatter(X[i]-cx, Y[i]-cy, marker='D', s=44, facecolor='none', ec=C3, lw=1.6, zorder=5)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect('equal')
    ax.set_xticks([-30, -15, 0, 15, 30]); ax.set_yticks([-30, -15, 0, 15, 30])
    runs = o['runs']; lab = fmt_runs(runs)
    if runs[-1][1] >= last_sol: lab += '+'
    title = f"{o['id']}. {o['name']}"
    ax.set_title(title + '\n' + '\n'.join(textwrap.wrap('sols ' + lab, 40)), fontsize=9.6, loc='left', color=INK)
    ax.text(0.98, 0.02, f"{fmt_counts(o)}  ·  {o['npart']} partial", transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2, color=INK2,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=7)
    if k % ncol == 0: ax.set_ylabel('Relative northing [m]', fontsize=9.5, color=INK)          # left-most column
    if k + ncol >= len(sites): ax.set_xlabel('Relative easting [m]', fontsize=9.5, color=INK)  # bottom-most panel in each column
    ax.tick_params(labelsize=7.8, colors=INK2); ax.grid(color='#e4e3df', lw=0.4, zorder=0)
    for sp in ax.spines.values(): sp.set_color(C2); sp.set_linewidth(1.6)   # orange border = block of rover stations
rest = list(axs.flat)[len(sites):]
for ax in rest: ax.axis('off')
lg = rest[0] if rest else axs.flat[-1]
lg.scatter([], [], s=42, color=C1, ec='white', label='End-of-drive waypoint: Navcam 360')
lg.scatter([], [], marker='s', s=20, facecolor='white', ec=C1, lw=1.0, label='Mid-drive waypoint: partial Navcam')
lg.scatter([], [], marker='x', s=18, color=INK2, label='Interpolated position (no images)')
lg.plot([], [], color=MUT, lw=0.8, label='Traverse')
lg.plot([], [], marker='o', ms=12, mfc='none', mec=C2, ls='none', alpha=0.5, label='10 m radius around each 360')
lg.scatter([], [], marker='D', s=44, facecolor='none', ec=C3, lw=1.6, label='Sample core (rover position)')
lg.scatter([], [], marker='^', s=74, facecolor='none', ec=C4, lw=1.6, label='Mastcam-Z numbered 360 (rover position)')
if dem_used:
    lg.plot([], [], color='#7a5c3a', lw=0.6, label='DEM contours, 0.5 m')
lg.legend(loc='center', fontsize=9.6, frameon=False)
bg = ('HiRISE DTM hillshade: ' + ', '.join(sorted(dem_used))) if dem_used else 'no DEM background (DTM file not available to this run)'
fig.suptitle_text = (f'M2020 blocks of rover stations, ordered by first sol: all waypoints within ±30 m of each block centre (N× Navcam 360s, Z× Mastcam-Z 360s) (local metres, north up; '
             f'labels = sol; title sols = sols the rover ended inside the square). {bg}.')
fig.suptitle('\n'.join(textwrap.wrap(fig.suptitle_text, 140)), x=0.01, ha='left', fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(out_png, dpi=int(__import__('os').environ.get('M20_DPI', 300)), facecolor=fig.get_facecolor())
print('saved', out_png, 'DEM:', sorted(dem_used) or 'none')
