import json, textwrap, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from refdata import PANOS, SAMPLES
from sitesdef import P, X, Y, S, F, HALF, fmt_runs, fmt_counts
from dem import LocalFrame, sample_dem, sample_ortho, ortho_stretch, background_mode, hillshade, dem_paths, save_figure
out_png = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs/m20_cluster_zooms.png'
sites = json.load(open('sites_zoom.json'))
LF = LocalFrame(P)
C1,C2,C3,C4='#2a78d6','#eb6834','#1baf7a','#eda100'; INK='#0b0b0b'; INK2='#52514e'; MUT='#898781'
OUTG = '#9d9b96'     # 20 Oct: waypoints that are not the block's (not in its sols or within 10 m of them): grey
last_sol = int(S.max())
def posi(s):
    w = np.where(S <= s)[0]; return w[-1] if len(w) else 0
ncol = int(__import__('os').environ.get('M20_ZOOM_COLS', 6)); nrow = -(-(len(sites) + 1) // ncol)   # 6 columns (20 Oct): the blocks + the legend
CONTOURS = __import__('os').environ.get('M20_ZOOM_CONTOURS', '0') == '1'   # 20 Oct: no contour lines by default
assert len(sites) + 1 <= ncol*nrow
fig, axs = plt.subplots(nrow, ncol, figsize=(3.45*ncol, 3.6*nrow), facecolor='#fcfcfb')
dem_used = set(); ortho_used = set(); BACKGROUND = background_mode()   # M20_BACKGROUND: 'dtm' (default) or 'ortho'
# 20 Oct: the orthomosaic panels in colour with ONE gentle stretch for all blocks (was: grey, 0.5-99.5 % per panel).
# A colour mosaic is shown as it is; a grey one (the TRN mosaic) is tinted by the DTM's height about the block's
# median (gist_earth over +-ZOOM_TINT_M), so the panels read alike.
ZOOM_TINT_M = float(__import__('os').environ.get('M20_ZOOM_TINT_M', 3.0))
TINT = plt.get_cmap('gist_earth')
g = np.linspace(-HALF, HALF, 121); GE, GN = np.meshgrid(g, g)
go = np.linspace(-HALF, HALF, 241); GEo, GNo = np.meshgrid(go, go)
raw = {}
if BACKGROUND == 'ortho':          # orthomosaic at ~0.25 m: sample a finer grid than the DTM; raw values first
    for o in sites:
        img, oused = sample_ortho(*LF.to_lonlat(GEo + o['cx'], GNo + o['cy']), stretch=None)
        if np.isfinite(img).any():
            raw[o['id']] = img; ortho_used.update(oused)
STRETCH = ortho_stretch(list(raw.values())) if raw else None
for k, (ax, o) in enumerate(zip(axs.flat, sites)):
    cx, cy = o['cx'], o['cy']
    # ---- DEM background (local metres, via lon/lat) ----
    lon, lat = LF.to_lonlat(GE + cx, GN + cy)
    z, used = sample_dem(lon, lat)
    shown = False
    if o['id'] in raw:
        lo, hi = STRETCH
        img = np.clip((raw[o['id']] - lo) / max(hi - lo, 1e-6), 0, 1)
        if img.ndim == 3:
            rgb = np.where(np.isfinite(img), img, 0.93)
        else:
            gimg = np.where(np.isfinite(img), img, 0.93)
            if np.isfinite(z).any():   # the height tint on the ortho grid (nearest DTM sample, 0.5 m)
                zr_ = z - np.nanmedian(z)
                zi = zr_[np.clip(np.round((GNo + HALF) / (g[1] - g[0])).astype(int), 0, len(g) - 1),
                         np.clip(np.round((GEo + HALF) / (g[1] - g[0])).astype(int), 0, len(g) - 1)]
                tint = TINT(np.clip(0.15 + 0.7 * (np.nan_to_num(zi) + ZOOM_TINT_M) / (2 * ZOOM_TINT_M), 0, 1))[..., :3]
                hue = tint / np.maximum(tint.mean(axis=-1, keepdims=True), 1e-3)   # the tint's colour at unit brightness
                rgb = np.clip(gimg[..., None] * (0.6 + 0.4 * hue), 0, 1)
            else:
                rgb = np.repeat(gimg[..., None], 3, axis=-1)
        ax.imshow(rgb, extent=[-HALF, HALF, -HALF, HALF], origin='lower', interpolation='bilinear', zorder=0)
        shown = True
    if np.isfinite(z).any():
        dem_used.update(used)
        hs = hillshade(z, g[1]-g[0], g[1]-g[0])
        zr = z - np.nanmedian(z)
        if not shown:
            ax.imshow(hs, extent=[-HALF, HALF, -HALF, HALF], origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.9, zorder=0)
        lev = np.arange(np.floor(np.nanmin(zr)), np.ceil(np.nanmax(zr)) + 0.5, 0.5)
        if CONTOURS and len(lev) > 1:
            cs = ax.contour(g, g, zr, levels=lev, colors='#7a5c3a', linewidths=0.4, alpha=0.6, zorder=1)
    # ---- waypoints in the square ----
    inb = np.where((np.abs(X - cx) <= HALF) & (np.abs(Y - cy) <= HALF))[0]
    # 20 Oct: the block's own waypoints (its primary sols + every waypoint within 10 m, from_sites.block_members) in
    # colour; the other waypoints in the square in grey - not in the block, nor in its N and Z counts
    mem = set(o.get('members') or inb.tolist())
    own = np.array([i for i in inb if i in mem], int); oth = np.array([i for i in inb if i not in mem], int)
    # traverse polyline (all waypoints, clipped by axes)
    ax.plot(X - cx, Y - cy, color=MUT, lw=0.8, zorder=2)
    for i in own:
        if F[i] == 'y':
            ax.add_patch(plt.Circle((X[i]-cx, Y[i]-cy), 10, fill=False, ec=C2, lw=1.0, alpha=0.45, zorder=2))
    for ids, col, ec_, z in ((oth, OUTG, OUTG, 3), (own, C1, C1, 4)):
        if not len(ids): continue
        e = ids[F[ids] == 'y']; m = ids[F[ids] == 'm']; it = ids[F[ids] == 'i']
        ax.scatter(X[e]-cx, Y[e]-cy, s=42, color=col, ec='white', lw=0.8, zorder=z)
        ax.scatter(X[m]-cx, Y[m]-cy, marker='s', s=20, facecolor='white', ec=ec_, lw=1.0, zorder=z)
        ax.scatter(X[it]-cx, Y[it]-cy, marker='x', s=18, color=INK2 if col == C1 else OUTG, lw=0.9, zorder=z)
    for i in inb:
        ax.annotate(str(S[i]), (X[i]-cx, Y[i]-cy), xytext=(3, 2.5), textcoords='offset points', fontsize=7.2,
                    color=(INK if F[i] == 'y' else INK2) if i in mem else OUTG, zorder=6)
    for num, ss, _, _ in PANOS:
        i = posi(ss[0])
        if abs(X[i]-cx) <= HALF and abs(Y[i]-cy) <= HALF:
            ax.scatter(X[i]-cx, Y[i]-cy, marker='^', s=74, facecolor='none', ec=C4 if i in mem else OUTG, lw=1.6, zorder=5)
    for _, name, loc, s in SAMPLES:
        i = posi(s)
        if abs(X[i]-cx) <= HALF and abs(Y[i]-cy) <= HALF:
            ax.scatter(X[i]-cx, Y[i]-cy, marker='D', s=44, facecolor='none', ec=C3 if i in mem else OUTG, lw=1.6, zorder=5)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect('equal')
    ax.set_xticks([-30, -15, 0, 15, 30]); ax.set_yticks([-30, -15, 0, 15, 30])
    runs = o['runs']; lab = fmt_runs(runs)
    if runs[-1][1] >= last_sol: lab += '+'
    title = f"{o['id']}. {o['name']}"
    ax.set_title(title + '\n' + '\n'.join(textwrap.wrap('sols ' + lab, 40)), fontsize=9.6, loc='left', color=INK)
    ax.text(0.98, 0.02, fmt_counts(o), transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2, color=INK2,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75), zorder=7)
    if k % ncol == 0: ax.set_ylabel('Relative northing [m]', fontsize=9.5, color=INK)          # left-most column
    if k + ncol >= len(sites): ax.set_xlabel('Relative easting [m]', fontsize=9.5, color=INK)  # bottom-most panel in each column
    ax.tick_params(labelsize=7.8, colors=INK2)
    for sp in ax.spines.values(): sp.set_color(C2); sp.set_linewidth(1.6)   # orange border = block of rover stations
rest = list(axs.flat)[len(sites):]
for ax in rest: ax.axis('off')
lg = rest[0] if rest else axs.flat[-1]
lg.scatter([], [], s=42, color=C1, ec='white', label='End-of-drive waypoint: Navcam 360')
lg.scatter([], [], marker='s', s=20, facecolor='white', ec=C1, lw=1.0, label='Mid-drive waypoint: partial Navcam')
lg.scatter([], [], marker='x', s=18, color=INK2, label='Interpolated position (no images)')
lg.plot([], [], color=MUT, lw=0.8, label='Traverse')
lg.plot([], [], marker='o', ms=12, mfc='none', mec=C2, mew=1.0, ls='none', alpha=0.6, label='10 m radius around each 360 of the block')
lg.scatter([], [], s=42, color=OUTG, ec='white', lw=0.8, label='Waypoint not in the block (not counted)')
lg.scatter([], [], marker='D', s=44, facecolor='none', ec=C3, lw=1.6, label='Sample core (rover position)')
lg.scatter([], [], marker='^', s=74, facecolor='none', ec=C4, lw=1.6, label='Mastcam-Z numbered 360 (rover position)')
if dem_used and CONTOURS:
    lg.plot([], [], color='#7a5c3a', lw=0.6, label='DEM contours, 0.5 m')
lg.legend(loc='center', fontsize=9.6, frameon=False)
bg = 'HiRISE DTM hillshade' if dem_used else 'no 1 m DTM background (set M20_DEMS to the HiRISE DTM files)'
if ortho_used:
    bg = ('HiRISE orthomosaic' + (' (one stretch for all panels)' if STRETCH else '')
          + (f', tinted by the DTM height about each block\'s median (±{ZOOM_TINT_M:g} m)' if dem_used and raw and next(iter(raw.values())).ndim == 2 else '')
          + ('; 0.5 m DTM contours' if dem_used and CONTOURS else ''))
fig.suptitle_text = (f'M2020 blocks of rover stations, ordered by first sol, ±30 m around each block centre: the block\'s waypoints (its sols and every waypoint within 10 m of them) in colour, other waypoints in grey; N× Navcam 360s and Z× Mastcam-Z 360s of the block (local metres, north up; '
             f'labels = sol; title sols = the block\'s sols). {bg}.')
fig.suptitle('\n'.join(textwrap.wrap(fig.suptitle_text, 140)), x=0.01, ha='left', fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975))
out_png = save_figure(fig, out_png, dpi=int(__import__('os').environ.get('M20_DPI', 300)), facecolor=fig.get_facecolor())
print('saved', out_png, 'DEM:', sorted(dem_used) or 'none')
