"""Traverse map (figure 2: A-C) and the expanded version with regional insets (figure 3: A-H).
Background: HiRISE DTM where a file is available (see dem.py), otherwise the coarse HiRISE+MOLA terrain tiles (~70 m)."""
import json, sys, textwrap, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from adjustText import adjust_text
from terrain_tiles import dem as tile_dem
from refdata import PANOS, SAMPLES
from sitesdef import P, X, Y, S, F, HALF, fmt_runs, fmt_counts
from dem import LocalFrame, sample_dem
sites = json.load(open('sites_zoom.json')); LF = LocalFrame(P)
lon = np.array([p['lon'] for p in P]); lat = np.array([p['lat'] for p in P]); isE = F == 'y'
C1,C2,C3,C4='#2a78d6','#eb6834','#1baf7a','#eda100'; INK='#0b0b0b'; INK2='#52514e'
COS = np.cos(np.deg2rad(18.45)); R = 3389500.0
M_LON = np.deg2rad(1)*R*COS; M_LAT = np.deg2rad(1)*R           # metres per degree
CMAP = plt.get_cmap('gist_earth')
BOX_A = (77.1, 78.1, 17.9, 18.8)
BOX_B = (77.205, 77.47, 18.38, 18.515)
used_all = set()
import os
import matplotlib.patheffects as pe
DPI = int(os.environ.get('M20_DPI', 300))     # output resolution of every figure (set M20_DPI=300 for smaller files)
FS = 1.15                                   # global font scale
plt.rcParams.update({'font.size': 10*FS})

def site_ll(o): return LF.to_lonlat(o['cx'], o['cy'])
def box_around(ids, size_m=None, pad_m=150):
    ll = np.array([site_ll(o) for o in sites if o['id'] in ids])
    lo0, lo1, la0, la1 = ll[:,0].min(), ll[:,0].max(), ll[:,1].min(), ll[:,1].max()
    clo, cla = (lo0+lo1)/2, (la0+la1)/2
    if size_m:  # fixed box in metres: (width, height)
        hw, hh = size_m[0]/2/M_LON, size_m[1]/2/M_LAT
    else:
        hw = (lo1-lo0)/2 + pad_m/M_LON; hh = (la1-la0)/2 + pad_m/M_LAT
    return (clo-hw, clo+hw, cla-hh, cla+hh)

def terrain(box, step, tile_levels):
    gx, gy, Z = tile_dem(tile_levels, box, step)
    GX, GY = np.meshgrid(gx, gy)
    Zd, used = sample_dem(GX, GY)
    used_all.update(used)
    if np.isfinite(Zd).any(): Z = np.where(np.isfinite(Zd), Zd, Z)
    return gx, gy, Z, bool(used)

def draw_terrain(ax, gx, gy, Z, norm, ci, exag=3, alt=40, flat=None, coslat=None):
    """Tinted hillshade. exag = vertical exaggeration. flat: if set, rescale the shading so level ground has this brightness
    (1.0 = as bright as the unshaded tint), which keeps a strongly exaggerated relief from darkening the whole panel.
    coslat: east-west metres-per-degree factor for this grid (default: Jezero's latitude)."""
    dx = np.deg2rad(gx[1]-gx[0])*R*(COS if coslat is None else coslat); dy = np.deg2rad(gy[1]-gy[0])*R
    Zf = np.where(np.isnan(Z), np.nanmean(Z), Z)
    hs = LightSource(azdeg=315, altdeg=alt).hillshade(Zf[::-1], vert_exag=exag, dx=dx, dy=dy)[::-1]
    shade = 0.35 + 0.65*hs
    if flat is not None: shade = np.clip(shade*flat/(0.35 + 0.65*np.sin(np.deg2rad(alt))), 0, 1.15)
    rgb = CMAP(norm(Zf))[..., :3]; rgb = 0.55*rgb + 0.45*0.86
    out = np.clip(rgb*shade[..., None], 0, 1); out = np.where(np.isnan(Z)[..., None], 0.93, out)
    ax.imshow(out, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin='lower', interpolation='bilinear', zorder=0)
    lev = np.arange(np.floor(np.nanmin(Zf)/ci)*ci, np.nanmax(Zf)+ci, ci)
    ax.contour(gx, gy, Zf, levels=lev, colors='k', linewidths=0.3, alpha=0.25, zorder=1)

def scalebar(ax, box, metres, label, where='left'):
    dl = metres/M_LON; y0 = box[2] + 0.06*(box[3]-box[2])
    x0 = box[0] + 0.05*(box[1]-box[0]) if where == 'left' else box[1] - 0.05*(box[1]-box[0]) - dl
    ax.plot([x0, x0+dl], [y0, y0], color=INK, lw=3, zorder=9, solid_capstyle='butt')
    ax.text(x0+dl/2, y0 + 0.025*(box[3]-box[2]), label, ha='center', fontsize=8*FS, zorder=9,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))

def posll(s):
    w = np.where(S <= s)[0]; i = w[-1] if len(w) else 0; return lon[i], lat[i]
ZP = np.array([posll(ss[0]) for _, ss, _, _ in PANOS]); SP = np.array([posll(s) for *_, s in SAMPLES])

def overlay(ax, box, ids=None, eod=False, squares=False, ms=150, fs=8.5*FS, label_kw=None):
    ax.plot(lon, lat, color='white', lw=3.0, zorder=2); ax.plot(lon, lat, color=C1, lw=1.4, zorder=3)
    if eod:
        ax.scatter(lon[isE], lat[isE], s=10, color='white', ec=C1, lw=0.7, zorder=4)
        m = F == 'm'; ax.scatter(lon[m], lat[m], s=8, marker='s', color='white', ec=C1, lw=0.6, zorder=4)
    ax.scatter(ZP[:,0], ZP[:,1], marker='^', s=46, color=C4, ec=INK, lw=0.5, zorder=7)
    ax.scatter(SP[:,0], SP[:,1], marker='D', s=30, color=C3, ec=INK, lw=0.5, zorder=7)
    txt = []
    for o in sites:
        if ids is not None and o['id'] not in ids: continue
        lo, la = site_ll(o)
        if not (box[0] <= lo <= box[1] and box[2] <= la <= box[3]): continue
        if squares:
            lo0, la0 = LF.to_lonlat(o['cx']-HALF, o['cy']-HALF); lo1, la1 = LF.to_lonlat(o['cx']+HALF, o['cy']+HALF)
            ax.add_patch(plt.Rectangle((lo0, la0), lo1-lo0, la1-la0, fill=False, ec=C2, lw=1.6, zorder=6))
        else:
            ax.scatter(lo, la, s=ms, marker='s', facecolor='none', ec=C2, lw=2.0, zorder=6)
        txt.append(ax.text(lo, la, str(o['id']), fontsize=fs, fontweight='bold', color=INK, zorder=8,
                   bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=C2, lw=0.8, alpha=0.92)))
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect(1/COS)
    adjust_text(txt, ax=ax, expand=(1.35, 1.6), arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6), max_move=(60, 60))

def panel_A(ax, letter='A', box_label='B', bbox=None, crater=None, scalebar_right=False, names=False):
    bbox = bbox or BOX_B; CA = crater or BOX_A
    gx, gy, Z = tile_dem([8, 9, 10, 11], (CA[0]-0.02, CA[1]+0.02, CA[2]-0.02, CA[3]+0.02), 0.003)
    nA = plt.Normalize(np.nanpercentile(Z, 1), np.nanpercentile(Z, 99))
    draw_terrain(ax, gx, gy, Z, nA, 100)
    ax.plot(lon, lat, color='white', lw=3.2); ax.plot(lon, lat, color=C1, lw=1.6)
    ax.add_patch(plt.Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], fill=False, ec=INK, lw=1.2, zorder=5))
    ax.text(bbox[0]+0.01, bbox[3]+0.012, box_label, fontsize=11*FS, fontweight='bold', color=INK, zorder=6)
    ax.plot(lon[0], lat[0], marker='*', ms=13, color=C4, mec=INK, mew=0.6, zorder=7)   # above the black box
    if names:   # figure 5/6: landing site and crater named in white with a dark outline
        halo = [pe.withStroke(linewidth=2.2, foreground=INK)]
        ax.annotate('Octavia E. Butler\nLanding Site', (lon[0], lat[0]), xytext=(10, -4), textcoords='offset points', ha='left', va='top',
                    fontsize=9*FS, fontweight='bold', color='white', linespacing=1.1, zorder=8, path_effects=halo)
        ax.text(77.66, 18.24, 'Jezero crater', ha='center', va='center', fontsize=12*FS, fontweight='bold', style='italic',
                color='white', zorder=8, path_effects=halo)
    else:
        ax.annotate('Landing (OEB)', (lon[0], lat[0]), xytext=(8, -14), textcoords='offset points', fontsize=8*FS, color=INK)
    ax.set_xlim(CA[0], CA[1]); ax.set_ylim(CA[2], CA[3]); ax.set_aspect(1/COS)
    ax.set_title(f'{letter}. Jezero crater and the Perseverance traverse (sols {S.min()}–{S.max()})', loc='left', fontsize=10*FS, color=INK)
    ax.set_xlabel('Longitude [°E]'); ax.set_ylabel('Latitude [°N]')
    dl = 10000/M_LON; x0, y0 = (CA[1] - 0.05 - dl if scalebar_right else 77.15), CA[2] + 0.04
    ax.plot([x0, x0+dl], [y0, y0], color=INK, lw=3); ax.text(x0+dl/2, y0+0.015, '10 km', ha='center', fontsize=8*FS)

def panel_B(ax, insets=None, letter='B', bbox=None, bg_in_title=True, landing_label=False):
    bbox = bbox or BOX_B
    gx, gy, Z, hasdtm = terrain(bbox, 0.00005, [12, 13])
    nB = plt.Normalize(np.nanpercentile(Z, 1), np.nanpercentile(Z, 99)); draw_terrain(ax, gx, gy, Z, nB, 100)
    overlay(ax, bbox)
    if insets:
        for lab, b in insets.items():
            ax.add_patch(plt.Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], fill=False, ec=INK, lw=1.1, zorder=9))
            ax.text(b[0], b[3]+0.0012, lab, fontsize=10*FS, fontweight='bold', color=INK, zorder=10)
    h = [plt.Line2D([], [], color=C1, lw=1.4, label='Traverse'),
         plt.Line2D([], [], marker='^', color=C4, mec=INK, ls='none', ms=7, label='Mastcam-Z numbered 360'),
         plt.Line2D([], [], marker='D', color=C3, mec=INK, ls='none', ms=6, label='Sample core'),
         plt.Line2D([], [], marker='s', mfc='none', mec=C2, mew=2, ls='none', ms=11, label='Block of rover stations')]
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.6, 0.01), fontsize=7.6*FS, framealpha=0.92, facecolor='white', edgecolor='#d6d5d0')
    bg = 'HiRISE DTM where available, coarse HiRISE+MOLA terrain elsewhere' if hasdtm else 'coarse HiRISE-DTM-based terrain, ~70 m grid (HiRISE DTM not yet connected)'
    ax.set_title(f'{letter}. Blocks of rover stations' + (', numbered by first sol. Background: ' + bg if bg_in_title else ''), loc='left', fontsize=9.5*FS, color=INK)
    ax.set_xlabel('Longitude [°E]')
    scalebar(ax, bbox, 1000, '1 km')
    if landing_label:   # landing star, and its name in white just below the landing-site blocks (no leader line)
        ax.plot(lon[0], lat[0], marker='*', ms=12, color=C4, mec=INK, mew=0.6, zorder=12)
        ax.text(lon[0], bbox[2] + 0.255*(bbox[3]-bbox[2]), 'Octavia E. Butler\nLanding Site', ha='center', va='top',
                fontsize=9.5*FS, fontweight='bold', color='white', zorder=12, linespacing=1.1,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=INK)])
    return nB

def panel_C(ax, fs=7.2*FS, ncol=3, per=12, title=None):
    ax.axis('off'); lines = []
    NW = max(len(o['name']) for o in sites)
    for o in sites:
        lab = fmt_runs(o['runs']) + ('+' if o['runs'][-1][1] >= S.max() else '')
        lines.append(f"{o['id']:>2}  {o['name'][:34]:<{NW}} {fmt_counts(o, pad=True):<10} sols {lab}")
    for k in range(ncol):
        ax.text(k/ncol, 1.0, '\n'.join(lines[per*k:per*(k+1)]), va='top', ha='left', family='DejaVu Sans Mono', fontsize=fs, color=INK,
                transform=ax.transAxes, linespacing=1.25)
    ax.set_title(title or ('C. Blocks of rover stations: number, name, N× = end-of-drive Navcam 360s and Z× = numbered Mastcam-Z 360s '
                 'within ±30 m of the block centre, and the sols the rover ended inside that square'), loc='left', fontsize=10*FS, color=INK)

def colorbar(fig, rect, norm, label, extend='neither'):
    cax = fig.add_axes(rect); sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, extend=extend); cb.set_label(label, fontsize=8*FS); cb.ax.tick_params(labelsize=7*FS)

def ab_layout(W, top, h, left=0.85, gap=0.95, cbar=1.25, bbox=None, crater=None):
    bbox = bbox or BOX_B; CA = crater or BOX_A
    rA = (CA[1]-CA[0])*COS/(CA[3]-CA[2]); rB = (bbox[1]-bbox[0])*COS/(bbox[3]-bbox[2])
    hh = (W - left - gap - cbar)/(rA + rB) if h is None else h
    return (left, top-hh, rA*hh, hh), (left + rA*hh + gap, top-hh, rB*hh, hh), hh

def fig2(out):
    W = 18.0; Htot = 7.4
    rect = lambda x, y, w, h: [x/W, y/Htot, w/W, h/Htot]
    a, b, hh = ab_layout(W, Htot-0.45, None)
    fig = plt.figure(figsize=(W, Htot), facecolor='#fcfcfb')
    axA = fig.add_axes(rect(*a)); axB = fig.add_axes(rect(*b))
    panel_A(axA); nB = panel_B(axB)
    colorbar(fig, rect(b[0]+b[2]+0.12, b[1], 0.2, b[3]), nB, 'Elevation [m] (tint)')
    axC = fig.add_axes(rect(0.85, 0.1, W-1.2, a[1]-1.05)); panel_C(axC)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor()); print('saved', out)

REGION_KEYS = {
 'rimw':   ['cluster21', 'cluster22', 'groloy', 'cluster23', 'marble_mountain'],
 'whh':    ['cluster16', 'cluster17', 'cluster18', 'cluster19'],
 'margin': ['cluster13', 'overlook_mountain', 'bright_angle', 'cluster14', 'plateau_point', 'cluster15', 'rio_chiquito'],
 'fan':    ['tenby_berea', 'cluster11', 'wild_basin_lake', 'cluster12', 'airey_hill'],
 'delta':  ['cluster10', 'threeforks_north', 'cluster6', 'cluster7', 'cluster8', 'cluster9', 'observation_mountain', 'whale_mountain'],
 'land':   ['cluster1', 'cluster2', 'chal_rocks_large', 'cluster3', 'rochette', 'cluster4', 'cluster5']}
KEY2ID = {o['key']: o['id'] for o in sites}
RID = {r: sorted(KEY2ID[k] for k in ks) for r, ks in REGION_KEYS.items()}
def rng(r): return f"{RID[r][0]}–{RID[r][-1]}"
REGION_TITLE = {'rimw': 'Crater rim west: Origny to Marble Mountain', 'whh': 'Witch Hazel Hill: South Arm to Taylorfjellet',
                'margin': 'Margin unit to the rim: Bunsen Peak to Rio Chiquito', 'fan': 'Upper fan: Tenby to Airey Hill',
                'delta': 'Delta front: Three Forks to Knob Mountain', 'land': 'Crater floor: landing site and Séítah'}
GROUPS = {'D': RID['rimw'], 'E': RID['whh'], 'F': RID['delta'], 'G': RID['land'], 'H': RID['fan'], 'I': RID['margin']}

def fig3(out):
    for k, ids in GROUPS.items():
        names = {o['id'] for o in sites}; assert set(ids) <= names, k
    BW, BH = 1000.0, 1500.0                              # D-G boxes: 1 km wide, 1.5 km tall
    boxes = {k: box_around(ids, size_m=(BW, BH)) for k, ids in GROUPS.items() if k in 'DEFG'}
    W = 18.0
    hA = ab_layout(W, 0, None)[2]
    gap, cslot = 0.55, 1.45                               # gap between panels; colour-bar slot
    s = (W - 0.85 - 3*gap - 2*cslot - 0.2) / 4           # D-G panel width [in]
    sh = s*BH/BW                                          # D-G panel height [in]
    wHI = 2*s + gap                                       # H and I take the footprint of a D-E pair
    def fit_box(ids, w_in, h_in, pad_m=180):              # widen whichever side is short so the box matches the panel shape
        hb = box_around(ids, pad_m=pad_m)
        wm, hm = (hb[1]-hb[0])*M_LON, (hb[3]-hb[2])*M_LAT
        if wm/hm < w_in/h_in: wm = hm*w_in/h_in
        else: hm = wm*h_in/w_in
        cl, ca = (hb[0]+hb[1])/2, (hb[2]+hb[3])/2
        return (cl - wm/2/M_LON, cl + wm/2/M_LON, ca - hm/2/M_LAT, ca + hm/2/M_LAT)
    boxes['H'] = fit_box(GROUPS['H'], wHI, sh); boxes['I'] = fit_box(GROUPS['I'], wHI, sh)
    hC = 1.6
    Htot = 0.45 + hA + 0.75 + hC + 0.45 + sh + 0.85 + sh + 0.75
    rect = lambda x, y, w, h: [x/W, y/Htot, w/W, h/Htot]
    fig = plt.figure(figsize=(W, Htot), facecolor='#fcfcfb')
    y = Htot - 0.45
    a, b, _ = ab_layout(W, y, hA)
    axA = fig.add_axes(rect(*a)); axB = fig.add_axes(rect(*b))
    panel_A(axA); nB = panel_B(axB, insets={k: boxes[k] for k in 'DEFGHI'})
    colorbar(fig, rect(b[0]+b[2]+0.12, b[1], 0.2, b[3]), nB, 'Elevation [m] (tint)')
    y = b[1] - 0.75
    axC = fig.add_axes(rect(0.85, y-hC, W-1.2, hC)); panel_C(axC)
    y1 = y - hC - 0.45                 # top of row 1 (H, F, G)
    y2 = y1 - sh - 0.85                # top of row 2 (D, E, I)
    T = {k: terrain(boxes[k], 0.000015, [12, 13]) for k in 'DEFG'}
    T['H'] = terrain(boxes['H'], 0.00002, [12, 13]); T['I'] = terrain(boxes['I'], 0.00002, [12, 13])
    def norm_of(keys):
        z = np.concatenate([T[k][2][np.isfinite(T[k][2])].ravel() for k in keys]); return plt.Normalize(np.percentile(z, 1), np.percentile(z, 99))
    norms = {'D': norm_of('DE'), 'E': norm_of('DE'), 'F': norm_of('FG'), 'G': norm_of('FG'), 'H': norm_of('H'), 'I': norm_of('I')}
    rmap = {'D': 'rimw', 'E': 'whh', 'F': 'delta', 'G': 'land', 'H': 'fan', 'I': 'margin'}
    titles = {k: f"{k}. {REGION_TITLE[r]}" for k, r in rmap.items()}
    ci = {'D': 5, 'E': 5, 'F': 5, 'G': 5, 'H': 10, 'I': 10}
    bar = {'D': (200, '200 m'), 'E': (200, '200 m'), 'F': (200, '200 m'), 'G': (200, '200 m'), 'H': (500, '500 m'), 'I': (500, '500 m')}
    x0 = 0.85
    # (panel, x, y_top, width, colour bar after it?, colour-bar label)
    xF = x0 + wHI + cslot
    layout = [('H', x0, y1, wHI, 'Elevation [m], H'), ('F', xF, y1, s, None), ('G', xF + s + gap, y1, s, 'Elevation [m], F and G'),
              ('D', x0, y2, s, None), ('E', x0 + s + gap, y2, s, 'Elevation [m], D and E'), ('I', xF, y2, wHI, 'Elevation [m], I')]
    for k, x, yt, w, cl in layout:
        ax = fig.add_axes(rect(x, yt-sh, w, sh))
        gx, gy, Z, _ = T[k]; draw_terrain(ax, gx, gy, Z, norms[k], ci[k], exag=2)
        overlay(ax, boxes[k], ids=GROUPS[k], eod=True, squares=True, fs=8*FS)
        scalebar(ax, boxes[k], *bar[k])
        ax.set_title(titles[k], loc='left', fontsize=9.5*FS, color=INK)
        ax.tick_params(labelsize=6.5*FS); ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3 if w == s else 6)); ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        if cl: colorbar(fig, rect(x + w + 0.12, yt-sh, 0.18, sh), norms[k], cl)
    fig.text(0.85/W, 0.25/Htot, 'D–I: squares = blocks of rover stations (±30 m, as in the block panels); '
             'dots = end-of-drive Navcam 360; small squares = mid-drive partial Navcam; triangles = Mastcam-Z numbered 360s; diamonds = sample cores.',
             fontsize=8.5*FS, color=INK2)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor()); print('saved', out, 'DTM files:', sorted(used_all) or 'none')

def fig4(out):
    """Wide layout: row 1 = A | B | C (one-column list); row 2 = six regional panels in reverse chronological order, D-I left to right."""
    order = ['rimw', 'whh', 'margin', 'fan', 'delta', 'land']           # reverse chronological
    letters = dict(zip(order, 'DEFGHI'))
    skinny = {'rimw', 'whh', 'delta', 'land'}
    BH = 2400.0                                                          # every regional panel 2.4 km tall (same scale across the row)
    BW_SK, BW_WIDE = 1000.0, 3000.0
    def centred(ids, w, h):
        hb = box_around(ids, pad_m=0); cl, ca = (hb[0]+hb[1])/2, (hb[2]+hb[3])/2
        return (cl - w/2/M_LON, cl + w/2/M_LON, ca - h/2/M_LAT, ca + h/2/M_LAT)
    boxes = {}
    for r in order:
        if r in skinny: boxes[r] = centred(RID[r], BW_SK, BH)
    boxes['margin'] = centred(RID['margin'], BW_WIDE, BH)               # old I: a little narrower (3.3 -> 3.0 km), a little taller
    hb = box_around(RID['fan'], pad_m=180)                               # old H: keep its east edge, extend west and up/down to 3.0 x 2.4 km
    fit_w = max((hb[1]-hb[0])*M_LON, (hb[3]-hb[2])*M_LAT*1.45)
    east = (hb[0]+hb[1])/2 + fit_w/2/M_LON; ca = (hb[2]+hb[3])/2
    boxes['fan'] = (east - BW_WIDE/M_LON, east, ca - BH/2/M_LAT, ca + BH/2/M_LAT)
    for r in order:                                                      # every block of the region must sit inside its box
        for i in RID[r]:
            o = next(q for q in sites if q['id'] == i); lo, la = site_ll(o); b = boxes[r]
            assert b[0] <= lo <= b[1] and b[2] <= la <= b[3], (r, i)
    W = 24.0
    left, gap, cslot, right = 0.9, 0.55, 1.65, 0.2
    km = sum(BW_SK if r in skinny else BW_WIDE for r in order)/1000.0
    inch_per_km = (W - left - 2*gap - 4*cslot - right) / km              # gaps D|E and H|I; colour-bar slots after E, F, G and I
    sh = BH/1000.0*inch_per_km
    wC = 5.6
    a, b, hA = ab_layout(W - wC - 0.4, 0, None)
    Htot = 0.45 + hA + 1.25 + sh + 0.75
    rect = lambda x, y, w, h: [x/W, y/Htot, w/W, h/Htot]
    fig = plt.figure(figsize=(W, Htot), facecolor='#fcfcfb')
    y = Htot - 0.45
    a, b, _ = ab_layout(W - wC - 0.4, y, hA)
    axA = fig.add_axes(rect(*a)); axB = fig.add_axes(rect(*b))
    panel_A(axA); nB = panel_B(axB, insets={letters[r]: boxes[r] for r in order})
    colorbar(fig, rect(b[0]+b[2]+0.12, b[1], 0.2, b[3]), nB, 'Elevation [m] (tint)')
    axC = fig.add_axes(rect(W - wC - 0.15, b[1], wC, b[3])); panel_C(axC, fs=7.4*FS, ncol=1, per=len(sites),
        title='C. Blocks of rover stations\nN× Navcam 360s, Z× Mastcam-Z 360s within ±30 m')
    axC.title.set_fontsize(8.6)
    yt = b[1] - 1.25
    T = {r: terrain(boxes[r], 0.00002, [12, 13]) for r in order}
    def norm_of(rs):
        z = np.concatenate([T[r][2][np.isfinite(T[r][2])].ravel() for r in rs]); return plt.Normalize(np.percentile(z, 1), np.percentile(z, 99))
    NORM = {'rimw': norm_of(['rimw', 'whh']), 'whh': norm_of(['rimw', 'whh']), 'margin': norm_of(['margin']), 'fan': norm_of(['fan']),
            'delta': norm_of(['delta', 'land']), 'land': norm_of(['delta', 'land'])}
    CB = {'whh': 'Elevation [m], D and E', 'margin': 'Elevation [m], F', 'fan': 'Elevation [m], G', 'land': 'Elevation [m], H and I'}
    LONG = {'rimw': 'Crater rim west', 'whh': 'Crater rim', 'margin': 'Margin Unit and Neretva Vallis', 'fan': 'Upper Fan and Belva Crater',
            'delta': 'Delta Front', 'land': 'Crater Floor and Séítah'}
    last = int(S.max())
    def sol_span(r):
        bl = [q for q in sites if q['id'] in RID[r]]
        a = min(q['runs'][0][0] for q in bl); bmax = max(q['runs'][-1][1] for q in bl)
        return f"{a}–{bmax}" + ('+' if bmax >= last else '')
    x = left
    for k, r in enumerate(order):
        w = (BW_SK if r in skinny else BW_WIDE)/1000.0*inch_per_km
        ax = fig.add_axes(rect(x, yt-sh, w, sh))
        gx, gy, Z, _ = T[r]; draw_terrain(ax, gx, gy, Z, NORM[r], 10, exag=2)
        overlay(ax, boxes[r], ids=RID[r], eod=True, squares=True, fs=7.5*FS)
        scalebar(ax, boxes[r], 200 if r in skinny else 500, '200 m' if r in skinny else '500 m', where='right' if r == 'margin' else 'left')
        lab = (f"{letters[r]}. {LONG[r]},\nsols {sol_span(r)},\nblocks {rng(r)}" if r in skinny
               else f"{letters[r]}. {LONG[r]},\nsols {sol_span(r)}, blocks {rng(r)}")
        ax.set_title(lab, loc='left', fontsize=9.2*FS, color=INK)
        ax.tick_params(labelsize=6.3*FS); ax.ticklabel_format(useOffset=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2 if r in skinny else 5)); ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        x += w
        if r in CB: colorbar(fig, rect(x+0.1, yt-sh, 0.17, sh), NORM[r], CB[r]); x += cslot
        else: x += gap
    fig.text(left/W, 0.22/Htot, 'D–I share one scale (2.4 km tall). Squares = blocks of rover stations (±30 m); dots = end-of-drive Navcam 360; '
             'small squares = mid-drive partial Navcam; triangles = Mastcam-Z numbered 360s; diamonds = sample cores.', fontsize=8.5*FS, color=INK2)
    fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor()); print('saved', out)

ISIDIS_LATC = 14.0
ISIDIS = (85.5 - 15.0/np.cos(np.deg2rad(ISIDIS_LATC)), 85.5 + 15.0/np.cos(np.deg2rad(ISIDIS_LATC)), ISIDIS_LATC - 15.0, ISIDIS_LATC + 15.0)   # square, 30 deg tall
def box_2to1(b):
    hm = (b[3]-b[2])*M_LAT; wm = 2*hm; c = (b[0]+b[1])/2
    return (c - wm/2/M_LON, c + wm/2/M_LON, b[2], b[3])
BOX_B5 = box_2to1(BOX_B)
BOX_C5 = (BOX_A[0], BOX_A[1], BOX_A[3] - (BOX_A[1]-BOX_A[0])*COS, BOX_A[3])     # crater panel square (1:1): y max 18.8 kept, bottom extended to ~17.85
def fig5(out, with_I=True):
    """A global Mars | B Isidis (true elevation range); C crater (2:1) | D traverse (2:1); E-H tall regional panels;
    I one wide panel (upper fan, Belva, margin unit, Neretva Vallis) at the same scale. with_I=False drops row 4 (figure 6).
    Every row spans the same width: its last colour bar ends where row 2's does."""
    W = 18.0; left, right, gapC = 0.85, 0.2, 0.95
    # ---- row 1 geometry ----
    rI = (ISIDIS[1]-ISIDIS[0])*np.cos(np.deg2rad(ISIDIS_LATC))/(ISIDIS[3]-ISIDIS[2])   # = 1 (square)
    h1 = (W - left - gapC - 1.05 - right)/(2.0 + rI)          # colour bar ends at W - 0.93, like row 2's
    # ---- row 2 geometry (same as the other figures) ----
    a2, b2, h2 = ab_layout(W, 0, None, bbox=BOX_B5, crater=BOX_C5)
    # ---- rows 3-4 geometry: [wide][cb][skinny g skinny][cb] / [skinny g skinny][cb][wide][cb] ----
    g, cslot, cbw = 0.75, 1.65, 0.27                 # gap between skinny panels (room for F's and H's y tick labels), middle colour-bar slot, colour bar + its pad
    R = W - 0.93                                     # right edge of the row-2 colour bar; rows 1, 3 and 4 end there too
    s_in = (R - left - 2*g - cslot - cbw)/4          # E-H: 1.5 km wide panels filling the row
    inch_per_km = s_in/1.5
    sh, sh4 = 4.23, 5.29                             # panel heights (in) kept from the previous layout ...
    BH = sh/inch_per_km*1000; BH4 = sh4/inch_per_km*1000   # ... so the wider panels show less terrain at the top (y max lowered)
    w4_in = R - left - cbw; w4_m = w4_in/inch_per_km*1000
    Htot = 0.4 + h1 + 0.9 + h2 + 0.95 + sh + 0.82 + (sh4 if with_I else sh) + 1.15   # figure 6: E F / G H two over two
    rect = lambda x, y, w, h: [x/W, y/Htot, w/W, h/Htot]
    fig = plt.figure(figsize=(W, Htot), facecolor='#fcfcfb')
    # ===== row 1: global Mars and Isidis =====
    y = Htot - 0.4
    gx, gy, Zg = tile_dem([4], (-180, 180, -90, 90), 0.25)
    gxi, gyi, Zi = tile_dem([5, 6, 7, 8], ISIDIS, 0.03)   # z7/z8 tiles resolve Jezero itself (z5/z6 alone smooth it away)
    nG = plt.Normalize(np.nanpercentile(Zg, 1), np.nanpercentile(Zg, 99))   # clipped at the 1st/99th percentile (~±5.9 km)
    axG = fig.add_axes(rect(left, y-h1, 2*h1, h1))
    draw_terrain(axG, gx, gy, Zg, nG, 2000, exag=0.02)
    axG.set_xlim(-180, 180); axG.set_ylim(-90, 90); axG.set_aspect('equal')
    axG.set_xticks(range(-180, 181, 60)); axG.set_yticks(range(-90, 91, 30))
    axG.add_patch(plt.Rectangle((ISIDIS[0], ISIDIS[2]), ISIDIS[1]-ISIDIS[0], ISIDIS[3]-ISIDIS[2], fill=False, ec=INK, lw=1.3, zorder=5))
    axG.text(ISIDIS[0], ISIDIS[3]+3, 'B', fontsize=11*FS, fontweight='bold', color=INK)
    axG.plot(lon[0], lat[0], marker='*', ms=11, color=C4, mec=INK, mew=0.6, zorder=7)   # above the black box
    axG.set_title('A. Mars (MOLA-based global terrain)', loc='left', fontsize=10*FS, color=INK)
    axG.set_xlabel('Longitude [°E]'); axG.set_ylabel('Latitude [°N]')
    axI = fig.add_axes(rect(left + 2*h1 + gapC, y-h1, rI*h1, h1))
    draw_terrain(axI, gxi, gyi, Zi, nG, 500, exag=0.3)
    axI.set_xlim(ISIDIS[0], ISIDIS[1]); axI.set_ylim(ISIDIS[2], ISIDIS[3]); axI.set_aspect(1/np.cos(np.deg2rad(ISIDIS_LATC)))
    axI.plot(lon[0], lat[0], marker='*', ms=6, color=C4, mec=INK, mew=0.4, zorder=7)      # landing site, above the black box; small so Jezero's rim shows
    axI.add_patch(plt.Rectangle((BOX_C5[0], BOX_C5[2]), BOX_C5[1]-BOX_C5[0], BOX_C5[3]-BOX_C5[2], fill=False, ec=INK, lw=0.9, zorder=5))
    axI.text(BOX_C5[0]-0.2, BOX_C5[3]+0.6, 'C', fontsize=11*FS, fontweight='bold', color=INK)
    axI.text(88.0, 13.0, 'Isidis Planitia', ha='center', fontsize=9*FS, color=INK, style='italic')
    axI.set_title('B. Isidis basin and Jezero crater', loc='left', fontsize=10*FS, color=INK)
    axI.set_xlabel('Longitude [°E]')
    colorbar(fig, rect(left + 2*h1 + gapC + rI*h1 + 0.12, y-h1, 0.2, h1), nG, 'Elevation [m], A and B', extend='both')
    # ===== row 2: crater and traverse =====
    y = y - h1 - 0.9
    a, b, _ = ab_layout(W, y, h2, bbox=BOX_B5, crater=BOX_C5)
    axA = fig.add_axes(rect(*a)); axB = fig.add_axes(rect(*b))
    order3 = ['rimw', 'whh', 'delta', 'land']; order4 = ['mfan'] if with_I else []      # row 3: four tall panels; row 4: one wide panel
    letters = dict(zip(order3 + order4, 'EFGHI'))
    RID['mfan'] = sorted(RID['margin'] + RID['fan'])
    skinny = {'rimw', 'whh', 'delta', 'land'}
    def centred(ids, w, h, h_ref):
        """box w x h metres, x centred on the blocks; bottom edge where an h_ref-tall centred box would put it (only the top moves)"""
        hb = box_around(ids, pad_m=0); cl, ca = (hb[0]+hb[1])/2, (hb[2]+hb[3])/2
        y0 = ca - h_ref/2/M_LAT
        return (cl - w/2/M_LON, cl + w/2/M_LON, y0, y0 + h/M_LAT)
    boxes = {r: centred(RID[r], 1500.0, BH, 2000.0) for r in skinny}
    if with_I: boxes['mfan'] = centred(RID['mfan'], w4_m, BH4, 2500.0)    # I: margin unit + upper fan in one panel; may overlap F and G
    for r in boxes:
        for i in RID[r]:
            o = next(q for q in sites if q['id'] == i); lo, la = site_ll(o); bb = boxes[r]
            assert bb[0] <= lo <= bb[1] and bb[2] <= la <= bb[3], (r, i)
    panel_A(axA, letter='C', box_label='D', bbox=BOX_B5, crater=BOX_C5, scalebar_right=True, names=True)
    nB = panel_B(axB, insets={letters[r]: boxes[r] for r in boxes}, letter='D', bbox=BOX_B5, bg_in_title=False)
    colorbar(fig, rect(b[0]+b[2]+0.12, b[1], 0.2, b[3]), nB, 'Elevation [m] (tint)')
    # brighten A and B (same colours and shading) so their mean brightness matches C and D
    lum = lambda im: np.asarray(im.get_array())[..., :3].mean()
    target = np.mean([lum(axA.images[0]), lum(axB.images[0])])
    for ax_ in (axG, axI):
        im = ax_.images[0]; g_ = target/lum(im); im.set_data(np.clip(np.asarray(im.get_array())*g_, 0, 1))
        print(f'brightness gain {ax_.get_title()[:2]} {g_:.2f}')
    # ===== rows 3-4: regional panels =====
    T = {r: terrain(boxes[r], 0.00002, [12, 13]) for r in boxes}
    def norm_of(rs):
        z = np.concatenate([T[r][2][np.isfinite(T[r][2])].ravel() for r in rs]); return plt.Normalize(np.percentile(z, 1), np.percentile(z, 99))
    NORM = {'delta': norm_of(['delta', 'land']), 'land': norm_of(['delta', 'land']),
            'rimw': norm_of(['rimw', 'whh']), 'whh': norm_of(['rimw', 'whh'])}
    if with_I: NORM['mfan'] = norm_of(['mfan'])
    CB = {'whh': 'Elevation [m], E and F', 'land': 'Elevation [m], G and H', 'mfan': 'Elevation [m], I'}
    LONG = {'rimw': 'Crater rim west', 'whh': 'Crater rim', 'margin': 'Margin Unit and Neretva Vallis', 'fan': 'Upper Fan and Belva Crater',
            'delta': 'Delta Front', 'land': 'Crater Floor and Séítah', 'mfan': 'Upper Fan, Belva Crater, Margin Unit and Neretva Vallis'}
    last = int(S.max())
    def sol_span(r):
        bl = [q for q in sites if q['id'] in RID[r]]
        a0 = min(q['runs'][0][0] for q in bl); b0 = max(q['runs'][-1][1] for q in bl)
        return f"{a0}–{b0}" + ('+' if b0 >= last else '')
    y3 = b[1] - 0.95; y4 = y3 - sh - 0.82
    if with_I: layout = ((order3, y3), (order4, y4))
    else:      layout = ((['rimw', 'whh'], y3), (['delta', 'land'], y4))     # figure 6: two over two, block list on the right
    bottom = layout[-1][0]
    for row, yt in layout:
        x = left
        for r in row:
            w = s_in if r in skinny else w4_in; hh = sh if r in skinny else sh4
            ax = fig.add_axes(rect(x, yt-hh, w, hh))
            gxr, gyr, Zr, _ = T[r]; draw_terrain(ax, gxr, gyr, Zr, NORM[r], 10, exag=2)
            overlay(ax, boxes[r], ids=RID[r], eod=True, squares=True, fs=10)        # block numbers
            scalebar(ax, boxes[r], 500, '500 m', where='right' if r == 'mfan' else 'left')
            room = (w + (g if r in skinny and r not in CB else (cslot if r in CB else 0)) - 0.1)*fig.dpi   # panel width + the gap to its right
            fs_t = 9.5*FS
            while True:                                            # one-line title, shrunk only if it would run into the next panel
                t = ax.set_title(f"{letters[r]}. {LONG[r]}, sols {sol_span(r)}, blocks {RID[r][0]}–{RID[r][-1]}", loc='left', fontsize=fs_t, color=INK)
                if t.get_window_extent(fig.canvas.get_renderer()).width <= room or fs_t < 7: break
                fs_t -= 0.25
            ax.tick_params(labelsize=7.8*FS); ax.ticklabel_format(useOffset=False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3 if r in skinny else 10)); ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            if r == row[0]: ax.set_ylabel('Latitude [°N]')       # axis labels only on the left-most column ...
            if row is bottom: ax.set_xlabel('Longitude [°E]')    # ... and the bottom row
            x += w
            if r in CB: colorbar(fig, rect(x+0.1, yt-hh, cbw-0.1, hh), NORM[r], CB[r]); x += cslot
            else: x += g
    if not with_I:   # ---- figure 6: list of all blocks to the right of E-H ----
        x_l = left + 2*s_in + g + cslot + 0.15; top = y3 + 0.05; bot = y4 - sh; wl = R - x_l
        axL = fig.add_axes(rect(x_l, bot, wl, top - bot)); axL.axis('off')
        NW = max(len(o['name']) for o in sites)
        lines = [f"{o['id']:>2}  {o['name']:<{NW}}  {fmt_counts(o, pad=True):<10} sols {fmt_runs(o['runs'])}" + ('+' if o['runs'][-1][1] >= last else '')
                 for o in sites]
        txt = axL.text(0, 0.965, '\n'.join(lines), va='top', ha='left', family='DejaVu Sans Mono', fontsize=10, color=INK,
                       transform=axL.transAxes, linespacing=1.3)
        rr = fig.canvas.get_renderer(); bb = txt.get_window_extent(rr)
        k = min(wl*fig.dpi/bb.width, 0.95*(top - bot)*fig.dpi/bb.height)   # largest font that fits the box
        txt.set_fontsize(min(10*k, 11.5*FS))
        axL.text(0, 1.0, 'I. Blocks of rover stations: N× = Navcam 360s, Z× = Mastcam-Z 360s within ±30 m; sols ended inside',
                 va='top', ha='left', fontsize=9.5*FS, color=INK, transform=axL.transAxes)
        print('list font', round(min(10*k, 11.5*FS), 2))
    dpi5 = DPI
    scale_note = (f'E–I share one scale (E–H {BH/1000:.1f} km tall, I {BH4/1000:.1f} km tall; box I in D overlaps F and G).' if with_I
                  else f'E–H share one scale ({BH/1000:.1f} km tall, 1.5 km wide).')
    bg_note = ' Background C–I: coarse HiRISE-DTM-based terrain, ~70 m grid (HiRISE DTM not yet connected).' if not used_all else ' Background C–I: HiRISE DTM where available.'
    fig.text(left/W, 0.1/Htot, scale_note + bg_note + '\nSquares = blocks of rover stations (±30 m, numbered by first sol); dots = end-of-drive Navcam 360; '
             'small squares = mid-drive partial Navcam; triangles = Mastcam-Z numbered 360s; diamonds = sample cores.', fontsize=8.5*FS, color=INK2)
    fig.savefig(out, dpi=dpi5, facecolor=fig.get_facecolor()); print('saved', out, dpi5, 'dpi')

def fig6(out):
    """Figure 5 without row 4 (panel I)."""
    fig5(out, with_I=False)

if __name__ == '__main__':
    od = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs'
    which = sys.argv[2:] or ['2', '3', '4', '5', '6']
    names = {'2': (fig2, 'm20_jezero_traverse_clusters_map.png'), '3': (fig3, 'm20_jezero_traverse_sites_regions.png'),
             '4': (fig4, 'm20_jezero_traverse_sites_regions_wide.png'), '5': (fig5, 'm20_mars_to_rover_sites.png'),
             '6': (fig6, 'm20_mars_to_rover_sites_no_I.png')}
    if len(which) > 1:   # one process per figure keeps peak memory down at 300 dpi
        import subprocess
        for w in which: subprocess.run([sys.executable, __file__, od, w], check=True)
    else:
        f, n = names[which[0]]; f(f'{od}/{n}')
