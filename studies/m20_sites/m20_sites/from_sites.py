"""Blocks for the figures from MPPP's sites.json (version 5, "official_blocks") instead of re-deriving them.

    python from_sites.py [SITES_JSON]        -> sites_zoom.json (the block list every figure script reads)

SITES_JSON defaults to $M20_SITES_FILE, then to the package copy (src/mppp/data/sites.json), which is
where the latest sites information always lives.  Each official block keeps its number, name, region, centre
(waypoint easting/northing), visits, Navcam 360 count, partial-Navcam count and numbered Mastcam-Z 360s, so the
figures show exactly what sites.json says.  sitesdef.py (re-deriving blocks from waypoints) is still there for when
the block definitions themselves change."""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'src', 'mppp', 'data', 'sites.json'))


def sites_file(arg=None):
    for p in (arg, os.environ.get('M20_SITES_FILE'), DEFAULT):
        if p and os.path.exists(p):
            return p
    sys.exit(f'sites.json not found (tried the argument, $M20_SITES_FILE and {DEFAULT})')


# visits left out of the figures' sol ranges only (titles, tables; sites.json keeps them): Three Forks North's
# single sol 413 so its range fits on one line (Christian, 20 Oct)
MAP_OMIT_VISITS = {'threeforks_north': [[413, 413]]}


NEARBY_RADIUS_M = 10.0      # the block-building rule (mppp.waypoints.blocks.NEARBY_RADIUS_M)


def _waypoints():
    """(sol, easting, northing, final) arrays of the study's waypoint file (the one the figures draw)."""
    P = [f['properties'] for f in json.load(open(os.path.join(HERE, 'M20_waypoints.json')))['features']]
    return (np.array([p['sol'] for p in P]), np.array([p['easting'] for p in P], float),
            np.array([p['northing'] for p in P], float), np.array([p['final'] for p in P]))


def block_members(runs, site, W, radius_m=NEARBY_RADIUS_M):
    """Indices of the waypoints of a block by the block-building rule (Christian, 20 Oct): the waypoints on the sols
    of ``runs`` (the primary sol ranges, after MAP_OMIT_VISITS) plus every waypoint within ``radius_m`` of one of
    them - none before the first sol when the site has NEARBY_FROM_FIRST_SOL (its exclusion)."""
    S, X, Y, F = W
    m = np.zeros(len(S), bool)
    for a, b in runs:
        m |= (S >= a) & (S <= b)
    if m.any():
        d = np.sqrt((X[:, None] - X[None, m]) ** 2 + (Y[:, None] - Y[None, m]) ** 2).min(axis=1)
        near = (d <= radius_m) & ~m
        if ((site or {}).get('settings') or {}).get('NEARBY_FROM_FIRST_SOL'):
            near &= S >= min(a for a, _ in runs)
        m |= near
    return np.flatnonzero(m)


def block_counts(members, W, panos):
    """N360 (end-of-drive Navcam 360s), partial Navcam and the numbered Mastcam-Z 360s of the block's waypoints."""
    S, X, Y, F = W
    mem = set(int(i) for i in members)
    pos = lambda s: (np.where(S <= s)[0][-1] if (S <= s).any() else 0)          # noqa: E731
    z = [int(num) for num, ss, _, _ in panos if int(pos(ss[0])) in mem]
    return {'n360': int(sum(F[i] == 'y' for i in mem)), 'npart': int(sum(F[i] == 'm' for i in mem)), 'zpanos': z, 'nz': len(z)}


def blocks_from_sites(path):
    doc = json.load(open(path, encoding='utf-8'))
    if 'official_blocks' not in doc:
        sys.exit(f'{path}: no "official_blocks" table (sites.json version {doc.get("version")}; version 5 is needed)')
    out = []
    try:
        W = _waypoints()
        sys.path.insert(0, HERE)
        from refdata import PANOS
    except (OSError, ImportError, KeyError, ValueError):
        W, PANOS = None, []
    for num, b in sorted(doc['official_blocks'].items(), key=lambda kv: int(kv[0])):
        runs = sorted([int(a), int(z)] for a, z in b['visits'])          # chronological for titles and tables
        omit = MAP_OMIT_VISITS.get(b.get('site'), [])
        shown = [r for r in runs if r not in omit] or runs
        c = b['centre']
        out.append({
            'id': int(num), 'key': b.get('source_key') or b.get('site'), 'site': b.get('site'), 'name': b['name'],
            'region': b.get('region', ''), 'kind': b.get('kind', ''),
            'cx': float(c['easting']), 'cy': float(c['northing']), 'lon': c.get('lon'), 'lat': c.get('lat'),
            'half_size_m': float(b.get('half_size_m', 30.0)),
            'runs': shown, 'first': runs[0][0], 'first_all': min([runs[0][0]] + list(b.get('end_of_drive_sols', []))),
            'n360': int(b.get('n_navcam_360', 0)), 'npart': int(b.get('n_partial_navcam', 0)), 'ninterp': 0,
            'zpanos': list(b.get('mastcam_z_360', [])), 'nz': len(b.get('mastcam_z_360', [])),
        })
        if W is not None:      # 20 Oct: the block's own waypoints (primary sols + 10 m), and the counts of them
            mem = block_members(shown, doc.get('sites', {}).get(b.get('site')), W)
            out[-1]['members'] = [int(i) for i in mem]
            out[-1].update(block_counts(mem, W, PANOS))
            out[-1]['n360_sites_json'], out[-1]['nz_sites_json'] = int(b.get('n_navcam_360', 0)), len(b.get('mastcam_z_360', []))
    return doc, out


if __name__ == '__main__':
    path = sites_file(sys.argv[1] if len(sys.argv) > 1 else None)
    doc, blocks = blocks_from_sites(path)
    json.dump(blocks, open(os.path.join(HERE, 'sites_zoom.json'), 'w'), indent=1)
    print(f'{len(blocks)} official blocks from {path} (version {doc.get("version")}) -> sites_zoom.json')
    for o in blocks:
        print(f"{o['id']:>3} {o['name']:<24} {o['region']:<28} N×{o['n360']:<3} Z×{o['nz']}  visits {o['runs']}")
