"""Blocks for the figures from MPPP's sites.json (version 5, "official_blocks") instead of re-deriving them.

    python from_sites.py [SITES_JSON]        -> sites_zoom.json (the block list every figure script reads)

SITES_JSON defaults to $M20_SITES_FILE, then to the package copy (src/mppp/data/sites.json), which is
where the latest sites information always lives.  Each official block keeps its number, name, region, centre
(waypoint easting/northing), visits, Navcam 360 count, partial-Navcam count and numbered Mastcam-Z 360s, so the
figures show exactly what sites.json says.  sitesdef.py (re-deriving blocks from waypoints) is still there for when
the block definitions themselves change."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'src', 'mppp', 'data', 'sites.json'))


def sites_file(arg=None):
    for p in (arg, os.environ.get('M20_SITES_FILE'), DEFAULT):
        if p and os.path.exists(p):
            return p
    sys.exit(f'sites.json not found (tried the argument, $M20_SITES_FILE and {DEFAULT})')


def blocks_from_sites(path):
    doc = json.load(open(path, encoding='utf-8'))
    if 'official_blocks' not in doc:
        sys.exit(f'{path}: no "official_blocks" table (sites.json version {doc.get("version")}; version 5 is needed)')
    out = []
    for num, b in sorted(doc['official_blocks'].items(), key=lambda kv: int(kv[0])):
        runs = sorted([int(a), int(z)] for a, z in b['visits'])          # chronological for titles and tables
        c = b['centre']
        out.append({
            'id': int(num), 'key': b.get('source_key') or b.get('site'), 'site': b.get('site'), 'name': b['name'],
            'region': b.get('region', ''), 'kind': b.get('kind', ''),
            'cx': float(c['easting']), 'cy': float(c['northing']), 'lon': c.get('lon'), 'lat': c.get('lat'),
            'half_size_m': float(b.get('half_size_m', 30.0)),
            'runs': runs, 'first': runs[0][0], 'first_all': min([runs[0][0]] + list(b.get('end_of_drive_sols', []))),
            'n360': int(b.get('n_navcam_360', 0)), 'npart': int(b.get('n_partial_navcam', 0)), 'ninterp': 0,
            'zpanos': list(b.get('mastcam_z_360', [])), 'nz': len(b.get('mastcam_z_360', [])),
        })
    return doc, out


if __name__ == '__main__':
    path = sites_file(sys.argv[1] if len(sys.argv) > 1 else None)
    doc, blocks = blocks_from_sites(path)
    json.dump(blocks, open(os.path.join(HERE, 'sites_zoom.json'), 'w'), indent=1)
    print(f'{len(blocks)} official blocks from {path} (version {doc.get("version")}) -> sites_zoom.json')
    for o in blocks:
        print(f"{o['id']:>3} {o['name']:<24} {o['region']:<28} N×{o['n360']:<3} Z×{o['nz']}  visits {o['runs']}")
