"""mppp.map prototype: one entry point for the M2020 block-of-rover-stations figures and the MPPP sites export.

Run from this folder:

    python mppp_map.py all     [OUTDIR]          # sites -> zooms -> figures 2-6, from the official blocks in sites.json
    python mppp_map.py sites                     # from_sites.py    -> sites_zoom.json from sites.json version 5 (default)
    python mppp_map.py zooms   [OUTDIR]          # zooms2.py        -> figure 1, the +-30 m panels
    python mppp_map.py figures [OUTDIR] [2 3 4 5 6 7] # mapplot3.py -> the map figures (one process each; 7 = the GitHub banner)
    python mppp_map.py derive  [OUTDIR]          # blocks -> zooms -> figures -> export: re-derive the blocks from waypoints
    python mppp_map.py blocks                    # sitesdef.py      -> sites_zoom.json re-derived (clusters + windows)
    python mppp_map.py export  [OUTDIR]          # export_sites.py  -> sites_blocks.json (sites v4) from re-derived blocks
    python mppp_map.py clusters                  # cluster.py + analyze.py (raw 10 m clusters; after new waypoints)

Environment:
    M20_SITES_FILE   sites.json to draw (default: src/mppp/data/sites.json of this checkout)
    M20_DEMS         1 m HiRISE DTM GeoTIFFs (os.pathsep-separated); without them the ~70 m terrain tiles are used
    M20_ORTHOS       HiRISE orthomosaic GeoTIFFs (same form)
    M20_BACKGROUND   dtm (default) | ortho | white (the cluster zooms on white)
    M20_DPI          output resolution (default 300)
    M20_TAG          appended to every figure name (m20_..._<TAG>.png), so runs do not overwrite each other

Each step runs in its own Python process, which keeps peak memory low and matches how the scripts were written
(module-level code). See docs/mppp_map.md.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))

def _run(*args):
    t = time.time()
    print('>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)
    print(f'   done in {time.time()-t:.0f} s', flush=True)

def clusters(): _run('cluster.py'); _run('analyze.py')
def tagged(name):
    t = os.environ.get('M20_TAG', '').strip()
    return name if not t else name[:-4] + '_' + t + '.png'

def sites(path=None): _run('from_sites.py', *([path] if path else []))
def blocks(): _run('sitesdef.py')
def zooms(outdir): _run('zooms2.py', os.path.join(outdir, tagged('m20_cluster_zooms.png')))
def figures(outdir, which=('2', '3', '4', '5', '6', '7')): _run('mapplot3.py', outdir, *which)
def export(outdir): _run('export_sites.py', os.path.join(outdir, 'sites_blocks.json'))

def main(argv):
    if not argv or argv[0] in ('-h', '--help'): print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    outdir = os.path.abspath(rest[0]) if rest and cmd != 'sites' else HERE
    os.makedirs(outdir, exist_ok=True)
    if cmd == 'all':
        sites(); zooms(outdir); figures(outdir)
    elif cmd == 'derive':
        blocks(); zooms(outdir); figures(outdir); export(outdir)
    elif cmd == 'sites': sites(rest[0] if rest else None)
    elif cmd == 'clusters': clusters()
    elif cmd == 'blocks': blocks()
    elif cmd == 'zooms': zooms(outdir)
    elif cmd == 'figures': figures(outdir, tuple(rest[1:]) or ('2', '3', '4', '5', '6', '7'))
    elif cmd == 'export': export(outdir)
    else: sys.exit(f'unknown command {cmd!r}; try --help')

if __name__ == '__main__':
    main(sys.argv[1:])
