"""mppp.map prototype: one entry point for the M2020 block-of-rover-stations figures and the MPPP sites export.

Run from this folder (the scripts read M20_WAYPOINTS-style inputs from the working directory):

    python mppp_map.py all   [OUTDIR]          # blocks -> zooms -> figures 2-6 -> sites_blocks.json
    python mppp_map.py blocks                  # sitesdef.py      -> sites_zoom.json (block list, centres, visits, counts)
    python mppp_map.py zooms   [OUTDIR]        # zooms2.py        -> m20_cluster_zooms.png
    python mppp_map.py figures [OUTDIR] [2 3 4 5 6] # mapplot3.py -> the five map figures (one process each)
    python mppp_map.py export  [OUTDIR]        # export_sites.py  -> sites_blocks.json (MPPP sites v4)
    python mppp_map.py clusters                # cluster.py + analyze.py (raw 10 m clusters; only needed after new waypoints)

Each step runs in its own Python process, which keeps peak memory low at 300 dpi and matches how the scripts were
written (module-level code). When this moves into the package, the same steps become functions in mppp.map
(see docs/mppp_map.md, "Proposed API").
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))

def _run(*args):
    t = time.time()
    print('>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)
    print(f'   done in {time.time()-t:.0f} s', flush=True)

def clusters(): _run('cluster.py'); _run('analyze.py')
def blocks(): _run('sitesdef.py')
def zooms(outdir): _run('zooms2.py', os.path.join(outdir, 'm20_cluster_zooms.png'))
def figures(outdir, which=('2', '3', '4', '5', '6')): _run('mapplot3.py', outdir, *which)
def export(outdir): _run('export_sites.py', os.path.join(outdir, 'sites_blocks.json'))

def main(argv):
    if not argv or argv[0] in ('-h', '--help'): print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    outdir = os.path.abspath(rest[0]) if rest else HERE
    os.makedirs(outdir, exist_ok=True)
    if cmd == 'all':
        blocks(); zooms(outdir); figures(outdir); export(outdir)
    elif cmd == 'clusters': clusters()
    elif cmd == 'blocks': blocks()
    elif cmd == 'zooms': zooms(outdir)
    elif cmd == 'figures': figures(outdir, tuple(rest[1:]) or ('2', '3', '4', '5', '6'))
    elif cmd == 'export': export(outdir)
    else: sys.exit(f'unknown command {cmd!r}; try --help')

if __name__ == '__main__':
    main(sys.argv[1:])
