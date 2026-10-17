"""DEM access for figures. Looks for the USGS HiRISE DTM mosaics (1 m posts, equirectangular, R = 3,396,190 m).
Order of preference: MSR TRN HiRISE DTM (Bland et al. 2024/25, 77.057-77.380E) then Mars 2020 TRN HiRISE DTM (Fergason et al. 2020).
Set M20_DEMS to a os.pathsep-separated list of GeoTIFF paths, or drop the files into ./dem/.
Waypoint easting/northing (lat_ts ~18.47 deg) are converted through lon/lat, never used directly against the DEM grid."""
import os, glob, json, numpy as np
from scipy.ndimage import map_coordinates
try:
    import rasterio
    from pyproj import Transformer
except Exception:
    rasterio = None

def dem_paths():
    env = os.environ.get('M20_DEMS', '')
    found = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dem', '**', '*.tif'), recursive=True))
    # the USGS zips also hold orthoimages and confidence maps: keep elevation rasters only
    found = [p for p in found if any(t in os.path.basename(p).upper() for t in ('DTM', 'DEM')) and not any(t in os.path.basename(p).upper() for t in ('ORTHO', 'CONF', 'FOM'))]
    paths = [p for p in env.split(os.pathsep) if p] + found
    # MSR first (name contains MSR), then the rest
    return sorted(set(paths), key=lambda p: (0 if 'MSR' in os.path.basename(p).upper() else 1, p))

class LocalFrame:
    """Linear map between waypoint easting/northing and lon/lat, fitted on the waypoint file."""
    def __init__(self, P):
        lon = np.array([p['lon'] for p in P]); lat = np.array([p['lat'] for p in P])
        self.ce = np.polyfit(lon, [p['easting'] for p in P], 1); self.cn = np.polyfit(lat, [p['northing'] for p in P], 1)
    def to_lonlat(self, e, n):
        return (np.asarray(e) - self.ce[1]) / self.ce[0], (np.asarray(n) - self.cn[1]) / self.cn[0]

def sample_dem(lon, lat):
    """Bilinear DEM heights at lon/lat arrays. Returns (z, source_name) with NaN where no DEM covers the point."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    z = np.full(lon.shape, np.nan); used = []
    if rasterio is None: return z, used
    for path in dem_paths():
        todo = np.isnan(z)
        if not todo.any(): break
        with rasterio.open(path) as src:
            tr = Transformer.from_crs("+proj=longlat +R=3396190 +no_defs", src.crs, always_xy=True)
            x, y = tr.transform(lon[todo], lat[todo])
            r, c = rasterio.transform.rowcol(src.transform, x, y, op=lambda v: v)  # fractional
            r = np.asarray(r, float) - 0.5; c = np.asarray(c, float) - 0.5
            ok = (r >= 0) & (c >= 0) & (r <= src.height - 1) & (c <= src.width - 1)
            if not ok.any(): continue
            r0, r1 = int(np.floor(r[ok].min())), int(np.ceil(r[ok].max())) + 1
            c0, c1 = int(np.floor(c[ok].min())), int(np.ceil(c[ok].max())) + 1
            win = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
            # decimate on read for large footprints (overview maps)
            step = max(1, int(max(win.height, win.width) / 3000))
            arr = src.read(1, window=win, out_shape=(int(win.height // step) or 1, int(win.width // step) or 1), masked=True).astype(float)
            arr = arr.filled(np.nan)
            vals = map_coordinates(arr, [(r[ok] - r0) / step, (c[ok] - c0) / step], order=1, mode='nearest', cval=np.nan)
            sub = z[todo]; sub[ok] = vals; z[todo] = sub
            used.append(os.path.basename(path))
    return z, used

def hillshade(z, dx, dy, az=315, alt=40, exag=1.0):
    zf = np.where(np.isnan(z), np.nanmean(z) if np.isfinite(z).any() else 0, z)
    gy, gx = np.gradient(zf * exag, dy, dx)
    slope = np.pi/2 - np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
    a, b = np.deg2rad(az), np.deg2rad(alt)
    hs = np.sin(b)*np.sin(slope) + np.cos(b)*np.cos(slope)*np.cos(a - aspect)
    return np.clip(hs, 0, 1)
