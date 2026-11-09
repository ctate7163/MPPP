"""Does COLMAP's default pipeline fail on single Mars stereo pairs, and does the camera model explain it?
For every stereo pair of the e2e project (9 Navcam half-res pairs, 5 Mastcam-Z 34 pairs) run COLMAP's incremental mapper
on the pair alone under several camera assumptions, and record whether a two-image model comes out."""
import sys, json, os, shutil, time, sqlite3
sys.path.insert(0, '/home/claude/MPPP_repo/src')
import numpy as np, pycolmap
from pathlib import Path
from mppp.sfm.project import SfmProject
from mppp.sfm.calibration import label_model_from_meta

S = Path('/tmp/claude-0/-home-claude/5c36b371-e247-5cde-81cf-dc530db4e816/scratchpad')
root = S / 'e2e2/e2etest_colmap_nav_zcam34'
proj = SfmProject.load(root / 'colmap')
manifest = {m['filename']['stem']: m for m in json.load(open(root / 'processed/mppp_manifest_v0p22.json'))['images']}
imgdir = root / 'colmap/images'
fdb = pycolmap.Database.open(str(root / 'colmap/features.db'))
fimgs = {im.name: im for im in fdb.read_all_images()}
kp = {n: fdb.read_keypoints(im.image_id) for n, im in fimgs.items()}
ds = {n: fdb.read_descriptors(im.image_id) for n, im in fimgs.items()}
fdb.close()
by = {}
for r in proj.images:
    by.setdefault((r['sclk_key'], r['instrument'][:1]), {})[r['eye']] = r
pairs = [(d['L'], d['R']) for k, d in sorted(by.items()) if 'L' in d and 'R' in d]
print(len(pairs), 'stereo pairs')

def cam_variant(r, variant):
    """(model, params, refine) for image row r (native size) under a variant."""
    w, h = r['native_size']; s = float(r['downsample_scale'])
    if variant == 'default':                                    # COLMAP: SIMPLE_RADIAL, focal = 1.2 max(w,h), refined
        f = 1.2 * max(w, h)
        return 'SIMPLE_RADIAL', [f, w / 2, h / 2, 0.0], True
    c = proj.cameras[r['instrument']]
    p = np.array(c['params'], float)
    if variant in ('mppp', 'mppp_refine'):                        # MPPP camera (rational Navcam / Z34 label median), scaled
        q = p.copy(); q[:4] *= s
        return c['model'], q.tolist(), variant == 'mppp_refine'
    if variant == 'label_cahvor':                               # the label read as perspective CAHVOR: pinhole + R1,R2 as k1,k2
        m = manifest[r['stem']]
        cm = label_model_from_meta(m)                           # full-frame, camera frame
        _, lin = cm.decompose()
        k1 = float(cm.R[1]) if cm.R is not None else 0.0
        k2 = float(cm.R[2]) if cm.R is not None else 0.0
        fx, fy, cx, cy = lin['hs'] * s, lin['vs'] * s, (lin['hc'] + 0.5) * s, (lin['vc'] + 0.5) * s
        return 'OPENCV', [fx, fy, cx, cy, k1, k2, 0.0, 0.0], False
    raise ValueError(variant)

VARIANTS = {
    'default COLMAP (unknown intrinsics)': ('default', {}, {}),
    'default COLMAP, no planar rejection (max_H_inlier_ratio 1)': ('default', {}, {'max_H_inlier_ratio': 1.0}),
    'default COLMAP, no planar rejection, init 1 deg': ('default', {'init_min_tri_angle': 1.0}, {'max_H_inlier_ratio': 1.0}),
    'label as CAHVOR pinhole+k1k2 fixed, no planar rejection, init 1 deg': ('label_cahvor', {'init_min_tri_angle': 1.0}, {'max_H_inlier_ratio': 1.0}),
    'MPPP camera fixed, default': ('mppp', {}, {}),
    'MPPP camera fixed, no planar rejection': ('mppp', {}, {'max_H_inlier_ratio': 1.0}),
    'MPPP camera fixed, no planar rejection, init 1 deg': ('mppp', {'init_min_tri_angle': 1.0}, {'max_H_inlier_ratio': 1.0}),
}
res = {}
for vname, (variant, mopts, vopts) in VARIANTS.items():
    rows = []
    for L, R in pairs:
        wd = S / '_pair'
        shutil.rmtree(wd, ignore_errors=True); wd.mkdir()
        db = pycolmap.Database.open(str(wd / 'db.db'))
        ids = []
        for r in (L, R):
            model, params, refine = cam_variant(r, variant)
            w, h = r['native_size']
            cam = pycolmap.Camera(model=model, width=w, height=h, params=params)
            cam.has_prior_focal_length = variant != 'default'         # COLMAP without EXIF: focal unknown
            cid = db.write_camera(cam)
            iid = db.write_image(pycolmap.Image(name=r['name'], camera_id=cid))
            db.write_keypoints(iid, kp[r['name']]); db.write_descriptors(iid, ds[r['name']])
            ids.append(iid)
            # one trivial rig and frame per image (pycolmap 4: images without frames cannot be registered)
            sen = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)
            rig = pycolmap.Rig(); rig.add_ref_sensor(sen); rid = db.write_rig(rig)
            fr = pycolmap.Frame(); fr.rig_id = rid; fr.add_data_id(pycolmap.data_t(sensor_id=sen, id=iid)); db.write_frame(fr)
        db.close()
        mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = 2
        vo = pycolmap.TwoViewGeometryOptions()
        for k, v in vopts.items():
            setattr(vo, k, v)
        pycolmap.match_exhaustive(str(wd / 'db.db'), matching_options=mo, verification_options=vo, device=pycolmap.Device.cpu)
        d2 = sqlite3.connect(str(wd / 'db.db'))
        tv = d2.execute('select rows, config from two_view_geometries').fetchall(); d2.close()
        inl = int(tv[0][0]) if tv else 0
        cfg = int(tv[0][1]) if tv else 0
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = refine; opts.ba_refine_extra_params = refine; opts.ba_refine_principal_point = False
        opts.triangulation.ignore_two_view_tracks = False           # a 2-image dataset has only two-view tracks
        for k, v in mopts.items():
            setattr(opts.mapper, k, v)
        (wd / 'sparse').mkdir()
        t = time.time()
        recs = pycolmap.incremental_mapping(str(wd / 'db.db'), str(imgdir), str(wd / 'sparse'), options=opts)
        ok = any(rec.num_reg_images() == 2 for rec in recs.values()) if recs else False
        row = {'pair': L['name'][:8], 'family': L['instrument'][:1], 'inliers': inl, 'config': cfg, 'ok': ok}
        if ok:
            rec = [r_ for r_ in recs.values() if r_.num_reg_images() == 2][0]
            row.update(points=rec.num_points3D(), mean_reproj_px=float(rec.compute_mean_reprojection_error()),
                       mean_track=float(rec.compute_mean_track_length()))
        rows.append(row)
    n = len(rows); nok = sum(r['ok'] for r in rows)
    res[vname] = {'pairs': n, 'ok': nok, 'ok_N': sum(r['ok'] for r in rows if r['family'] == 'N'),
                  'N_pairs': sum(1 for r in rows if r['family'] == 'N'), 'ok_Z': sum(r['ok'] for r in rows if r['family'] == 'Z'),
                  'median_reproj_px': float(np.median([r['mean_reproj_px'] for r in rows if r['ok']])) if nok else None,
                  'median_points': float(np.median([r['points'] for r in rows if r['ok']])) if nok else None,
                  'median_inliers': float(np.median([r['inliers'] for r in rows])),
                  'planar_classified': sum(1 for r in rows if r['config'] in (4, 5, 6)), 'rows': rows}
    print(vname, {k: v for k, v in res[vname].items() if k != 'rows'}, flush=True)
json.dump(res, open(S / 'pair_exp.json', 'w'), indent=1)
