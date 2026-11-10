"""
The Mastcam-Z and Navcam cameras on the mast head (0.95.5, 8 Nov 2026; Christian: "I love zcam rig figure. Add that to
the list of standard figure. Please also do this for the other zooms and look for trends of movement or center pixel
pointing with zoom in the mast frame.").

Every Mastcam-Z block's frames go through the deck_joint mast chain (``deck_joint_mast_model.json``, resolver angles):
  1. per station, the block <- RNAV similarity from the station's Navcam frames (their chain poses against their BA
     poses; badly held frames - horizon views - dropped by a 3 x median cut);
  2. every Z frame's BA pose into RNAV and back through the chain into the head frame: its centre c (from the
     elevation axis) and rotation Q, as the chain's own Navcam c_L / Q_L;
  3. the level mast frame: origin the elevation axis on the azimuth axis, x the NL boresight with the head level,
     y right, z down.  The image-centre pixel's ray and the optical axis as yaw (+ right) / pitch (+ up) there.

Chain (the deck_joint symbols; symbols and definitions in the chain file):
  R_cam<-RNAV = Rot(Q) . Ry(el)^T . Rz(az)^T . Rot(W);   C = [P_x, P_y, 0] + Rot(G) . Rz(az) . (h + Ry(-el) . c)

Writes into --out (with --tag in the names):
  zcam<zoom>_navcam_on_mast_head_<tag>.png   2 x 2 per zoom: top view, side view, pair centre / height against sol
                                             and temperature, centre along the boresight against focus
  zcam_mast_head_vs_zoom_<tag>.png           change from 34 mm of the centre (y, z) and the image-centre pointing
                                             (yaw, pitch); a station with every zoom joined
  zcam_mast_head_<tag>.json                  station table and the mount per zoom (Q_Z per zoom and eye; c_Z per eye
                                             from the 34 mm across the boresight, along the boresight held at its
                                             near-focus median - the along-axis centre trades with the focal law)

Frames need RSM resolver angles in the rover-frame label table (``rover_frame_labels.json`` of the rover_frame stage).

    python -m mppp_studies.calibration.zcam_mast_head --root D:\\rover_blocks --chain <deck_joint_mast_model.json>
           [--labels D:\\rover_blocks\\camera_analysis\\rover_frame_labels.json] [--blocks folder ...] [--out ...] [--tag ...]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ chain
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def RV(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


def chain_pose(m, eye, az, el, T=None):
    """R (cam <- RNAV) and C (RNAV, m) of Navcam eye 'L' or 'R' (deck_joint_model.pose)."""
    p = m["params"]
    ka, ke = p.get("k_az", 0.0), p.get("k_el", 0.0)
    dT = 0.0 if T is None else (T - p.get("T0_C", -20.0))
    R = (RV(np.asarray(p[f"qT_{eye}_rad_per_C"]) * dT) @ RV(p[f"Q_{eye}_rotvec_rad"]) @ Ry(el * (1 + ke)).T
         @ Rz(az * (1 + ka)).T @ RV(p["W_rotvec_rad"]))
    C = (np.r_[p["P_xy_m"], 0.0] + RV(p["G_rotvec_rad"]) @ Rz(az * (1 + ka))
         @ (np.asarray(p["h_m"]) + Ry(-el * (1 + ke)) @ np.asarray(p[f"c_{eye}_m"])))
    return R, C


class Mast:
    """The level mast frame of a chain."""

    def __init__(self, m):
        from scipy.optimize import brentq
        self.m = m
        p = m["params"]
        self.G, self.W, self.h = RV(p["G_rotvec_rad"]), RV(p["W_rotvec_rad"]), np.asarray(p["h_m"])
        self.P0 = np.r_[p["P_xy_m"], 0.0]
        self.ka, self.ke = p.get("k_az", 0.0), p.get("k_el", 0.0)
        self.el_level = brentq(lambda e: chain_pose(m, "L", 0.0, e)[0][2][2], np.radians(60), np.radians(120))
        RL, _ = chain_pose(m, "L", 0.0, self.el_level)
        x = RL[2].copy(); x[2] = 0; x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])
        self.F = np.array([x, np.cross(z, x), z])                    # mast <- RNAV
        self.O = self.P0 + self.G @ self.h
        self.L = self.F @ self.G @ Ry(-self.el_level)                # mast <- head (position frame)

    def head(self, C, R, az, el):
        """c (head frame, m) and Q (cam <- head rotation frame) of a camera pose in RNAV."""
        a, e = az * (1 + self.ka), el * (1 + self.ke)
        c = Ry(-e).T @ (Rz(a).T @ self.G.T @ (C - self.P0) - self.h)
        return c, R @ self.W.T @ Rz(a) @ Ry(e)

    def mast_point(self, c):
        return 1e3 * self.F @ (self.P0 + self.G @ (self.h + Ry(-self.el_level) @ np.asarray(c)) - self.O)

    def mast_dir(self, v, az, el):
        return self.L @ (Ry(-el).T @ Rz(az).T @ self.G.T @ np.asarray(v))


def yaw_pitch(b):
    return 1e3 * np.arctan2(b[1], b[0]), 1e3 * np.arctan2(-b[2], b[0])


def rsd(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


# ------------------------------------------------------------------ blocks
def read_block(folder: Path):
    """pycolmap reconstruction of colmap/sparse/cahv_ba (an empty points file when the folder has none)."""
    import pycolmap
    src = folder / "colmap" / "sparse" / "cahv_ba"
    if not (src / "points3D.bin").is_file():
        tmp = Path(tempfile.mkdtemp())
        for fn in ("cameras.bin", "frames.bin", "images.bin", "rigs.bin"):
            if (src / fn).is_file():
                os.symlink(os.path.abspath(src / fn), tmp / fn)
        (tmp / "points3D.bin").write_bytes(struct.pack("<Q", 0))
        src = tmp
    return pycolmap.Reconstruction(str(src))


def block_rows(folder: Path, labels, mast: Mast, min_nav=6):
    """Per frame (Z and Navcam) at every station with >= min_nav Navcam frames and a Z frame: head-frame c, Q, the
    mast-frame centre, optical axis and image-centre ray; and the per-station fit."""
    from scipy.spatial.transform import Rotation as Rot
    proj = json.loads((folder / "colmap" / "project.json").read_text(encoding="utf-8"))
    info = {i["stem"]: i for i in proj["images"]}
    rec = read_block(folder)
    frames = []
    for im in rec.images.values():
        st = im.name[:-4] if im.name.endswith(".png") else im.name
        e, i = labels.get(st), info.get(st)
        if e is None or i is None or not e.get("rsm") or not im.has_pose:
            continue
        cw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.asarray(cw.rotation.matrix()); C = -R.T @ np.asarray(cw.translation)
        cam = rec.cameras[im.camera_id]
        ctr = np.asarray(cam.cam_from_img(np.array([cam.width / 2.0, cam.height / 2.0])), float).ravel()[:2]
        dc = np.r_[ctr, 1.0]; dc /= np.linalg.norm(dc)
        frames.append(dict(stem=st, cam=st[:2], st=i["station"], sol=i["sol"], seq=i.get("sequence"), R=R, C=C, dc=dc,
                           az=e["rsm"]["az_resolver"], el=e["rsm"]["el_resolver"], focus=i.get("focus_count"),
                           T=i.get("zcam_optics_temperature_degC") if st[0] == "Z" else i.get("camera_temperature_degC"),
                           zoom=int(st.split("_")[-1][:3]) if st[0] == "Z" else None,
                           full=(int(cam.width), int(cam.height)) == (1648, 1200)))
    by = collections.defaultdict(list)
    for f in frames:
        by[f["st"]].append(f)
    rows, fits = [], {}
    for st, F in sorted(by.items()):
        nav = [f for f in F if f["cam"] in ("NL", "NR")]
        if len(nav) < min_nav or not any(f["cam"] in ("ZL", "ZR") for f in F):
            continue
        Ms, Cc, Cb = [], [], []
        for f in nav:
            Rc, Cch = chain_pose(mast.m, f["cam"][1], f["az"], f["el"], f["T"])
            Ms.append(f["R"].T @ Rc); Cc.append(Cch); Cb.append(f["C"])
        Ms, Cc, Cb = np.array(Ms), np.array(Cc), np.array(Cb)
        keep = np.ones(len(nav), bool)
        for _ in range(4):
            Mr = Rot.from_matrix(Ms[keep]).mean().as_matrix()
            X = (Mr @ Cc.T).T
            A = np.c_[X[keep].reshape(-1, 1), np.tile(np.eye(3), (keep.sum(), 1))]
            sol, *_ = np.linalg.lstsq(A, Cb[keep].reshape(-1), rcond=None)
            s, t = sol[0], sol[1:]
            d = np.linalg.norm(Cb - (s * X + t), axis=1)
            new = d < max(0.003, 3 * np.median(d[keep]))
            if (new == keep).all():
                break
            keep = new
        spread = np.degrees(np.linalg.norm((Rot.from_matrix(Ms[keep]) * Rot.from_matrix(Mr).inv()).as_rotvec(), axis=1)) * 1e3
        fits[st] = dict(scale=float(s), nav=int(keep.sum()), dropped=int((~keep).sum()),
                        pos_rms_mm=float(1e3 * np.sqrt(np.mean(d[keep] ** 2))), rot_rms_mdeg=float(np.sqrt(np.mean(spread ** 2))))
        for f in F:
            if f["cam"] not in ("ZL", "ZR"):
                continue
            Cr = Mr.T @ (f["C"] - t) / s; Rr = f["R"] @ Mr
            c, Q = mast.head(Cr, Rr, f["az"], f["el"])
            rows.append(dict(blk=folder.name, st=st, cam=f["cam"], zoom=f["zoom"], sol=f["sol"], seq=f["seq"],
                             focus=f["focus"], T=f["T"], el_deg=float(np.degrees(f["el"])), full=f["full"],
                             c=c, Q=Rot.from_matrix(Q).as_rotvec(), m=mast.mast_point(c),
                             bm=mast.mast_dir(Rr[2], f["az"], f["el"]), cm=mast.mast_dir(Rr.T @ f["dc"], f["az"], f["el"])))
    return rows, fits


def station_table(rows):
    from scipy.spatial.transform import Rotation as Rot
    st = collections.defaultdict(list)
    for r in rows:
        st[(r["zoom"], r["cam"], r["blk"], r["st"])].append(r)
    S = []
    for (z, cam, blk, stn), rr in sorted(st.items()):
        if len(rr) < 4:
            continue
        m = np.array([r["m"] for r in rr]); B = np.array([r["bm"] for r in rr])
        bz = np.median(B, 0); bz /= np.linalg.norm(bz)
        along = m @ bz; perp = m - np.outer(along, bz)
        fo = [r["focus"] for r in rr if r["focus"] is not None]
        near = np.array([r["focus"] is not None and r["focus"] <= np.median(fo) for r in rr]) if fo else np.ones(len(rr), bool)
        A = np.array([yaw_pitch(b) for b in B]); Cn = np.array([yaw_pitch(r["cm"]) for r in rr if r["full"]])
        S.append(dict(zoom=z, cam=cam, blk=blk, st=stn, n=len(rr), sol=int(np.median([r["sol"] for r in rr])),
                      T=float(np.median([r["T"] for r in rr if r["T"] is not None] or [np.nan])),
                      xyz=np.median(m, 0).tolist(), perp=np.median(perp, 0).tolist(), along_near=float(np.median(along[near])),
                      axis_mrad=np.median(A, 0).tolist(), centre_mrad=np.median(Cn, 0).tolist() if len(Cn) else None,
                      Q_rotvec=Rot.from_rotvec(np.array([r["Q"] for r in rr])).mean().as_rotvec().tolist(), boresight=bz.tolist()))
    return S


def mount_model(S, rows, mast: Mast):
    """Q_Z per zoom and eye; c_Z per eye from 34 mm (across the boresight: station medians; along: near-focus median)."""
    from scipy.spatial.transform import Rotation as Rot
    out = {}
    for cam in ("ZL", "ZR"):
        s34 = [s for s in S if s["zoom"] == 34 and s["cam"] == cam]
        e = {"zooms": {}}
        if s34:
            perp = np.median([s["perp"] for s in s34], 0); bz = np.median([s["boresight"] for s in s34], 0); bz /= np.linalg.norm(bz)
            along = float(np.median([s["along_near"] for s in s34]))
            mpt = perp - np.dot(perp, bz) * bz + along * bz
            C = mast.F.T @ (mpt * 1e-3) + mast.O
            c = Ry(-mast.el_level).T @ (mast.G.T @ (C - mast.P0) - mast.h)
            e.update(c_Z_m=c.tolist(), centre_mast_mm=mpt.tolist(), along_boresight_mm_held=along, stations_34=len(s34))
        for z in sorted({s["zoom"] for s in S}):
            ss = [s for s in S if s["zoom"] == z and s["cam"] == cam]
            if not ss:
                continue
            Qs = Rot.from_rotvec(np.array([s["Q_rotvec"] for s in ss])); Qm = Qs.mean()
            fr = [r for r in rows if r["zoom"] == z and r["cam"] == cam]
            res = np.degrees(np.linalg.norm((Rot.from_rotvec(np.array([r["Q"] for r in fr])) * Qm.inv()).as_rotvec(), axis=1)) * 1e3
            cp = [s["centre_mrad"] for s in ss if s["centre_mrad"]]
            e["zooms"][str(z)] = dict(Q_Z_rotvec_rad=Qm.as_rotvec().tolist(), stations=len(ss), frames=len(fr),
                                      frame_rms_mdeg=float(np.sqrt(np.mean(res ** 2))),
                                      centre_pixel_yaw_pitch_mrad=np.median(cp, 0).tolist() if cp else None,
                                      optical_axis_yaw_pitch_mrad=np.median([s["axis_mrad"] for s in ss], 0).tolist())
        out[cam] = e
    return out


# ------------------------------------------------------------------ figures
COL = {"ZL": "#1f5fbf", "ZR": "#c8502a", "NL": "#1f5fbf", "NR": "#c8502a"}


def fig_zoom_head(zoom, S, rows, nav, mast, out, tag, dpi=200, label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mppp.calib.constrain import panel_letters
    SZ = [s for s in S if s["zoom"] == zoom]; rz = [r for r in rows if r["zoom"] == zoom]
    if not SZ:
        return None
    cm = plt.get_cmap("viridis"); solc = lambda s: cm(min(s, 2000) / 2000)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for j, ax in enumerate(axs[0]):
        X = (lambda v: v[1]) if j == 0 else (lambda v: v[0]); Y = (lambda v: v[0]) if j == 0 else (lambda v: v[2])
        for c in ("NL", "NR"):
            v = nav[c]
            ax.scatter(X(v), Y(v), s=120, marker="s", color=COL[c], edgecolors="k", zorder=5,
                       label=f"{c} (deck_joint chain): ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")
            if j == 0:
                ax.annotate("", (v[1], v[0] + 60), (v[1], v[0]), arrowprops=dict(arrowstyle="->", color=COL[c]))
        for s in SZ:
            ax.scatter(X(s["xyz"]), Y(s["xyz"]), s=40 + 0.4 * s["n"], color=solc(s["sol"]), edgecolors=COL[s["cam"]], lw=1.5, zorder=4)
        for c in ("ZL", "ZR"):
            ss = [s for s in SZ if s["cam"] == c]
            if not ss:
                continue
            m = np.median([s["xyz"] for s in ss], 0); b = np.median([s["boresight"] for s in ss], 0)
            ax.scatter(X(m), Y(m), s=150, marker="*", color=COL[c], edgecolors="k", zorder=6,
                       label=f"{c} {zoom} mm, median of {len(ss)} station{'s' if len(ss) > 1 else ''}: ({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})")
            if j == 0:
                ax.annotate("", (m[1] + 60 * b[1], m[0] + 60 * b[0]), (m[1], m[0]), arrowprops=dict(arrowstyle="->", color=COL[c]))
        ax.scatter([0], [0], marker="+", s=200, color="k", label="origin: elevation axis on the azimuth axis")
        ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")
        if j == 0:
            ax.set_xlabel("y right (mm)"); ax.set_ylabel("x forward, level boresight (mm)")
            ax.set_title(f"Top view of the head, level (resolver el {np.degrees(mast.el_level):.1f}°); arrows: optical axes", fontsize=10)
            ax.legend(fontsize=6.3, loc="lower center")
        else:
            ax.set_xlabel("x forward (mm)"); ax.set_ylabel("z down (mm)"); ax.invert_yaxis()
            ax.set_title("Side view: Z centres along their boresight scatter with the focal law", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 2000)); cb = fig.colorbar(sm, ax=axs[0][1], fraction=0.04); cb.set_label("sol (station points)")
    ax = axs[1][0]
    pairs = sorted({(s["blk"], s["st"]) for s in SZ if sum(q["st"] == s["st"] and q["blk"] == s["blk"] for q in SZ) == 2},
                   key=lambda k: [s["sol"] for s in SZ if (s["blk"], s["st"]) == k][0])
    for k_, key in enumerate(pairs):
        zl = np.array([s["xyz"] for s in SZ if (s["blk"], s["st"]) == key and s["cam"] == "ZL"][0])
        zr = np.array([s["xyz"] for s in SZ if (s["blk"], s["st"]) == key and s["cam"] == "ZR"][0])
        s0 = [s for s in SZ if (s["blk"], s["st"]) == key][0]; T = np.nanmean([s["T"] for s in SZ if (s["blk"], s["st"]) == key])
        ax.scatter(s0["sol"], (zl[1] + zr[1]) / 2, color="#6a3d9a", s=60, zorder=4)
        ax.scatter(s0["sol"], (zl[2] + zr[2]) / 2, color="#33a02c", s=60, marker="D", zorder=4)
        nm = key[0].split("_sol_")[-1][5:].replace("_zcam", "").replace("_", " ")
        ax.annotate(f"{nm}\n{T:.0f} °C", (s0["sol"], (zl[2] + zr[2]) / 2), xytext=(4, -24 if k_ % 2 == 0 else 10),
                    textcoords="offset points", fontsize=6.5, color="0.3")
    ax.axhline((nav["NL"][1] + nav["NR"][1]) / 2, color="#6a3d9a", ls=":", lw=1.2, label=f"Navcam pair centre y {(nav['NL'][1] + nav['NR'][1]) / 2:+.1f} mm")
    ax.axhline((nav["NL"][2] + nav["NR"][2]) / 2, color="#33a02c", ls=":", lw=1.2, label=f"Navcam height z {(nav['NL'][2] + nav['NR'][2]) / 2:.1f} mm")
    ax.scatter([], [], color="#6a3d9a", s=60, label=f"Z{zoom} pair centre y"); ax.scatter([], [], color="#33a02c", marker="D", s=60, label=f"Z{zoom} pair height z")
    ax.set_xlim(0, 2000); ax.set_xlabel("sol"); ax.set_ylabel("mm (mast frame)"); ax.set_ylim(-85, 5); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.3, loc="center right")
    ax.set_title(f"Sol and temperature ({len(pairs)} station{'s' if len(pairs) != 1 else ''} at {zoom} mm)", fontsize=10)
    ax = axs[1][1]; mks = ["o", "^", "v", "D", "P", "X", "<", ">"]; allA = []
    keys = sorted({(r["blk"], r["st"]) for r in rz}, key=lambda k: -sum((r["blk"], r["st"]) == k for r in rz))
    for k_, key in enumerate(keys):
        for c in ("ZL", "ZR"):
            rr = [r for r in rz if (r["blk"], r["st"]) == key and r["cam"] == c and r["focus"] is not None]
            sz = [s for s in SZ if (s["blk"], s["st"]) == key and s["cam"] == c]
            if len(rr) < 8 or not sz:
                continue
            bz = np.array(sz[0]["boresight"])
            a = np.array([np.dot(r["m"], bz) for r in rr]); f = np.array([r["focus"] for r in rr], float)
            lab = f"{key[0].split('_sol_')[-1][5:].replace('_zcam', '').replace('_', ' ')} {c} ({len(rr)})"
            if np.ptp(f) > 150 and len(rr) >= 15:
                ok = np.abs(a - np.median(a)) < 4 * rsd(a) + 1e-6
                lab += f": {np.polyfit(f[ok] / 1000, a[ok], 1)[0]:+.1f} mm / 1000 counts (robust)"
            ax.scatter(f, a, s=10, marker=mks[k_ % len(mks)], color=COL[c], alpha=0.5, label=lab); allA.extend(a)
    if allA:
        lo, hi = np.percentile(allA, [4, 96]); ax.set_ylim(lo - 25, hi + 25); ax.legend(fontsize=6.3, loc="best")
    ax.set_xlabel(f"Z{zoom} focus motor count"); ax.set_ylabel("centre along the Z boresight (mm, mast frame)"); ax.grid(alpha=0.3)
    ax.set_title("Focus: the along-axis centre trades with the focal law", fontsize=10)
    panel_letters(axs.ravel())
    fig.suptitle(f"Mastcam-Z {zoom} mm and Navcam on the mast head{', ' + label if label else ''}: Z frames through the deck_joint chain "
                 f"(resolver angles); {len(rz)} frames at {len({(s['blk'], s['st']) for s in SZ})} stations\n"
                 "mast frame: origin the elevation axis on the azimuth axis, x the level boresight, y right, z down; Navcam from the chain, "
                 "Z from each station's BA poses", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(out) / f"zcam{zoom}_navcam_on_mast_head_{tag}.png"
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path


def fig_vs_zoom(S, out, tag, dpi=200, label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mppp.calib.constrain import panel_letters
    Z = sorted({s["zoom"] for s in S})
    if len(Z) < 2 or 34 not in Z:
        return None
    full = {}                                   # the station with the most zooms (both eyes), joined
    for s in S:
        full.setdefault((s["blk"], s["st"]), set()).add(s["zoom"])
    ref_st = max(full, key=lambda k: (len(full[k]), -min(s["sol"] for s in S if (s["blk"], s["st"]) == k)))
    mk = {"ZL": "o", "ZR": "s"}; dz = {"ZL": -1.2, "ZR": 1.2}
    get = lambda s, key, i: None if s[key] is None else s[key][i]
    specs = [("xyz", 1, "y change from 34 mm (mm, + right)", "Centre across the boresight, y", "mm"),
             ("xyz", 2, "z change from 34 mm (mm, + down)", "Centre height z", "mm"),
             ("centre_mrad", 0, "image-centre yaw change from 34 mm (mrad, + right)", "Pointing of the image-centre pixel: yaw", "mrad"),
             ("centre_mrad", 1, "image-centre pitch change from 34 mm (mrad, + up)", "Pointing of the image-centre pixel: pitch", "mrad")]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.0))
    nm = ref_st[0].split("_sol_")[-1][5:].replace("_zcam", "").replace("_", " ")
    for ax, (key, i, yl, tt, un) in zip(axs.ravel(), specs):
        for cam in ("ZL", "ZR"):
            ss = [s for s in S if s["cam"] == cam and get(s, key, i) is not None]
            r34 = [get(s, key, i) for s in ss if s["zoom"] == 34]
            if not r34:
                continue
            ref = np.median(r34)
            for s in ss:
                if (s["blk"], s["st"]) == ref_st:
                    continue
                ax.scatter(s["zoom"] + dz[cam], get(s, key, i) - ref, s=12 + 2.2 * np.sqrt(s["n"]), marker=mk[cam], color=COL[cam], alpha=0.45)
            sei = sorted([s for s in ss if (s["blk"], s["st"]) == ref_st], key=lambda s: s["zoom"])
            if sei and sei[0]["zoom"] == 34:
                r0 = get(sei[0], key, i)
                ax.plot([s["zoom"] + dz[cam] for s in sei], [get(s, key, i) - r0 for s in sei], "-", marker=mk[cam], ms=6, mec="k",
                        color=COL[cam], lw=1.4, label=f"{cam}: {nm} {ref_st[1]} ({len(full[ref_st])} zooms at one station), from its own 34 mm")
            med = [np.median([get(s, key, i) for s in ss if s["zoom"] == z]) - ref if any(s["zoom"] == z for s in ss) else np.nan for z in Z]
            ax.plot(np.array(Z) + dz[cam], med, "--", color=COL[cam], lw=1, alpha=0.7, label=f"{cam}: station median, from 34 mm = {ref:+.1f} {un}")
        ax.axhline(0, color="0.4", lw=0.8); ax.set_xticks(Z); ax.set_xlabel("zoom (mm)"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        ax.set_title(tt, fontsize=10); ax.legend(fontsize=6.2, loc="best")
    axs[0][1].invert_yaxis()
    panel_letters(axs.ravel())
    fig.suptitle(f"Mastcam-Z on the mast head against zoom{', ' + label if label else ''}: Z blocks through the deck_joint chain (resolver angles); "
                 "points: medians of the other stations (size ~ sqrt frames)\nlevel mast frame (x the NL boresight, y right, z down); "
                 "far-field-only stations do not hold the centre (the 110 mm one)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(out) / f"zcam_mast_head_vs_zoom_{tag}.png"
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path


def zcam_blocks(root: Path):
    from mppp.waypoints.sites import load_site_table
    tab = load_site_table()
    names = list(dict.fromkeys((tab["groups"].get("zcam_consensus") or []) + (tab["groups"].get("zcam34_joint") or [])))
    out = []
    for n in names:
        for f in sorted(glob.glob(str(root / f"mars2020_sol_*_{n}_zcam"))):
            if (Path(f) / "colmap" / "run_done.json").is_file() and (Path(f) / "colmap" / "sparse" / "cahv_ba" / "images.bin").is_file():
                out.append(Path(f))
    return out


def run(root, chain, labels=None, blocks=None, out=None, tag="run", dpi=200, label=""):
    root = Path(root) if root else None
    out = Path(out or (HERE / "Claude outputs")); out.mkdir(parents=True, exist_ok=True)
    m = json.loads(Path(chain).read_text(encoding="utf-8"))
    mast = Mast(m)
    labels = labels or (root / "camera_analysis" / "rover_frame_labels.json")
    lab = json.loads(Path(labels).read_text(encoding="utf-8"))
    folders = [Path(b) for b in blocks] if blocks else zcam_blocks(root)
    rows, fits = [], {}
    for f in folders:
        try:
            r, ft = block_rows(f, lab, mast)
            rows += r; fits[f.name] = ft
            print(f"mast head: {f.name}: {len(r)} Z frames at {len(ft)} station(s)")
        except Exception as e:                                             # noqa: BLE001
            print(f"mast head: {f.name}: skipped ({type(e).__name__}: {e})")
    if not rows:
        print("mast head: no Z frames with resolver angles - nothing drawn")
        return []
    S = station_table(rows)
    nav = {c: mast.mast_point(np.asarray(m["params"][f"c_{c[1]}_m"])).tolist() for c in ("NL", "NR")}
    model = mount_model(S, rows, mast)
    rec = dict(chain=str(chain), labels=str(labels), el_level_deg=float(np.degrees(mast.el_level)), navcam_mast_mm=nav,
               station_fits=fits, stations=S, mount=model,
               symbols={"c": "camera centre from the elevation axis, head frame (m)", "Q": "camera <- head rotation (rotvec rad)",
                        "mast frame": "origin the elevation axis on the azimuth axis, x the level NL boresight, y right, z down (mm)",
                        "yaw / pitch": "mrad, + right / + up of the NL boresight in the level mast frame"})
    (out / f"zcam_mast_head_{tag}.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    paths = [fig_zoom_head(z, S, rows, nav, mast, out, tag, dpi, label) for z in sorted({s["zoom"] for s in S})]
    paths.append(fig_vs_zoom(S, out, tag, dpi, label))
    paths = [p for p in paths if p]
    for p in paths:
        print(f"mast head: {p}")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"))
    ap.add_argument("--chain", required=True, help="deck_joint_mast_model.json (the deck_joint BA's mast chain)")
    ap.add_argument("--labels", default=None, help="rover_frame_labels.json (default <root>\\camera_analysis\\)")
    ap.add_argument("--blocks", nargs="*", default=None, help="Z block folders (default: the finished zcam_consensus / zcam34_joint blocks)")
    ap.add_argument("--out", default=str(HERE / "Claude outputs"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--label", default="")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)
    return 0 if run(a.root, a.chain, a.labels, a.blocks, a.out, a.tag, a.dpi, a.label) else 1


if __name__ == "__main__":
    raise SystemExit(main())
