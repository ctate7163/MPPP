"""v0p71: blocks that share images in the joint studies, unique scape labels, progress lines, the study shown live in
notebook 04, "all" ending with the consensus, and the notebook 05 dLMST figure."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _scape(tmp_path, name):
    from helpers import _build_rec, _synthetic
    from mppp.sfm.navcal import Scape
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path / name, noise_native_px=0.2)
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    temps = {r["name"]: {"T": -20.0 + i % 5} for i, r in enumerate(proj.images)}
    return Scape(name, proj.root, proj, rec, temps)


def test_merge_scapes_keeps_images_shared_by_two_blocks(tmp_path):
    from mppp.sfm import navcal as NC
    a, b = _scape(tmp_path, "Bell Island"), _scape(tmp_path, "Bell Island (bell_island_large)")
    assert {im.name for im in a.rec.images.values()} == {im.name for im in b.rec.images.values()}   # the same images
    cams = {"NL": a.rec.cameras[1], "NR": a.rec.cameras[2]}
    rig = (np.eye(3), np.array([-0.4244, 0.0, 0.0]))
    rec, proj, idx = NC.merge_scapes([a, b], cams, rig, points_per_scape=None)
    n = a.rec.num_reg_images()
    assert rec.num_reg_images() == 2 * n and idx["duplicates"] == {"Bell Island (bell_island_large)": n}
    names = [r["name"] for r in proj.images]
    assert len(set(names)) == len(names) and all(r.get("source_name") for r in proj.images if "@" in r["name"])
    renamed = [r for r in proj.images if "@" in r["name"]]
    assert all(idx["temps"][r["name"]] == idx["temps"][r["source_name"]] for r in renamed)
    assert len(idx["images"]["Bell Island"]) == len(idx["images"]["Bell Island (bell_island_large)"]) == n


def test_discover_scapes_keeps_sites_with_the_same_label(tmp_path, monkeypatch):
    from mppp.sfm import sites as S
    rows = [{"label": "Bell Island", "site": "bell_island", "folder": str(tmp_path / "a"), "ok": True},
            {"label": "Bell Island", "site": "bell_island_large", "folder": str(tmp_path / "b"), "ok": True},
            {"label": "Van Zyl", "site": "van_zyl", "folder": str(tmp_path / "c"), "ok": True},
            {"label": "Bell Island", "site": "x", "folder": str(tmp_path / "d"), "ok": False}]
    monkeypatch.setattr(S, "scan_scapes", lambda *a, **k: [dict(r) for r in rows])
    out = S.discover_scapes(tmp_path, verbose=False)
    assert set(out) == {"Bell Island", "Bell Island (bell_island_large)", "Van Zyl"}
    assert out["Bell Island (bell_island_large)"] == tmp_path / "b"


def test_progress_line():
    import time
    from mppp.sfm.navcal import progress
    s = progress(3, 12, time.time() - 180, "leave-one-out")
    assert s.startswith("[leave-one-out 3/12, 3 min elapsed, ~9 min left, at ")


def test_fit_early_offsets_reports_rounds(monkeypatch, capsys):
    from types import SimpleNamespace
    from mppp.sfm import navcal as NC
    from mppp.sfm import navcal_consensus as NCC
    truth = np.array([0.0, 0.0, -0.6, -0.2])

    def fake_joint(rec, proj, temps, ppm, T0, max_iterations=50, rig_slopes=None, extra_map=None, **kw):
        d = extra_map.offsets - truth
        return {"final_cost": float(5e5 + 0.5 * d @ (400 * np.eye(4)) @ d), "observations": 10 ** 6, "iterations": 3}
    monkeypatch.setattr(NC, "joint_adjust", fake_joint)
    proj = SimpleNamespace(images=[{"image_id": 1, "sol": 60}, {"image_id": 2, "sol": 900}])
    NCC.fit_early_offsets(object(), proj, {}, 38.0, -20.0, {}, 380, start=(0.1, 0.1, -2.3, 0.4), verbose=True)
    out = capsys.readouterr().out
    assert "adjustment (at most) 1/" in out and "min left" in out and "round 1/" in out


def test_study_scripts_and_notebooks_v0p71():
    s = (ROOT / "scripts" / "navcam_calibration_study.py").read_text(encoding="utf-8")
    i = s.index("def cmd_all")
    assert "cmd_consensus(a, scapes_cfg, samples)" in s[i:i + 2000] and "NC.progress(" in s
    nb4 = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert "subprocess.Popen(" in src4 and 'STUDY_CMD = "consensus"' in src4
    md4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "markdown")
    assert "between sols 278 and 360" in md4 and "near sol 250" not in md4 and "promote_cmods.py" in md4
    nb5 = json.loads((ROOT / "notebooks" / "05_error_analysis.ipynb").read_text(encoding="utf-8"))
    src5 = "".join("".join(c["source"]) for c in nb5["cells"] if c["cell_type"] == "code")
    j = src5.index("# |dLMST|: cross-station rate")
    assert "fig, ax = plt.subplots(figsize=(10, 6))\ndl_edges = np.arange(0, 9, 0.5)" in src5[j:j + 300]
