"""v0p71: blocks that share images in the joint studies, unique scape labels, progress lines, the study shown live in
notebook 04, "all" ending with the consensus, and the notebook 05 dLMST figure."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_study_scripts_and_notebooks_v0p71():
    nb5 = json.loads((ROOT / "notebooks" / "05_error_analysis.ipynb").read_text(encoding="utf-8"))
    src5 = "".join("".join(c["source"]) for c in nb5["cells"] if c["cell_type"] == "code")
    j = src5.index("# |dLMST|: cross-station rate")
    assert "fig, ax = plt.subplots(figsize=(10, 6))\ndl_edges = np.arange(0, 9, 0.5)" in src5[j:j + 300]
