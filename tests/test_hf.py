"""0.99.3: the panoramas on Hugging Face (rpp.ibr.hf) - Christian, 16 Nov 2026: "v0p99 should upload rpv products to
hugging face near to where the mask CNN are" and "Prepare ibr run to reprocess everything and upload to hugging face"."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("huggingface_hub")


class FakeApi:
    def __init__(self, fail=()):
        self.commits, self.files, self.repos, self.fail = [], {}, [], set(fail)

    def create_repo(self, repo_id, repo_type=None, exist_ok=False, private=False):
        self.repos.append((repo_id, repo_type, private))

    def file_exists(self, repo_id, path, repo_type=None):
        return path in self.files

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=""):
        self.files[path_in_repo] = path_or_fileobj

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        label = commit_message.split(":")[0]
        if label in self.fail:
            raise ConnectionError("the network is down")
        self.commits.append((repo_id, repo_type, [o.path_in_repo for o in operations], commit_message))
        for o in operations:
            self.files[o.path_in_repo] = o.path_or_fileobj
        return type("Info", (), {"oid": f"c{len(self.commits):03d}0000000000"})()


def _station(root: Path, label: str):
    for sub, pre in (("", label), ("with_rover", label + "_rover")):
        d = root / label / sub
        d.mkdir(parents=True, exist_ok=True)
        for part in ("", "_enhlocal"):
            cv2.imwrite(str(d / f"{pre}{part}_360_3DTD.jpg"), np.full((64, 64, 3), 90, np.uint8))
        cv2.imwrite(str(d / f"{pre}_360_3DTD_16.png"), np.full((64, 64, 3), 9000, np.uint16))
        (d / f"{pre}_ods.json").write_text(json.dumps({"station": label}), encoding="utf-8")
        cv2.imwrite(str(d / f"{pre}_ods_left_16.png"), np.zeros((8, 8, 3), np.uint16))      # not in the record
    (root / label / "station.json").write_text("{}", encoding="utf-8")
    (root / label / "tone_levels.json").write_text("{}", encoding="utf-8")


def test_stations_go_to_the_dataset_and_panos_json_points_at_their_commit(tmp_path):
    from rpp.ibr import hf, web
    out, w = tmp_path / "single_sites", tmp_path / "RPV"
    _station(out, "Sol0001_S001D0001")
    _station(out, "Sol0002_S001D0002")
    (w / "panos").mkdir(parents=True)
    old_pic = w / "panos" / "Sol0002_S001D0002_360_3DTD.jpg"                 # the GitHub era
    old_pic.write_bytes(b"jpg")
    (w / "panos" / "welcome_muted_360_3DTD.jpg").write_bytes(b"jpg")
    before = {"panos": [{"id": "Sol0002_S001D0002", "title": "two", "variants": [{"key": "masked", "views": {"standard": "panos/" + old_pic.name}}]}]}
    (w / "panos.json").write_text(json.dumps(before), encoding="utf-8")
    labels = ["Sol0001_S001D0001", "Sol0002_S001D0002"]
    doc = web.publish(out, w, stations=labels, width=32, files_dir=hf.stage_dir(out))
    stage = hf.stage_dir(out)
    assert (stage / "Sol0001_S001D0001_360_3DTD.jpg").is_file() and not (w / "panos" / "Sol0001_S001D0001_360_3DTD.jpg").exists()
    api = FakeApi(fail={"Sol0002_S001D0002"})
    res = hf.publish(w, doc, labels, out, previous=before, api=api)
    assert res["uploaded"] == ["Sol0001_S001D0001"] and list(res["failed"]) == ["Sol0002_S001D0002"] and not res["ok"]
    assert api.repos == [(hf.DATASET, "dataset", False)] and "README.md" in api.files and "license: cc-by-4.0" in api.files["README.md"].decode()
    repo, kind, paths, msg = api.commits[0]
    assert kind == "dataset" and msg.startswith("Sol0001_S001D0001: panoramas (RPP ")
    assert "Sol0001_S001D0001/viewer/Sol0001_S001D0001_360_3DTD.jpg" in paths and "Sol0001_S001D0001/viewer/Sol0001_S001D0001_rover_enhlocal_360_3DTD.jpg" in paths
    assert "Sol0001_S001D0001/record/Sol0001_S001D0001_360_3DTD_16.png" in paths and "Sol0001_S001D0001/record/with_rover/Sol0001_S001D0001_rover_ods.json" in paths
    assert "Sol0001_S001D0001/record/tone_levels.json" in paths and "Sol0001_S001D0001/record/station.json" in paths
    assert not any("ods_left_16" in p for p in paths)
    j = json.loads((w / "panos.json").read_text(encoding="utf-8"))
    one = next(p for p in j["panos"] if p["id"] == "Sol0001_S001D0001")
    url = one["variants"][0]["views"]["standard"]
    assert url == f"https://huggingface.co/datasets/{hf.DATASET}/resolve/c0010000000000/Sol0001_S001D0001/viewer/Sol0001_S001D0001_360_3DTD.jpg"
    assert one["hosted"] == "huggingface" and not hf.refs(one)
    two = next(p for p in j["panos"] if p["id"] == "Sol0002_S001D0002")       # failed: the entry of before, its picture kept
    assert two == before["panos"][0] and old_pic.is_file() and (w / "panos" / "welcome_muted_360_3DTD.jpg").is_file()
    api2 = FakeApi()                                                           # the next run: it goes up, the old picture goes
    doc2 = web.publish(out, w, stations=["Sol0002_S001D0002"], width=32, files_dir=stage)
    res2 = hf.publish(w, doc2, ["Sol0002_S001D0002"], out, previous=j, api=api2)
    assert res2["ok"] and not old_pic.exists() and "1 old picture(s) removed" in res2["message"]
    j2 = json.loads((w / "panos.json").read_text(encoding="utf-8"))
    assert all(not hf.refs(p) for p in j2["panos"] if p["id"].startswith("Sol"))


def test_card_check_and_the_single_sites_options(tmp_path, monkeypatch):
    import argparse
    from rpp.ibr import hf, single_sites as SS
    monkeypatch.setenv("RPP_HF_DATASET", "someone/panos")
    monkeypatch.setenv("RPP_HF_LICENSE", "cc0-1.0")
    assert hf.dataset() == "someone/panos" and "license: cc0-1.0" in hf.card("someone/panos")
    assert "NASA/JPL-Caltech/MSSS/ASU" in hf.card("x")
    p = argparse.ArgumentParser()
    SS.add_parser(p.add_subparsers())
    a = p.parse_args(["single-sites", "--upload"])
    assert a.host == "hf" and a.hf_record is True
    assert p.parse_args(["single-sites", "--upload", "--host", "github", "--no-hf-record"]).hf_record is False
    r = hf.check("someone/panos", timeout=0.01)          # the sandbox cannot reach it: a clear "no", not an exception
    assert r["ok"] is False and r["url"].endswith("/resolve/main/README.md")


class MovingApi(FakeApi):
    """0.99.12: a hub that holds the dataset of the earlier name."""
    def __init__(self, existing=("ctate7163/rpv-panoramas",)):
        super().__init__()
        self.existing, self.moves = set(existing), []

    def repo_exists(self, repo_id, repo_type=None):
        return repo_id in self.existing

    def move_repo(self, from_id, to_id, repo_type=None):
        self.moves.append((from_id, to_id, repo_type))
        self.existing.discard(from_id)
        self.existing.add(to_id)

    def create_repo(self, repo_id, repo_type=None, exist_ok=False, private=False):
        super().create_repo(repo_id, repo_type, exist_ok, private)
        self.existing.add(repo_id)


def test_the_rpp_rpv_dataset_takes_over_the_earlier_one(monkeypatch):
    """Christian, 17 Nov 2026: "there should be a separate rpp-rpv repository" - the morning's rpv-panoramas is renamed,
    once; another dataset named by RPP_HF_DATASET is left alone; panos.json's old addresses follow."""
    from rpp.ibr import hf
    from rpp import env as rpp_env
    for n in rpp_env.names("HF_DATASET"):                        # the new name and the old one
        monkeypatch.delenv(n, raising=False)
    assert hf.dataset() == "ctate7163/rpp-rpv"
    api = MovingApi()
    assert hf.ensure_dataset(api=api) == "ctate7163/rpp-rpv"
    assert api.moves == [("ctate7163/rpv-panoramas", "ctate7163/rpp-rpv", "dataset")]
    hf.ensure_dataset(api=api)                                   # the second time: nothing to move
    assert len(api.moves) == 1
    other = MovingApi()
    assert hf.ensure_dataset("someone/panos", api=other) == "someone/panos" and not other.moves
    fresh = MovingApi(existing=())                               # neither there: created
    assert hf.ensure_dataset(api=fresh) == "ctate7163/rpp-rpv" and not fresh.moves
    assert ("ctate7163/rpp-rpv", "dataset", False) in fresh.repos
    e = {"id": "S", "views": {"standard": "https://huggingface.co/datasets/ctate7163/rpv-panoramas/resolve/abc/S/viewer/a.jpg"},
         "note": "rpv-panoramas in words stays"}
    n = hf.old_to_new(e, "ctate7163/rpp-rpv")
    assert n["views"]["standard"] == "https://huggingface.co/datasets/ctate7163/rpp-rpv/resolve/abc/S/viewer/a.jpg"
    assert n["note"] == e["note"]

