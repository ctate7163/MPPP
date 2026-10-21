"""The Mars 2020 example data set (mppp.example): selection, the zip in the archive layout, the manifest, fetch."""
import json
import zipfile

import pytest


def _touch(p, n=64):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    return p


def _archive(tmp_path):
    pds, zc = tmp_path / "datadrive", tmp_path / "zcam"
    nc = pds / "00461" / "ids" / "rdr" / "ncam"
    fc = pds / "00461" / "ids" / "rdr" / "fcam"
    for n in ("NLF_0461_0707870987_044RAD_N0260630NCAM03461_0A0195J01.IMG",
              "NLF_0461_0707870987_044RAD_N0260630NCAM03461_0A0195J02.IMG",       # a newer version of the same
              "NRF_0461_0707870987_044RAD_N0260630NCAM03461_0A0195J01.IMG",
              "NLF_0461_0707875039_745RAD_N0260630NCAM13461_0A0195J01.IMG",
              "NLF_0461_0707870543_258RAD_N0260630NCAM00708_0A0085J02.IMG",       # a drive image: not in the set
              "NLMV0461_0707867291_885RAD_N0260470VCE_16000_0A02LLJ01.IMG"):
        _touch(nc / n)
    _touch(fc / "FLF_0461_0707870635_633RAD_N0260630FHAZ02008_0A0085J02.IMG")
    _touch(fc / "FLF_0461_0707860000_633RAD_N0260470FHAZ02008_0A0085J02.IMG")    # another waypoint: not in the set
    for e in ("L", "R"):
        for t in ("0708130229_159", "0708130412_159", "0708130423_159", "0708130503_144"):
            _touch(zc / f"Z{e}0_0464_{t}RAD_N0260630ZCAM03381_0340LMA01.IMG")
        _touch(zc / f"Z{e}0_0464_0708130142_145RAD_N0260630ZCAM03381_034085A01.IMG")   # a thumbnail: not in the set
        _touch(zc / f"Z{e}4_0464_0708130280_159RAD_N0260630ZCAM03381_0340LMA01.IMG")   # another filter: not in the set
    return pds, zc


def test_select_and_build(tmp_path):
    from mppp import example as E
    pds, zc = _archive(tmp_path)
    sel = E.select_files(pds, zc)
    assert [p.name[-7:-4] for p in sel["navcam"]].count("J02") == 1 and len(sel["navcam"]) == 3
    assert sel["waypoints"] == ["0260630"] and len(sel["hazcam"]) == 1 and len(sel["zcam"]) == 8
    man = E.build_example(pds, zc, tmp_path / "out")
    with zipfile.ZipFile(man["zip"]) as zf:
        names = zf.namelist()
        m2 = json.loads(zf.read("example_manifest.json"))
    assert "datadrive/00461/ids/rdr/ncam/NRF_0461_0707870987_044RAD_N0260630NCAM03461_0A0195J01.IMG" in names
    assert sum(n.startswith("zcam/") for n in names) == 8 and m2["counts"] == {"navcam": 3, "hazcam": 1, "zcam": 8}
    assert (tmp_path / "out" / f"{E.EXAMPLE_NAME}.sha256").read_text().split()[0] == man["zip_sha256"]
    # fetch from a file URL, checked against the SHA-256
    url = (tmp_path / "out" / f"{E.EXAMPLE_NAME}.zip").as_uri()
    d = E.fetch_example(tmp_path / "dest", url=url)
    assert (d / "zcam").is_dir() and (d / "datadrive" / "00461").is_dir()
    with pytest.raises(ValueError):
        E.fetch_example(tmp_path / "dest2", url=url, sha256="0" * 64)
