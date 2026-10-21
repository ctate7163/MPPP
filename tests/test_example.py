"""The Mars 2020 example data set (mppp.example): the explicit file list, the zip in the archive layout, the
manifest, fetch."""
import json
import zipfile

import pytest


def _archive(tmp_path, skip=()):
    from mppp import example as E
    pds, zc = tmp_path / "datadrive", tmp_path / "zcam"
    for n in E.EXAMPLE_FILES:
        if n in skip:
            continue
        if n[0] == "Z":
            p = zc / n
        else:
            p = pds / f"{int(n.split('_')[1]):05d}" / "ids" / "rdr" / ("ncam" if n[0] == "N" else "fcam") / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 64)
    # not in the set: another Hazcam pair of sol 463, a drive image
    (pds / "00463" / "ids" / "rdr" / "fcam").mkdir(parents=True)
    (pds / "00463" / "ids" / "rdr" / "fcam" / "FLF_0463_0708057316_790RAD_N0260630FHAZ00206_0A0295J01.IMG").write_bytes(b"x")
    return pds, zc


def test_the_list():
    from mppp import example as E
    n = E.EXAMPLE_FILES
    assert len(n) == len(set(n)) == 32 + 4 + 8
    assert sum(x[:3] in ("FLF", "FRF") for x in n) == 4 and sum(x[0] == "Z" for x in n) == 8
    assert {x.split("_")[1] for x in n if x[0] == "F"} == {"0461", "0464"}


def test_select_and_build(tmp_path):
    from mppp import example as E
    pds, zc = _archive(tmp_path)
    sel = E.select_files(pds, zc)
    assert not sel["missing"] and len(sel["navcam"]) == 32 and len(sel["hazcam"]) == 4 and len(sel["zcam"]) == 8
    man = E.build_example(pds, zc, tmp_path / "out")
    with zipfile.ZipFile(man["zip"]) as zf:
        names = zf.namelist()
        m2 = json.loads(zf.read("example_manifest.json"))
    assert "datadrive/00461/ids/rdr/fcam/FLF_0461_0707870635_633RAD_N0260630FHAZ02008_0A0085J02.IMG" in names
    assert not any("FHAZ00206" in x for x in names)
    assert m2["counts"] == {"navcam": 32, "hazcam": 4, "zcam": 8}
    url = (tmp_path / "out" / f"{E.EXAMPLE_NAME}.zip").as_uri()
    d = E.fetch_example(tmp_path / "dest", url=url)
    assert (d / "zcam").is_dir() and (d / "datadrive" / "00464").is_dir()
    with pytest.raises(ValueError):
        E.fetch_example(tmp_path / "dest2", url=url, sha256="0" * 64)


def test_missing_files_are_named(tmp_path):
    from mppp import example as E
    gone = E.EXAMPLE_FILES[-1]
    pds, zc = _archive(tmp_path, skip=(gone,))
    with pytest.raises(FileNotFoundError, match=gone[:20]):
        E.build_example(pds, zc, tmp_path / "out")
