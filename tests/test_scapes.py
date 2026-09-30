"""Metashape scape definitions (mppp.scapes). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import pytest


def _touch(d, name):
    (d / name).write_bytes(b"x")
    return d / name


def _name(cam, sol, seq="SAPP00601", spec="_0A0", ds="0", ver="01", sd="N0332864"):
    s = f"{cam}_{sol:04d}_0729883381_848RAD_{sd}{seq}{spec}{ds}LLJ{ver}.IMG"
    assert len(s) == 58
    return s


def test_rockytop_groups_1_2_and_no_z110(tmp_path):
    from mppp.scapes import select_scape
    keep = [_name("NLF", 470, ds="1"),                                     # group 1 (_0A01) and group 2
            _name("NRF", 470),                                             # group 2 Navcam
            _name("FLF", 500), _name("FRF", 500),                          # group 2 Hazcams
            _name("ZL0", 480, seq="ZCAM08400", spec="_034"),               # group 2 Z34
            _name("ZR0", 480, seq="ZCAM08400", spec="_034")]
    drop = [_name("ZL0", 507, seq="ZCAM08529", spec="_110"),               # group 3 Z110 mosaic
            _name("ZL0", 518, seq="ZCAM08541", spec="_063"),               # group 3 Z063 (not _034)
            _name("NLF", 459), _name("NLF", 536),                          # outside sols 460-535
            _name("NRF", 470, ver="00")]                                   # superseded version
    for n in keep + drop:
        _touch(tmp_path, n)
    paths, rep = select_scape("Rockytop", tmp_path)
    assert sorted(p.name for p in paths) == sorted(keep)
    assert rep["groups"] == [1, 2] and rep["n_superseded_versions"] == 1
    assert rep["by_camera_group"] == {"FL0": 1, "FR0": 1, "NL1": 1, "NR0": 1, "ZL034": 1, "ZR034": 1}
    paths3, _ = select_scape("rockytop", tmp_path, groups=(1, 2, 3))    # group 3 still reachable
    assert _name("ZL0", 518, seq="ZCAM08541", spec="_063") in {p.name for p in paths3}
    assert _name("ZL0", 507, seq="ZCAM08529", spec="_110") not in {p.name for p in paths3}


def test_all_zooms_except_110_where_workspace_took_all(tmp_path):
    from mppp.scapes import select_scape
    z = {mm: _name("ZL0", 800, seq="ZCAM09000", spec=f"_{mm:03d}") for mm in (34, 48, 63, 110)}
    for n in list(z.values()) + [_name("NLF", 800), _name("FLF", 800)]:
        _touch(tmp_path, n)
    paths, rep = select_scape("belva", tmp_path)
    names = {p.name for p in paths}
    assert names == {z[34], z[48], z[63], _name("NLF", 800)}            # no Hazcams at Belva, no Z110
    assert rep["n_dropped_zoom"] == 1


@pytest.mark.parametrize("scape,sols", [("rockytop", (460, 535)), ("landing", (9, 48)), ("belva", (770, 835)),
                                         ("bunsen", (1055, 1095)), ("hellandfjellet", (1601, 1645))])
def test_definitions_match_workspace(scape, sols):
    from mppp.scapes import SCAPES
    assert SCAPES[scape]["sols"] == sols
    with pytest.raises(KeyError):
        from mppp.scapes import select_scape
        select_scape("bright_angel", ".")
