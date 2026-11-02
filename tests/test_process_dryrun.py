"""mppp process --dry-run counts what processing would process (2 Nov 2026: it counted before the exclusions)."""
from pathlib import Path

from mppp.pipeline import Settings, apply_exclusions, product_summary, zcam_include_entries

NAMES = [
    "NLF_0460_0707782756_433RAD_N0260470NCAM03460_0A0195J01.IMG",
    "FLF_0460_0707782608_869RAD_N0260470FHAZ02418_0A0295J01.IMG",
    "ZL0_0460_0707786698_613RAD_N0260470ZCAM07114_0340LMA01.IMG",
    "ZL0_0466_0708307173_769RAD_N0260756ZCAM08486_1100LMA01.IMG",
    "ZR0_0466_0708307173_769RAD_N0260756ZCAM08486_1100LMA01.IMG",
    "ZL0_0525_0713500000_769RAD_N0261000ZCAM08548_1100LMA01.IMG",
    "ZL0_0526_0713600000_769RAD_N0261000ZCAM08548_1100LMA01.IMG",
]


def test_zcam_include_applies_to_the_dry_run_count():
    paths = [Path("D:/data") / n for n in NAMES]
    s = Settings().replace(exclude=zcam_include_entries({"110": ["ZCAM08486"]}))
    kept = apply_exclusions(paths, s)
    names = {p.name for p in kept}
    assert len(kept) == 5 and not any("ZCAM08548" in n for n in names)
    lines = product_summary(kept)
    assert lines[0].startswith("products by camera:") and "Mastcam-Z 110 2" in lines[0] and "Navcam 1" in lines[0]
    assert "Mastcam-Z 110 mm: 2 frames in 1 sequences" in lines and any("ZCAM08486  sol 466  2 frames" in x for x in lines)


def test_summary_lists_every_sequence_with_its_sols():
    lines = product_summary([Path(n) for n in NAMES])
    assert any("ZCAM08548  sol 525-526  2 frames" in x for x in lines)
    assert "Mastcam-Z 110 mm: 4 frames in 2 sequences" in lines
