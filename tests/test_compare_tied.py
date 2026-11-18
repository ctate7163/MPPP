import json


def test_compare_tied_reads_cost_and_slopes(tmp_path, capsys):
    """0.99.24: the fx/fy against k1(T) test is read off one table of the joints."""
    from rpp_studies.calibration import compare_tied as C
    for name, cost, k1 in (("a", 45780.4, -1.33e-5), ("b", 45929.9, 0.0)):
        d = tmp_path / name
        d.mkdir()
        eye = {"fx_ppm_per_degC": 38.5, "fy_ppm_per_degC": 38.5, "k1_per_degC": k1,
               "sd": {"fx_ppm_per_degC": 0.9, "fy_ppm_per_degC": 0.9, "k1_per_degC": 7e-7}}
        (d / "tied_joint.json").write_text(json.dumps({"final_cost": cost, "residuals": {"median_px": 0.1267, "rms_px": 0.2649},
                                                       "law": {"NL": eye, "NR": eye}}))
    out = tmp_path / "t.md"
    assert C.main([f"iso={tmp_path / 'a'}", f"k1T0={tmp_path / 'b'}", f"gone={tmp_path / 'c'}", "--out", str(out)]) == 0
    txt = out.read_text(encoding="utf-8")
    assert "| k1T0 | 45929.9 | +149.5 |" in txt and "-1.33 ± 0.07" in txt and "+38.5 ± 0.9" in txt
    assert "gone: no tied_joint.json" in capsys.readouterr().out
