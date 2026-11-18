import os


def test_free_space_report_counts_links_once_and_removes_only_what_is_named(tmp_path, capsys):
    """0.99.24 (Christian, 18 Nov 2026: "let's remove the high data volume products from D:\\rover_blocks starting with
    the colmap databases and the redundant images"): the report first; a hard-linked frame takes its space once and
    removing one name frees nothing; a removal needs the word, takes the named category only, and lists what went."""
    from rpp_studies.misc import free_space as F
    b = tmp_path / "mars2020_sol_0001_x"
    (b / "processed").mkdir(parents=True)
    (b / "colmap" / "images").mkdir(parents=True)
    (b / "colmap" / "dense_block_N" / "stereo").mkdir(parents=True)
    (b / "colmap" / "sparse" / "0").mkdir(parents=True)
    (b / "processed" / "a.png").write_bytes(b"x" * 3000)
    os.link(b / "processed" / "a.png", b / "colmap" / "images" / "a.png")
    (b / "colmap" / "database.db").write_bytes(b"d" * 5000)
    (b / "colmap" / "database.db-wal").write_bytes(b"d" * 100)
    (b / "colmap" / "dense_block_N" / "stereo" / "d.bin").write_bytes(b"s" * 700)
    (b / "colmap" / "sparse" / "0" / "images.bin").write_bytes(b"m" * 50)
    (tmp_path / "_transfer").mkdir()
    (tmp_path / "_transfer" / "keep.db").write_bytes(b"k" * 10)
    (tmp_path / "msl_sol_0124_y").mkdir()
    (tmp_path / "msl_sol_0124_y" / "database.db").write_bytes(b"k" * 10)
    files = F.scan(tmp_path, progress=False)
    cats = {"/".join(f["rel"]): f["cat"] for f in files}
    assert cats["mars2020_sol_0001_x/colmap/database.db-wal"] == "colmap-db" and cats["mars2020_sol_0001_x/colmap/sparse/0/images.bin"] == "other"
    assert cats["mars2020_sol_0001_x/colmap/dense_block_N/stereo/d.bin"] == "dense"
    imgs = [f for f in files if f["cat"] == "images"]
    assert F.on_disk(imgs) == 3000 and F.freed_by(files, lambda f: f["rel"][1:2] == ("processed",)) == 0
    assert F.freed_by(files, lambda f: f["cat"] == "images") == 3000
    assert F.main(["--root", str(tmp_path), "--delete", "colmap-db"]) == 2 and (b / "colmap" / "database.db").is_file()
    assert F.main(["--root", str(tmp_path), "--delete", "colmap-db", "--confirm", "DELETE", "--keep", "msl_"]) == 0
    assert (tmp_path / "msl_sol_0124_y" / "database.db").is_file()
    assert not (b / "colmap" / "database.db").exists() and not (b / "colmap" / "database.db-wal").exists()
    assert (tmp_path / "_transfer" / "keep.db").is_file() and (b / "processed" / "a.png").is_file()
    assert (b / "colmap" / "dense_block_N" / "stereo" / "d.bin").is_file() and (b / "colmap" / "sparse" / "0" / "images.bin").is_file()
    lists = list((tmp_path / "mppp_pipeline").glob("free_space_removed_*.txt"))
    assert len(lists) == 1 and "2 files" in lists[0].read_text() and "database.db" in lists[0].read_text()
    assert "| colmap-db |" in (tmp_path / "mppp_pipeline" / "free_space_report.md").read_text()
