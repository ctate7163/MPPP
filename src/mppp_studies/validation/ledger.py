"""
mppp_studies.validation.ledger - the list of claims and their independent evidence, and its table for the report.

    python -m mppp_studies.validation.ledger [--out validation.md] [--ledger FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

FILE = Path(__file__).with_name("ledger.json")
LEVELS = {"A": "internal", "B": "new data, same method", "C": "a different measurement"}
STATUSES = ("confirmed", "consistent", "inconclusive", "contradicted", "open")


def load(path: Optional[Any] = None) -> List[Dict[str, Any]]:
    d = json.loads(Path(path or FILE).read_text(encoding="utf-8"))
    for e in d["entries"]:
        if e.get("level") not in LEVELS or e.get("status") not in STATUSES:
            raise ValueError(f"{e.get('id')}: level {e.get('level')!r} / status {e.get('status')!r}")
    return d["entries"]


def z_score(e: Dict[str, Any]) -> Optional[float]:
    if e.get("se") in (None, 0) or e.get("value") is None or e.get("expected") is None:
        return None
    return abs(float(e["value"]) - float(e["expected"])) / float(e["se"])


def add(entry: Dict[str, Any], path: Optional[Any] = None) -> None:
    """Add or replace (by ``id``) an entry of the ledger file."""
    f = Path(path or FILE)
    d = json.loads(f.read_text(encoding="utf-8"))
    d["entries"] = [e for e in d["entries"] if e["id"] != entry["id"]] + [entry]
    f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def table_md(entries: Optional[List[Dict[str, Any]]] = None) -> str:
    entries = load() if entries is None else entries
    n = {s: sum(1 for e in entries if e["status"] == s) for s in STATUSES}
    lines = ["## Independent validation", "",
             f"{len(entries)} claims: " + ", ".join(f"{v} {k}" for k, v in n.items() if v) + ". Level A: internal; B: new "
             "data, same method; C: a different measurement. z: the distance of the measured value from the expected "
             "one in standard errors.", "",
             "| claim | independent test | level | measured | expected | z | status |", "|---|---|---|---|---|---|---|"]
    for e in sorted(entries, key=lambda q: (STATUSES.index(q["status"]), q["id"])):
        z = z_score(e)
        val = "-" if e.get("value") is None else (f"{e['value']:g}" + (f" ± {e['se']:g}" if e.get("se") is not None else " (no se)")
                                                   + f" {e.get('unit', '')}")
        lines.append(f"| {e['claim']} | {e['test']} | {e['level']} | {val} | {e.get('expected', '-')} | "
                     f"{'-' if z is None else f'{z:.1f}'} | **{e['status']}** |")
    lines += ["", "Notes:", ""] + [f"- **{e['id']}** ({e.get('date', '')}, {e.get('source', '')}): {e['note']}"
                                   for e in entries if e.get("note")]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    md = table_md(load(a.ledger))
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        print(a.out)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
