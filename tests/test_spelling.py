"""0.99.2: American spelling (Christian, 16 Nov 2026: "scrub the code, text, and documentation for centre and colour and
other British-specific spelling. Use center and color.").  British spelling stays only in names that files, run keys
or other packages read (rpp.ondisk: SPELLING_KEPT_NAMES and the keys and file names listed there)."""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRITISH = ("colour colours coloured colouring colourful colourmap colourbar uncoloured centre centres centred centring "
           "recentre recentred grey greys greyed greyish greyscale greyshade darkgrey lightgrey dimgrey slategrey metre "
           "metres decimetres centimetres millimetres kilometres micrometre behaviour behaviours behavioural neighbour "
           "neighbours neighbouring neighbourhood honour honours honoured favour favourite labelled labelling "
           "mislabelled relabelled relabelling unlabelled levelled levelling modelled modelling modeller travelled "
           "cancelled acknowledgement acknowledgements aluminium artefact artefacts towards whilst amongst judgement "
           "analyse analysed analyser analysing initialise initialised initialisation normalise normalised normalises "
           "normalisation optimise optimised optimiser optimisation summarise summarised minimise minimised minimises "
           "minimising recognise recognised realise realised localised localisation linearised generalised quantise "
           "quantisation regularisation parameterised vectorised characterise symmetrised harmonised neutralised "
           "marginalised capitalised parenthesised factorisation").split()
WORD = re.compile(r"(?i)(?<![A-Za-z0-9_])(" + "|".join(sorted(BRITISH, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
SKIP = re.compile(r"^(docs/history/|LICENSE|NOTICE|.*\.bib$|.*/vendor/|src/rpp/data/cmods/(?!README\.md$)|tests/test_spelling\.py$)")
TEXT = re.compile(r"\.(py|md|json|ipynb|html|js|css|bat|yml|yaml|toml|cff|txt|cfg|in)$")
KEEP_TOKENS = ("colour-demosaicing", "colour_demosaicing", "colour-science")
SPAN = re.compile(r"``[^`\n]+``|`[^`\n]+`|\"[A-Za-z0-9_.\-/:]+\"|'[A-Za-z0-9_.\-/:]+'|\\\"[A-Za-z0-9_.\-/:]+\\\"")


def _files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        pytest.skip("not a git checkout")
    return [f for f in out.stdout.splitlines() if TEXT.search(f) and not SKIP.match(f) and (ROOT / f).is_file()]


def _standalone(text: str):
    """British words used as words: not part of a name (snake_case, a path, a flag) and not a single token in quotes
    or backticks (a key named in the text)."""
    text = SPAN.sub(" ", text)
    for k in KEEP_TOKENS:
        text = text.replace(k, " ")
    for m in WORD.finditer(text):
        a, b = m.start(), m.end()
        before, after = text[a - 1:a], text[b:b + 1]
        if before in "./\\" and a >= 2 and text[a - 2].isalnum():
            continue                                    # a path or a dotted name
        if after in "./\\" and text[b + 1:b + 2].isalnum():
            continue
        if before == "-" and not (a >= 2 and text[a - 2].isalpha()):
            continue                                    # --a-flag
        yield m.group(0)


def test_text_and_code_use_american_spelling():
    bad = []
    for f in _files():
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            for w in _standalone(line):
                bad.append(f"{f}:{n}: {w}")
    assert not bad, "British spelling (0.99.2: American; keys and file names are in rpp.ondisk):\n" + "\n".join(bad[:40])


def test_british_names_in_the_code_are_only_the_kept_ones():
    from rpp.ondisk import SPELLING_KEPT_NAMES
    stem = re.compile(r"(?i)colour|centre|grey|metre|neighbour|behaviour|normalis|optimis|summaris|quantis|labelled")
    found = {}
    for f in _files():
        if not f.endswith(".py"):
            continue
        try:
            tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.keyword) and node.arg:
                names = [node.arg]
            for n in names:
                if stem.search(n) and n not in SPELLING_KEPT_NAMES:
                    found.setdefault(n, f)
    assert not found, f"British names in the code: {found}"


def test_eyes_center_renders_the_pivot_file_of_before():
    import argparse
    from rpp.ibr import cli
    from rpp.ondisk import PIVOT_EYE
    p = argparse.ArgumentParser()
    cli.add_parsers(p.add_subparsers())
    for word in ("center", "centre"):
        a = p.parse_args(["ods", "W", "--station", "3_0", "--eyes", "left", "right", word])
        assert cli._eyes(a.eyes) == ["left", "right", PIVOT_EYE] == ["left", "right", "centre"]
