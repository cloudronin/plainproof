"""Reconstruct an original (sealed) file's bytes from its public copy.

The public copies differ from the sealed originals only by the fixed substitutions in substitutions.json (package
name, file locations, private data paths). Undoing them, in one longest-first pass, gives back the exact sealed
bytes, so the sha256 values in reports/*/freeze.json can be checked here.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / 'substitutions.json').read_text())


def public_path(original):
    for a, b in SPEC['paths']:
        if original.startswith(a):
            return b + original[len(a):]
    return original


def _reverse(text):
    table = {b: a for a, b in SPEC['subs']}
    pat = re.compile('|'.join(re.escape(k) for k in sorted(table, key=len, reverse=True)))
    return pat.sub(lambda m: table[m.group(0)], text)


def original_bytes(repo, original):
    p = Path(repo) / public_path(original)
    if p.suffix in ('.py', '.R', '.sh'):
        return _reverse(p.read_text()).encode()
    return p.read_bytes()
