"""Check every sealed code file against both freezes (reports/edm-v2-e0/freeze.json, reports/edm-v2-e0b/freeze.json)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'edm_v2_e0'))
from repro_paths import original_bytes, public_path  # noqa: E402

bad = 0
for fz in ('reports/edm-v2-e0/freeze.json', 'reports/edm-v2-e0b/freeze.json'):
    F = json.loads((ROOT / fz).read_text())
    for orig, sha in sorted(F['code'].items()):
        ok = hashlib.sha256(original_bytes(ROOT, orig)).hexdigest() == sha
        bad += not ok
        print(('ok  ' if ok else 'FAIL'), fz.split('/')[1], public_path(orig))
print('all sealed files match' if not bad else f'{bad} mismatches')
sys.exit(1 if bad else 0)
