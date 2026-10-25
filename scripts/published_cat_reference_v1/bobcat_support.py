"""Versioned BOBCAT execution bounds and atomic private receipts; no old policy edits."""
import json, os
from pathlib import Path
RUN_SECONDS = 120 * 3600
CUMULATIVE_SECONDS = 144 * 3600
NEW_BYTES = 2 * 1024**3
RSS_BYTES = 8 * 1024**3
THREADS = 6
ROOT = Path(__import__('os').environ.get('REPRO_DATA', 'data'), 'published-cat-audit-v1')
RUN = ROOT / 'bobcat-author-default-fold1-v1'
REPO = Path(__file__).resolve().parents[2]  # (public) repo root

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def stop_reason(elapsed, charged, rss, growth, retained, free, pressure):
    if elapsed > RUN_SECONDS: return '120h per-run ceiling'
    if charged + elapsed > CUMULATIVE_SECONDS: return '144h cumulative ceiling'
    if rss > RSS_BYTES: return '8GiB run RSS'
    if growth > NEW_BYTES: return '2GiB additional storage'
    if retained > 32 * 1024**3: return '32GiB aggregate retained storage'
    if free < 20 * 1024**3: return '20GiB free disk reserve'
    if pressure != 1: return 'memory pressure'
    return None
