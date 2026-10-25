"""Compare a retrained BOBCAT checkpoint with the original run: exact sha256 first, then the recorded result."""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ref = json.loads((ROOT / 'reports/receipts/bobcat-run-settings.json').read_text())
run = Path(os.environ.get('REPRO_DATA', 'data'), 'published-cat-audit-v1/bobcat-author-default-fold1-v1')
sha = hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()
done = json.loads((run / 'completion.json').read_text())
print('checkpoint sha256', 'MATCHES' if sha == ref['checkpoint_sha256'] else f'differs ({sha[:12]} vs {ref["checkpoint_sha256"][:12]})')
o = ref['original_result']
print('best epoch', done['best_epoch'], 'original', o['best_epoch'], '| test AUC', round(done['best_test_auc'], 6), 'original', round(o['best_test_auc'], 6))
if done['best_epoch'] != o['best_epoch']:
    print('The sealed code asserts best epoch 105, so the held-out rerun will stop. Use the published outputs (Zenodo) for the figures.')
    sys.exit(1)
