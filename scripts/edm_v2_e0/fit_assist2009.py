"""E0 setup: fit 2PL item parameters on ASSIST2009 development learners only (hashed-ID half).

Writes the fitted parameters and a receipt (source hash, split rule, item filter, fit settings, output hash)
to the private store. The held-out half is never read beyond its IDs. Not a reportable result.
Usage: python fit_assist2009.py <out_dir>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from plainproof.fit import dev_learner, fit_2pl  # noqa: E402

SRC = Path(__import__('os').environ.get('REPRO_DATA', 'data'), 'published-cat-audit-v1/arghosh-BOBCAT/data/train_task_assist2009.json')
SRC_SHA = 'c07ca9fa0f08207b06488cec527725938fe96e81e3c1762839e1ad0b84f974cc'
MIN_DEV_RESPONSES = 50   # 0 items reach 100 development learners; 156 reach 50


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SRC_SHA:
        raise SystemExit('source hash changed; stop')
    rows = [r for r in json.loads(raw) if dev_learner(r['user_id'])]
    counts = {}
    for r in rows:
        for q in set(r['q_ids']):
            counts[q] = counts.get(q, 0) + 1
    items = sorted(q for q, n in counts.items() if n >= MIN_DEV_RESPONSES)
    col = {q: j for j, q in enumerate(items)}
    resp = np.full((len(rows), len(items)), np.nan)
    for i, r in enumerate(rows):
        for q, y in zip(r['q_ids'], r['labels']):     # first attempt per item only
            j = col.get(q)
            if j is not None and np.isnan(resp[i, j]):
                resp[i, j] = y
    keep = ~np.all(np.isnan(resp), axis=1)
    resp = resp[keep]
    t0 = time.time()
    a, b, hist = fit_2pl(resp)
    params = [dict(q_id=int(q), a=float(a[j]), b=float(b[j]), n=int((~np.isnan(resp[:, j])).sum()))
              for q, j in col.items()]
    blob = json.dumps(params, sort_keys=True).encode()
    (out / 'assist2009-dev-2pl.json').write_bytes(blob)
    receipt = dict(source=str(SRC), source_sha256=SRC_SHA, split='sha256("edm-v2-e0:"+user_id)[0] % 2 == 0',
                   dev_learners=int(keep.sum()), items=len(items), min_dev_responses=MIN_DEV_RESPONSES,
                   first_attempt_only=True, model='2PL, c=0 (guessing set in the simulator, an assumption)',
                   quadrature=31, prior='log a ~ N(0, 0.5^2)', em_iterations=len(hist),
                   final_loglik=hist[-1], seconds=round(time.time() - t0, 2),
                   output_sha256=hashlib.sha256(blob).hexdigest(),
                   a_quantiles=np.quantile(a, [0, .25, .5, .75, 1]).round(3).tolist(),
                   b_quantiles=np.quantile(b, [0, .25, .5, .75, 1]).round(3).tolist(),
                   reportable=False, paid_usd=0)
    (out / 'assist2009-dev-2pl.receipt.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
