"""Item bank from the ASSIST2009 development fit, with synthetic families (ruling, 24 Oct 2026).

The ASSIST2009 file has no skill tags, so families are synthetic: items are sorted by difficulty, cut into
strata, and dealt into 12 families of 13 so each family spans the difficulty range. Two families are held out
for transfer only. Real q_ids are kept so BOBCAT can select among the same items.
"""
import json

import numpy as np

from .simulator import split_bank

N_FAMILIES, HELD_OUT = 12, (10, 11)


def load_fit(path):
    rows = json.loads(open(path).read())
    rows.sort(key=lambda r: r['q_id'])
    return rows


def build(fit_rows, c, seed):
    rng = np.random.default_rng(seed)
    q = np.array([r['q_id'] for r in fit_rows])
    a = np.array([r['a'] for r in fit_rows])
    b = np.array([r['b'] for r in fit_rows])
    order = np.argsort(b, kind='stable')
    family = np.empty(len(q), dtype=int)
    for s in range(0, len(order), N_FAMILIES):                  # one stratum = N_FAMILIES adjacent difficulties
        stratum = order[s:s + N_FAMILIES]
        family[stratum] = rng.permutation(N_FAMILIES)[:len(stratum)]
    bank = split_bank(a, b, np.full(len(q), float(c)), family, HELD_OUT, rng)
    return bank, q

PROBE_COUNT = 10   # ruling 25 Oct 2026: the hardest follow-up items by fitted difficulty, fixed before the freeze


def probes(bank, q_ids, count=PROBE_COUNT):
    """Global row indices and q_ids of the `count` hardest follow-up items (largest fitted b; ties by q_id).
    Probes are ordinary items for selection, stopping, calibration and every primary comparison; only the
    secondary signature analysis reads this list."""
    fu = bank.rows('followup')
    order = sorted(fu, key=lambda j: (-bank.b[j], int(q_ids[j])))[:count]
    return [int(j) for j in order], [int(q_ids[j]) for j in order]
