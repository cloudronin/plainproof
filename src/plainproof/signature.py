"""Secondary, descriptive: signatures of outsourcing in per-learner logs (ruling 25 Oct 2026).

Six per-learner statistics, each computed on the first k follow-ups of the learner's SPRT order (k = 5, 10, 15,
20), plus all of the learner's homework and transfer items. The direction in which outsourcing is expected to
move each statistic is fixed here, before any held-out data exists; AUC is P(statistic is further in that
direction for an outsourcing-cell learner than for an honest-cell learner), with ties counted half.
Separation line: AUC 0.75, fixed now. No selection among the six.

  1 difficulty_correctness   point-biserial correlation of item difficulty b with correctness. Honest answers
                             fall off with difficulty (negative); outsourced answers do not. Expected: higher.
  2 lz                       standardised log-likelihood person fit (Drasgow et al. 1985) at the honest EAP
                             theta. Outsourcing misfits the item model. Expected: lower (statistic = -lz).
  3 llr_outsourcing          log likelihood of the answers under "honest at theta plus uniform outsourcing at d
                             with AI accuracy a" minus under "honest at theta", at the honest EAP theta.
                             Zero when d = 0. Expected: higher.
  4 hw_minus_fu              EAP theta from homework minus EAP theta from follow-ups. Follow-ups inflated by
                             outsourcing. Expected: lower (statistic = -(hw - fu)).
  5 fu_minus_transfer        EAP theta from follow-ups minus EAP theta from transfer items (always own answers).
                             Expected: higher.
  6 sprt_confident_wrong     share of the k SPRT steps at which calibrated confidence was at least 0.90 and the
                             implied decision disagreed with truth. Uses truth: a simulation-only check, not
                             something a deployed system can compute. Expected: higher.
  7 probe_excess             among the fixed probe items (the 10 hardest follow-ups) that the method actually asked
                             in its first k, observed hit rate minus the rate predicted at the learner's EAP theta
                             from those same k follow-ups. Computed per method on that method's own order, never
                             pooled; NaN when no probe was asked, and the NaN count is reported. Expected: higher.
Statistics 1 to 6 use the SPRT order; statistic 7 uses each method's order.
"""
import math

import numpy as np

from .irt import p3pl

CHECKPOINTS = (5, 10, 15, 20)
SEPARATION = 0.75
DIRECTION = dict(difficulty_correctness=+1, lz=-1, llr_outsourcing=+1, hw_minus_fu=-1, fu_minus_transfer=+1,
                 sprt_confident_wrong=+1, probe_excess=+1)
REQUIRED_LOG_FIELDS = ('item', 'difficulty', 'bank', 'correct', 'outsourced')
GRID = np.linspace(-4, 4, 161)


def eap(a, b, c, u):
    if len(u) == 0:
        return float('nan')
    ll = -GRID ** 2 / 2
    for aj, bj, cj, uj in zip(a, b, c, u):
        p = p3pl(GRID, aj, bj, cj)
        ll = ll + np.log(p if uj else 1 - p)
    w = np.exp(ll - ll.max())
    return float((GRID * w).sum() / w.sum())


def lz(a, b, c, u, theta):
    p = p3pl(theta, np.asarray(a), np.asarray(b), np.asarray(c))
    u = np.asarray(u, dtype=float)
    l0 = np.sum(u * np.log(p) + (1 - u) * np.log(1 - p))
    e = np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
    v = np.sum(p * (1 - p) * np.log(p / (1 - p)) ** 2)
    return float((l0 - e) / math.sqrt(v)) if v > 0 else float('nan')


def llr_outsourcing(a, b, c, u, theta, d, a_ai):
    p = p3pl(theta, np.asarray(a), np.asarray(b), np.asarray(c))
    m = (1 - d) * p + d * a_ai
    u = np.asarray(u, dtype=float)
    return float(np.sum(u * (np.log(m) - np.log(p)) + (1 - u) * (np.log(1 - m) - np.log(1 - p))))


def point_biserial(x, u):
    x, u = np.asarray(x, dtype=float), np.asarray(u, dtype=float)
    if len(x) < 2 or x.std() == 0 or u.std() == 0:
        return 0.0
    return float(np.corrcoef(x, u)[0, 1])


def learner_stats(log, bank, sprt_order, k, d, a_ai, calibrated_conf, truth):
    """log: the world's rows for one learner. sprt_order: follow-up global item ids in SPRT order.
    calibrated_conf: calibrated SPRT confidence per step (len >= k)."""
    for r in log:
        for f in REQUIRED_LOG_FIELDS:
            if f not in r:
                raise ValueError(f'log row lacks {f}')
    fu = {r['item']: r for r in log if r['bank'] == 'followup'}
    asked = [fu[j] for j in sprt_order[:k]]
    hw = [r for r in log if r['bank'] == 'homework']
    tr = [r for r in log if r['bank'] == 'transfer']

    def arr(rows):
        it = [r['item'] for r in rows]
        return bank.a[it], bank.b[it], bank.c[it], [r['correct'] for r in rows]

    fa, fb, fc, fu_u = arr(asked)
    th_fu = eap(fa, fb, fc, fu_u)
    th_hw = eap(*arr(hw))
    th_tr = eap(*arr(tr))
    conf = np.asarray(calibrated_conf[:k])
    if truth in (0, 1):
        wrong = [(p >= 0.9 and truth == 0) or (p <= 0.1 and truth == 1) for p in conf]
        cw = float(np.mean(wrong))
    else:
        cw = float('nan')
    return dict(difficulty_correctness=point_biserial(fb, fu_u), lz=lz(fa, fb, fc, fu_u, th_fu),
                llr_outsourcing=llr_outsourcing(fa, fb, fc, fu_u, th_fu, d, a_ai),
                hw_minus_fu=th_hw - th_fu, fu_minus_transfer=th_fu - th_tr, sprt_confident_wrong=cw,
                outsourced_share=float(np.mean([r['outsourced'] for r in asked])) if asked else float('nan'))


def auc(honest, outsourcing, direction):
    """P(outsourcing value is further in `direction` than an honest value); ties count half. NaNs dropped."""
    h = np.asarray([x for x in honest if np.isfinite(x)]) * direction
    o = np.asarray([x for x in outsourcing if np.isfinite(x)]) * direction
    if len(h) == 0 or len(o) == 0:
        return float('nan')
    allv = np.concatenate([h, o])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):                      # average ranks for ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_o = ranks[len(h):].sum()
    return float((r_o - len(o) * (len(o) + 1) / 2) / (len(h) * len(o)))


def probe_excess(log, bank, order, k, probe_rows):
    """Statistic 7 for one learner and one method's order (follow-up global item ids, in the order asked)."""
    fu = {r['item']: r for r in log if r['bank'] == 'followup'}
    asked = [fu[j] for j in order[:k]]
    it = [r['item'] for r in asked]
    theta = eap(bank.a[it], bank.b[it], bank.c[it], [r['correct'] for r in asked])
    hit = [r for r in asked if r['item'] in set(probe_rows)]
    if not hit:
        return float('nan')
    pj = [r['item'] for r in hit]
    predicted = float(np.mean(p3pl(theta, bank.a[pj], bank.b[pj], bank.c[pj])))
    return float(np.mean([r['correct'] for r in hit])) - predicted
