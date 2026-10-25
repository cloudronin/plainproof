"""EDM v2 development pilot: per-learner questions needed, to size the held-out run. Development learners only.

Cells: the confirmatory settings (d = 0 and 0.5, uniform and uncertainty outsourcing, AI accuracy 0.95, c = 0,
h = 0.3). Policies: SPRT, fixed (one seeded random order shared by all learners), random (own order per
learner), BOBCAT. Every learner's follow-up answers are drawn once into a table that all policies read, so
policies are paired on identical answers. Questions needed = first question count at which the decision
confidence reaches 0.90 (SPRT: its own classification step), up to 20; not reached is recorded as 21 (censored)
and counted. Not a reportable result: it measures spread to set held-out n.
Usage: python pilot.py <out_dir>
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'published_cat_reference_v1'))
sys.path.insert(0, str(REPO / 'src'))
import torch  # noqa: E402
from bobcat_gate import BANK_SEED, FIT, FIT_SHA, logistic_fit, state_hash  # noqa: E402
from compare_selection_v1 import adapt, setup  # noqa: E402
from plainproof import bank_from_fit, simulator  # noqa: E402
from plainproof.irt import p3pl  # noqa: E402
from plainproof.sprt import SPRTConfig, run_sprt  # noqa: E402

CELLS = [('uniform', 0.0), ('uniform', 0.5), ('uncertainty', 0.0), ('uncertainty', 0.5)]
N_DEV, MAX_Q, TARGET, UNREACHED = 500, 20, 0.90, 21
SEED0 = 2026102510
CFG = SPRTConfig(cut=0.0, delta=0.3, alpha=0.05, beta=0.05, n_max=MAX_Q)
GRID = np.linspace(-4, 4, 161)
PRIOR = np.exp(-GRID ** 2 / 2)


def posterior_master(a, b, c, items, answers):
    like = PRIOR.copy()
    for j, u in zip(items, answers):
        p = p3pl(GRID, a[j], b[j], c[j])
        like *= p if u else 1 - p
    like /= like.sum()
    return float(like[GRID >= CFG.cut].sum())


def first_reach(ps):
    for k, p in enumerate(ps, 1):
        if max(p, 1 - p) >= TARGET:
            return k
    return UNREACHED


def run_cell(t, pr, bank, q_ids, model, d, seed):
    world = simulator.World(bank, N_DEV, cut=0.0, margin=0.25, h_share=0.3, a_ai=0.95, model=model, d=d, seed=seed)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    table = np.array([[world.respond(i, j) for j in range(m)] for i in range(N_DEV)])     # one draw, all policies
    rng = np.random.default_rng(seed + 7)
    fixed_order = rng.permutation(m)
    out = {k: np.zeros(N_DEV, dtype=int) for k in ('sprt', 'fixed', 'random', 'bobcat')}
    for i in range(N_DEV):
        arm = run_sprt(fa, fb, fc, CFG, lambda j, i=i: int(table[i, j]))
        out['sprt'][i] = len(arm.state.items) if arm.state.decision in (0, 1) else UNREACHED
        for name, order in (('fixed', fixed_order), ('random', rng.permutation(m))):
            items = order[:MAX_Q]
            ps = [posterior_master(fa, fb, fc, items[:k], table[i, items[:k]]) for k in range(1, MAX_Q + 1)]
            out[name][i] = first_reach(ps)
    # BOBCAT: its own selection and adaptation; p_mastery per step from a logistic map fitted on even IDs.
    fu_q = q_ids[bank.rows('followup')]
    col_to_fu = {int(q): j for j, q in enumerate(fu_q)}
    B, n = N_DEV, pr.n_question
    available = torch.zeros((B, n), dtype=torch.long); available[:, torch.as_tensor(fu_q, dtype=torch.long)] = 1
    labels = torch.zeros((B, n), dtype=torch.long); mask = torch.zeros((B, n), dtype=torch.long)
    batch = {'input_labels': labels, 'input_mask': available.clone()}
    thetas = []
    for _ in range(MAX_Q):
        with torch.no_grad():
            _, action = t.st_policy.policy((labels.float() - .5) * 2 * mask, available)
        for i, col in enumerate(action.tolist()):
            labels[i, col] = int(table[i, col_to_fu[col]])
        available[torch.arange(B), action] = 0; mask[torch.arange(B), action] = 1
        batch['input_labels'] = labels
        thetas.append(adapt(t, batch, mask)[:, 0].numpy().copy())
    truth = np.array([l.truth for l in world.learners])
    known = truth >= 0
    calib = known & (np.arange(B) % 2 == 0)
    ps = []
    for th in thetas:
        w = logistic_fit(th[calib], truth[calib].astype(float))
        ps.append(1 / (1 + np.exp(-(w[0] + w[1] * th))))
    ps = np.array(ps).T
    out['bobcat'] = np.array([first_reach(row) for row in ps])
    # Spread on odd IDs only for BOBCAT pairing (even IDs fitted its map); all IDs for the others.
    odd = np.arange(B) % 2 == 1
    summ = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), unreached=int((v == UNREACHED).sum())) for k, v in out.items()}
    diffs = {}
    for other, sel in (('fixed', np.ones(B, bool)), ('random', np.ones(B, bool)), ('bobcat', odd)):
        dlt = (out['sprt'] - out[other])[sel]
        diffs[f'sprt-{other}'] = dict(n=int(sel.sum()), mean=float(dlt.mean()), sd=float(dlt.std(ddof=1)))
    return dict(model=model, d=d, seed=seed, learners=B, borderline=int((~known).sum()), followup_items=m,
                policies=summ, paired=diffs)


def n_needed(sd, effect=0.4, alpha=0.05 / 8, power=0.8):
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil((z * sd / effect) ** 2)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(FIT.read_bytes()).hexdigest() != FIT_SHA:
        raise SystemExit('item fit changed; stop')
    t0 = time.time()
    t, pr = setup()
    before = state_hash(t)
    bank, q_ids = bank_from_fit.build(bank_from_fit.load_fit(FIT), c=0.0, seed=BANK_SEED)
    cells = [run_cell(t, pr, bank, q_ids, mdl, d, SEED0 + k) for k, (mdl, d) in enumerate(CELLS)]
    assert state_hash(t) == before, 'BOBCAT weights changed'
    worst = max(c['paired'][k]['sd'] for c in cells for k in ('sprt-fixed', 'sprt-bobcat'))
    report = dict(cells=cells, worst_paired_sd=worst,
                  held_out_n_for_0_4_questions=n_needed(worst),
                  sizing='paired two-sided z, alpha 0.05/8 (Bonferroni bound on Holm), power 0.8, effect 0.4 questions; '
                         'censoring at 21 understates spread when many learners are unreached',
                  development_only=True, reportable=False, seconds=round(time.time() - t0, 1), paid_usd=0)
    (out / 'pilot.json').write_text(json.dumps(report, indent=1))
    for c in cells:
        print(c['model'], c['d'], {k: (round(v['mean'], 2), round(v['sd'], 2), v['unreached']) for k, v in c['policies'].items()},
              {k: (round(v['mean'], 2), round(v['sd'], 2)) for k, v in c['paired'].items()})
    print('worst paired sd', round(worst, 2), 'held-out n per cell', report['held_out_n_for_0_4_questions'], 'seconds', report['seconds'])


if __name__ == '__main__':
    main(sys.argv[1])
