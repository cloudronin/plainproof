"""EDM v2 BOBCAT gate (hard, before the freeze). Development learners only.

BOBCAT's trained checkpoint (epoch 105) selects follow-ups only among the 156-item bank's follow-up items,
answers come from the simulator, and its own five-step adaptation updates theta after each answer, for 20
steps. Pass = every development learner completes, no action leaves the bank, the checkpoint weights are
unchanged, and a p_mastery (logistic map of the final theta, fitted on one half of the development learners)
can be scored by study/reliability.py on the other half. Writes a gate report; not a reportable result.
Usage: python bobcat_gate.py <out_dir>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'published_cat_reference_v1'))
sys.path.insert(0, str(REPO / 'src'))
import torch  # noqa: E402
from compare_selection_v1 import adapt, setup  # noqa: E402
from plainproof.reliability import matched_coverage  # noqa: E402
from plainproof import bank_from_fit, simulator  # noqa: E402

FIT = Path(__import__('os').environ.get('REPRO_DATA', 'data'), 'edm-v2-e0/item-fit/assist2009-dev-2pl.json')
FIT_SHA = '12a3b167cd9c9b78b3c3c7faf71b2e464b0943a454f6418faf0f99b1114cb7fc'
BANK_SEED = 2026102500
DEV_SEEDS = {('uniform', 0.0): 2026102501, ('uncertainty', 0.5): 2026102502}
N_DEV, STEPS = 200, 20


def state_hash(t):
    h = hashlib.sha256()
    for sd in (t.model.state_dict(), t.st_policy.policy.state_dict()):
        for k in sorted(sd):
            h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
    h.update(t.meta_params[0].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def logistic_fit(x, y, iters=50):
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, X.T @ (y - p))
    return w


def run_cell(t, p, bank, q_ids, model, d, seed):
    world = simulator.World(bank, N_DEV, cut=0.0, margin=0.25, h_share=0.3, a_ai=0.95, model=model, d=d, seed=seed)
    fu = bank.rows('followup')
    fu_q = q_ids[fu]
    col_to_fu = {int(q): i for i, q in enumerate(fu_q)}
    B, n = N_DEV, p.n_question
    available = torch.zeros((B, n), dtype=torch.long)
    available[:, torch.as_tensor(fu_q, dtype=torch.long)] = 1
    labels = torch.zeros((B, n), dtype=torch.long)
    mask = torch.zeros((B, n), dtype=torch.long)
    batch = {'input_labels': labels, 'input_mask': available.clone()}
    out_of_bank = 0
    for _ in range(STEPS):
        with torch.no_grad():
            _, action = t.st_policy.policy((labels.float() - .5) * 2 * mask, available)
        for i, col in enumerate(action.tolist()):
            if col not in col_to_fu or available[i, col] != 1:
                out_of_bank += 1
                raise RuntimeError(f'action outside the follow-up bank: learner {i}, q_id {col}')
            labels[i, col] = world.respond(i, col_to_fu[col])
        available[torch.arange(B), action] = 0
        mask[torch.arange(B), action] = 1
        batch['input_labels'] = labels
        theta = adapt(t, batch, mask)
    asked = mask.sum(1)
    assert torch.all(asked == STEPS), 'a learner did not complete 20 steps'
    th = theta[:, 0].numpy()
    truth = np.array([l.truth for l in world.learners])
    keep = truth >= 0
    ids = np.arange(B)[keep]
    calib, score = ids[ids % 2 == 0], ids[ids % 2 == 1]
    w = logistic_fit(th[calib], truth[calib].astype(float))
    pm = 1 / (1 + np.exp(-(w[0] + w[1] * th[score])))
    records = [dict(id=int(i), p_mastery=float(v), truth=int(truth[i])) for i, v in zip(score, pm)]
    rel = {str(cov): {k: v for k, v in matched_coverage(records, cov).items() if k != 'selected_ids'}
           for cov in (1.0, 0.5)}
    return dict(model=model, d=d, seed=seed, learners=B, borderline=int((~keep).sum()), steps=STEPS,
                out_of_bank_actions=out_of_bank, completed=int((asked == STEPS).sum()),
                theta_quantiles=np.quantile(th, [0, .25, .5, .75, 1]).round(3).tolist(),
                logistic_map=w.round(4).tolist(), calibration_learners=len(calib), scored_learners=len(score),
                reliability=rel)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(FIT.read_bytes()).hexdigest() != FIT_SHA:
        raise SystemExit('item fit changed; stop')
    t0 = time.time()
    t, p = setup()
    before = state_hash(t)
    bank, q_ids = bank_from_fit.build(bank_from_fit.load_fit(FIT), c=0.0, seed=BANK_SEED)
    assert q_ids.max() < p.n_question
    cells = [run_cell(t, p, bank, q_ids, m, d, s) for (m, d), s in DEV_SEEDS.items()]
    after = state_hash(t)
    gate = dict(passed=bool(before == after and all(c['out_of_bank_actions'] == 0 and c['completed'] == N_DEV for c in cells)),
                weights_unchanged=before == after, weights_sha256=before, cells=cells,
                bank=dict(items=len(q_ids), followup=len(bank.rows('followup')), homework=len(bank.rows('homework')),
                          transfer=len(bank.rows('transfer')), seed=BANK_SEED, c=0.0),
                development_only=True, held_out_learners_used=False, reportable=False,
                seconds=round(time.time() - t0, 1), paid_usd=0,
                note='p_mastery is a logistic map of BOBCAT theta fitted on even development IDs and scored on odd ones; '
                     'mechanical check only, not a comparison result.')
    (out / 'bobcat-gate.json').write_text(json.dumps(gate, indent=1))
    print(json.dumps(dict(passed=gate['passed'], weights_unchanged=gate['weights_unchanged'],
                          cells=[dict(model=c['model'], d=c['d'], completed=c['completed'],
                                      out_of_bank=c['out_of_bank_actions']) for c in cells],
                          bank=gate['bank'], seconds=gate['seconds']), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
