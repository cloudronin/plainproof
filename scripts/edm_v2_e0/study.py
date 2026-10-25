"""EDM v2 development stages before the freeze (rulings of 25 Oct 2026). Development learners only.

  tune       choose SPRT width delta and maximum length on tuning learners (declared rule below)
  calibrate  fit one isotonic calibration per method, once, on the same calibration learners for every method
  pilot      rerun the pilot with calibrated confidences and 10 seeded fixed orders; record the variance

Each stage uses its own development seeds. Methods: SPRT (adapts stopping: same order for everyone, stops on
its log ratio), fixed (10 seeded orders shared by all learners), random (own order per learner), BOBCAT (adapts
selection). Raw confidence per step: SPRT logistic(log ratio); fixed/random exact posterior P(theta >= cut);
BOBCAT its theta. Calibrated confidence = isotonic map of raw confidence to P(understands), fitted on
calibration learners (borderline excluded), pooled over the four confirmatory cells and all steps.
Questions needed (every method, ruling 25 Oct 2026): first step on the method's own path whose calibrated confidence reaches 0.90; SPRT's native stop is reported beside it.
Undecided learners count at the maximum length and are counted (undecided rate is an outcome).
Usage: python study.py <stage> <out_dir>
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
sys.path.insert(0, str(REPO / 'scripts' / 'edm_v2_e0'))
sys.path.insert(0, str(REPO / 'src'))
pass  # (public) the original added a local site-packages path for scikit-learn; install requirements.txt
import torch  # noqa: E402
from bobcat_gate import BANK_SEED, FIT, FIT_SHA, state_hash  # noqa: E402
from compare_selection_v1 import adapt, setup  # noqa: E402
from plainproof import bank_from_fit, simulator  # noqa: E402
from plainproof.irt import p3pl  # noqa: E402
from plainproof.sprt import SPRTConfig, run_sprt, run_sprt_full  # noqa: E402

CELLS = [('uniform', 0.0), ('uniform', 0.5), ('uncertainty', 0.0), ('uncertainty', 0.5)]
TARGET = 0.90
FIXED_SEEDS = [2026102600 + k for k in range(10)]
SEEDS = dict(tune=2026102520, calibrate=2026102540, pilot=2026102560)
N = dict(tune=500, calibrate=250, pilot=500)
TUNE_GRID = [(dl, mx) for dl in (0.3, 0.5, 0.7) for mx in (20, 30)]
TUNE_RULE = ('among (delta, max) with decision error <= 0.10 among decided known learners in both d = 0 cells, '
             'pick the smallest mean questions (undecided at max) averaged over the two d = 0 cells; ties -> smaller max, '
             'then smaller delta. Revised 25 Oct 2026 after the first tuning pass (development only, before the freeze): '
             'the first rule required error <= 0.10 in all four cells, which no setting met because outsourcing at '
             'd = 0.5 raises error to 0.11-0.27 for every setting; that rise is the study outcome, so tuning uses d = 0 only')
GRID = np.linspace(-4, 4, 161)
LOGPRIOR = -GRID ** 2 / 2


def world_for(bank, model, d, n, seed):
    return simulator.World(bank, n, cut=0.0, margin=0.25, h_share=0.3, a_ai=0.95, model=model, d=d, seed=seed)


def answer_table(world, m):
    return np.array([[world.respond(i, j) for j in range(m)] for i in range(len(world.learners))])


def posterior_path(fa, fb, fc, order, answers):
    ll = LOGPRIOR.copy()
    out = []
    for j, u in zip(order, answers):
        p = p3pl(GRID, fa[j], fb[j], fc[j])
        ll += np.log(p if u else 1 - p)
        w = np.exp(ll - ll.max())
        out.append(float(w[GRID >= 0.0].sum() / w.sum()))
    return out


def sprt_paths(fa, fb, fc, table, cfg, full=False):
    """full=False: the path stops at the native boundary (tuning and the fitted calibration used this).
    full=True (ruling 25 Oct 2026): the fixed order runs to the maximum; the native stop is recorded."""
    stop, dec, raw = [], [], []
    for i in range(len(table)):
        if full:
            _, _, lrs, ns, nd = run_sprt_full(fa, fb, fc, cfg, lambda j, i=i: int(table[i, j]))
            stop.append(ns if ns is not None else cfg.n_max); dec.append(nd)
        else:
            arm = run_sprt(fa, fb, fc, cfg, lambda j, i=i: int(table[i, j]))
            stop.append(len(arm.state.items) if arm.state.decision in (0, 1) else cfg.n_max)
            dec.append(arm.state.decision); lrs = arm.state.log_ratios
        raw.append([1 / (1 + math.exp(-lr)) for lr in lrs])
    return stop, dec, raw


def bobcat_paths(t, pr, fu_q, table, n_max):
    col = {int(q): j for j, q in enumerate(fu_q)}
    B, n = len(table), pr.n_question
    avail = torch.zeros((B, n), dtype=torch.long); avail[:, torch.as_tensor(fu_q, dtype=torch.long)] = 1
    labels = torch.zeros((B, n), dtype=torch.long); mask = torch.zeros((B, n), dtype=torch.long)
    batch = {'input_labels': labels, 'input_mask': avail.clone()}
    th = []
    for _ in range(n_max):
        with torch.no_grad():
            _, action = t.st_policy.policy((labels.float() - .5) * 2 * mask, avail)
        for i, c in enumerate(action.tolist()):
            assert c in col and avail[i, c] == 1, 'BOBCAT left the bank'
            labels[i, c] = int(table[i, col[c]])
        avail[torch.arange(B), action] = 0; mask[torch.arange(B), action] = 1
        batch['input_labels'] = labels
        th.append(adapt(t, batch, mask)[:, 0].numpy().copy())
    return np.array(th).T.tolist()


def all_paths(t, pr, bank, q_ids, model, d, n, seed, cfg, full_sprt=False):
    world = world_for(bank, model, d, n, seed)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    table = answer_table(world, m)
    truth = np.array([l.truth for l in world.learners])
    stop, dec, sraw = sprt_paths(fa, fb, fc, table, cfg, full=full_sprt)
    fixed = {}
    for s in FIXED_SEEDS:
        order = np.random.default_rng(s).permutation(m)[:cfg.n_max]
        fixed[s] = [posterior_path(fa, fb, fc, order, table[i, order]) for i in range(n)]
    rng = np.random.default_rng(seed + 7)
    rand = []
    for i in range(n):
        order = rng.permutation(m)[:cfg.n_max]
        rand.append(posterior_path(fa, fb, fc, order, table[i, order]))
    bob = bobcat_paths(t, pr, q_ids[bank.rows('followup')], table, cfg.n_max)
    return dict(truth=truth, sprt_stop=stop, sprt_dec=dec, sprt=sraw, fixed=fixed, random=rand, bobcat=bob)


class Iso:
    def __init__(self, x, y):
        from sklearn.isotonic import IsotonicRegression
        self.m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(x, y)
        self.x = self.m.X_thresholds_.tolist(); self.y = self.m.y_thresholds_.tolist()

    @staticmethod
    def apply(model, v):
        return np.interp(v, model['x'], model['y'])


def needed(cal, n_max):
    for k, p in enumerate(cal, 1):
        if max(p, 1 - p) >= TARGET:
            return k, int(p >= 0.5)
    return n_max, None


def stage_tune(t, pr, bank, q_ids, out):
    rows = []
    for dl, mx in TUNE_GRID:
        cfg = SPRTConfig(cut=0.0, delta=dl, alpha=0.05, beta=0.05, n_max=mx)
        per = []
        for k, (model, d) in enumerate(CELLS):
            world = world_for(bank, model, d, N['tune'], SEEDS['tune'] + k)
            fa, fb, fc = world.followup_bank()
            table = answer_table(world, len(fa))
            stop, dec, _ = sprt_paths(fa, fb, fc, table, cfg)
            truth = np.array([l.truth for l in world.learners])
            decided = [(x, y) for x, y in zip(dec, truth) if x in (0, 1) and y >= 0]
            err = np.mean([x != y for x, y in decided]) if decided else 1.0
            per.append(dict(model=model, d=d, mean_q=float(np.mean(stop)), undecided=float(np.mean([x not in (0, 1) for x in dec])), error=float(err)))
        d0 = [c for c in per if c['d'] == 0.0]
        rows.append(dict(delta=dl, max=mx, cells=per, ok=all(c['error'] <= 0.10 for c in d0),
                         mean_q=float(np.mean([c['mean_q'] for c in d0]))))
    ok = [r for r in rows if r['ok']]
    if not ok:
        (out / 'tune-failed.json').write_text(json.dumps(dict(rule=TUNE_RULE, grid=rows), indent=1))
        for r in rows:
            print(r['delta'], r['max'], [(c['model'], c['d'], round(c['error'], 3), round(c['undecided'], 3), round(c['mean_q'], 1)) for c in r['cells']])
        raise SystemExit('no SPRT setting meets the declared error rule; stop and report')
    best = min(ok, key=lambda r: (r['mean_q'], r['max'], r['delta']))
    rep = dict(rule=TUNE_RULE, grid=rows, chosen=dict(delta=best['delta'], max=best['max']), seeds=SEEDS['tune'], n_per_cell=N['tune'])
    (out / 'tune.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(chosen=rep['chosen'], table=[(r['delta'], r['max'], round(r['mean_q'], 2), r['ok'],
                                                         [round(c['error'], 3) for c in r['cells']],
                                                         [round(c['undecided'], 3) for c in r['cells']]) for r in rows]), indent=1))


def load_cfg(out):
    c = json.loads((out / 'tune.json').read_text())['chosen']
    return SPRTConfig(cut=0.0, delta=c['delta'], alpha=0.05, beta=0.05, n_max=c['max'])


def stage_calibrate(t, pr, bank, q_ids, out):
    cfg = load_cfg(out)
    xs = {k: [] for k in ('sprt', 'fixed', 'random', 'bobcat')}
    ys = {k: [] for k in xs}
    for k, (model, d) in enumerate(CELLS):
        P = all_paths(t, pr, bank, q_ids, model, d, N['calibrate'], SEEDS['calibrate'] + k, cfg)
        for i, y in enumerate(P['truth']):
            if y < 0:
                continue
            for name in ('sprt', 'random', 'bobcat'):
                for v in P[name][i]:
                    xs[name].append(v); ys[name].append(y)
            for s in FIXED_SEEDS:
                for v in P['fixed'][s][i]:
                    xs['fixed'].append(v); ys['fixed'].append(y)
    maps = {}
    for name in xs:
        iso = Iso(np.array(xs[name]), np.array(ys[name], dtype=float))
        maps[name] = dict(x=iso.x, y=iso.y, points=len(xs[name]))
    blob = json.dumps(maps, sort_keys=True).encode()
    (out / 'calibration.json').write_bytes(blob)
    rep = dict(sha256=hashlib.sha256(blob).hexdigest(), method='isotonic, pooled over the 4 confirmatory cells and all steps',
               learners_per_cell=N['calibrate'], seeds=[SEEDS['calibrate'] + k for k in range(4)],
               same_learners_for_every_method=True, sprt=dict(delta=cfg.delta, max=cfg.n_max),
               points={k: v['points'] for k, v in maps.items()})
    (out / 'calibration.receipt.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def stage_calibrate_sprt_full(t, pr, bank, q_ids, out):
    """Refit SPRT's map only, once (ruling 25 Oct 2026): same calibration seeds and learners, SPRT's full
    20-question paths. A map is fitted on the kind of path it scores; SPRT's first map was fitted on paths that
    stopped at the boundary (top calibrated value 0.878), while the common rule scores full paths.
    The other three maps are copied unchanged from calibration.json."""
    cfg = load_cfg(out)
    old = json.loads((out / 'calibration.json').read_bytes())
    xs, ys = [], []
    for k, (model, d) in enumerate(CELLS):
        world = world_for(bank, model, d, N['calibrate'], SEEDS['calibrate'] + k)
        fa, fb, fc = world.followup_bank()
        table = answer_table(world, len(fa))
        _, _, raw = sprt_paths(fa, fb, fc, table, cfg, full=True)
        for i, l in enumerate(world.learners):
            if l.truth >= 0:
                xs += raw[i]; ys += [l.truth] * len(raw[i])
    iso = Iso(np.array(xs), np.array(ys, dtype=float))
    new = dict(old); new['sprt'] = dict(x=iso.x, y=iso.y, points=len(xs))
    blob = json.dumps(new, sort_keys=True).encode()
    (out / 'calibration-v2.json').write_bytes(blob)
    rep = dict(sha256=hashlib.sha256(blob).hexdigest(), replaces='calibration.json for SPRT only',
               previous_sha256=hashlib.sha256((out / 'calibration.json').read_bytes()).hexdigest(),
               reason='a map is fitted on the kind of path it scores; SPRT now scores full 20-question paths',
               seeds=[SEEDS['calibrate'] + k for k in range(4)], learners_per_cell=N['calibrate'],
               unchanged=['fixed', 'random', 'bobcat'], sprt_points=len(xs),
               sprt_y_range=[min(iso.y), max(iso.y)], sprt_x_range=[min(iso.x), max(iso.x)])
    (out / 'calibration-v2.receipt.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def stage_pilot(t, pr, bank, q_ids, out):
    cfg = load_cfg(out)
    cal_file = out / 'calibration-v2.json'
    maps = json.loads(cal_file.read_bytes())
    cells = []
    for k, (model, d) in enumerate(CELLS):
        P = all_paths(t, pr, bank, q_ids, model, d, N['pilot'], SEEDS['pilot'] + k, cfg, full_sprt=True)
        n = len(P['truth'])
        sx = np.array([v for path in P['sprt'] for v in path])
        lo, hi = min(maps['sprt']['x']), max(maps['sprt']['x'])
        sprt_out_of_range = dict(share=float(np.mean((sx < lo) | (sx > hi))), fitted_range=[lo, hi], points=int(len(sx)))
        truth = P['truth']
        known = truth >= 0

        def score(name, paths):
            # One rule for every method (ruling 25 Oct 2026): first step whose calibrated confidence reaches 0.90 on
            # the method's own path; never reached (including SPRT stopping first) = undecided, counted at the maximum.
            res = [needed(Iso.apply(maps[name], np.array(paths[i])), cfg.n_max) for i in range(n)]
            qq = np.array([r[0] for r in res]); dec = [r[1] for r in res]
            pairs = [(x, y) for x, y, k in zip(dec, truth, known) if x is not None and k]
            err = float(np.mean([x != y for x, y in pairs])) if pairs else float('nan')
            return qq, float(np.mean([x is None for x in dec])), err

        q, und, err = {}, {}, {}
        for name in ('sprt', 'random', 'bobcat'):
            q[name], und[name], err[name] = score(name, P[name])
        native = [(x, y) for x, y, k in zip(P['sprt_dec'], truth, known) if x in (0, 1) and k]   # boundary fired within 20
        sprt_native = dict(mean_q=float(np.mean(P['sprt_stop'])), error=float(np.mean([x != y for x, y in native])) if native else float('nan'),
                           undecided=float(np.mean([x not in (0, 1) for x in P['sprt_dec']])))
        per_form = {s: score('fixed', P['fixed'][s]) for s in FIXED_SEEDS}
        q['fixed'] = np.mean([v[0] for v in per_form.values()], axis=0)
        und['fixed'] = float(np.mean([v[1] for v in per_form.values()]))
        err['fixed'] = float(np.nanmean([v[2] for v in per_form.values()]))
        paired = {}
        for other in ('fixed', 'random', 'bobcat'):
            dl = q['sprt'] - q[other]
            paired[f'sprt-{other}'] = dict(mean=float(dl.mean()), sd=float(dl.std(ddof=1)))
        form_sd = [float((q['sprt'] - v[0]).std(ddof=1)) for v in per_form.values()]
        cells.append(dict(model=model, d=d, n=n, mean_q={k: float(v.mean()) for k, v in q.items()},
                          undecided={k: float(v) for k, v in und.items()}, error={k: float(v) for k, v in err.items()},
                          paired=paired, sprt_native_stop=sprt_native, sprt_calibration_out_of_range=sprt_out_of_range,
                          fixed_forms=dict(mean_q=[float(v[0].mean()) for v in per_form.values()],
                                           error=[float(v[2]) for v in per_form.values()], paired_sd_per_form=form_sd)))
    worst = max([c['paired'][k]['sd'] for c in cells for k in ('sprt-fixed', 'sprt-bobcat')] +
                [max(c['fixed_forms']['paired_sd_per_form']) for c in cells])
    from scipy.stats import norm
    z = norm.ppf(1 - 0.05 / 16) + norm.ppf(0.8)
    n_needed = math.ceil((z * worst / 0.4) ** 2)
    rep = dict(cells=cells, worst_paired_sd=worst, held_out_n_for_0_4=n_needed, sprt=dict(delta=cfg.delta, max=cfg.n_max),
               calibration_sha256=hashlib.sha256(cal_file.read_bytes()).hexdigest(),
               fixed_seeds=FIXED_SEEDS, seeds=[SEEDS['pilot'] + k for k in range(4)], development_only=True, reportable=False)
    (out / 'pilot.json').write_text(json.dumps(rep, indent=1))
    for c in cells:
        print(c['model'], c['d'], 'q', {k: round(v, 2) for k, v in c['mean_q'].items()}, 'err', {k: round(v, 3) for k, v in c['error'].items()},
              'und', {k: round(v, 3) for k, v in c['undecided'].items()}, 'native', {k: round(v, 3) for k, v in c['sprt_native_stop'].items()},
              'sprt_out_of_range', round(c['sprt_calibration_out_of_range']['share'], 3),
              {k: (round(v['mean'], 2), round(v['sd'], 2)) for k, v in c['paired'].items()},
              'fixed forms', [round(x, 1) for x in c['fixed_forms']['mean_q']])
    print('worst paired sd', round(worst, 2), 'held-out n per cell', n_needed)


def main(stage, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(FIT.read_bytes()).hexdigest() != FIT_SHA:
        raise SystemExit('item fit changed; stop')
    t0 = time.time()
    t, pr = setup()
    before = state_hash(t)
    bank, q_ids = bank_from_fit.build(bank_from_fit.load_fit(FIT), c=0.0, seed=BANK_SEED)
    dict(tune=stage_tune, calibrate=stage_calibrate, calibrate_sprt_full=stage_calibrate_sprt_full,
         pilot=stage_pilot)[stage](t, pr, bank, q_ids, out)
    assert state_hash(t) == before, 'BOBCAT weights changed'
    print('seconds', round(time.time() - t0, 1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
