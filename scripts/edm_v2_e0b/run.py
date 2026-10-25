"""E0b: do planted hard probes separate outsourcing from honest learners? (ruling 25 Oct 2026)

Same 20 cells as E0 (2 outsourcing models x 2 AI accuracies x 5 d). Scored paths come from the frozen E0 code
(world, answer table, SPRT order, 10 fixed forms, random orders, BOBCAT in chunks), cut to 17 scored items;
three unscored probes are planted at combined positions 4, 9 and 14 (plainproof.planted).
Per learner, method and checkpoint k in (5, 10, 15, 20) combined positions:
  raw hits       probes answered correctly among those asked by k (0 to 3);
  probe excess   their hit rate minus the mean 3PL probability at the EAP theta from the scored items asked by k.
AUC: d = 0 cell against each d > 0 cell, same model and AI accuracy, all learners and split by true theta at the
cut (theta >= 0 / < 0). Pre-registered line: AUC 0.75 at d = 0.5, AI accuracy 0.95, both models, k = 15.
Usage: python run.py pilot|run <out_dir>;  python run.py summarize <out_dir>
"""
import gzip
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'edm_v2_e0'))
import heldout  # noqa: E402  (imports the frozen study module)
import study  # noqa: E402
from plainproof import bank_from_fit, planted, signature, simulator  # noqa: E402
from plainproof.irt import p3pl  # noqa: E402
from plainproof.sprt import run_sprt_full  # noqa: E402

CHECKPOINTS = (5, 10, 15, 20)
SCORED = planted.scored_before(20)          # 17 scored items within 20 combined positions
CELLS = [(m, a, d) for m in ('uniform', 'uncertainty', 'giveup') for a in (0.95, 0.75) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
SEEDS = dict(pilot=2026114000, run=2026120000)   # pilot v1 (2 models) used 2026113000
N = dict(pilot=500, run=2000)
RSS_CAP_GIB, SECONDS_CAP = 2.0, 1800
LINE = dict(auc=0.75, d=0.5, a_ai=0.95, k=15, models=['uniform'],
            uncertainty='null expected by construction (outsources where own chance is near 0.5; probes sit near 0)',
            giveup='exploratory')


def run_cell(t, pr, bank, q_ids, cfg, model, a_ai, d, seed, n):
    world = planted.make_world(bank, n, model, d, a_ai, seed)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    table = study.answer_table(world, m)
    so, _, _, _, _ = run_sprt_full(fa, fb, fc, cfg, lambda j: int(table[0, j]))
    orders = {'sprt': [list(so[:SCORED])] * n}
    for s in study.FIXED_SEEDS:
        orders[f'fixed{s}'] = [list(np.random.default_rng(s).permutation(m)[:SCORED])] * n
    rng = np.random.default_rng(seed + 7)
    orders['random'] = [list(rng.permutation(m)[:cfg.n_max])[:SCORED] for _ in range(n)]
    _, bo = heldout.bobcat_chunked(t, pr, q_ids[bank.rows('followup')], table, SCORED)
    orders['bobcat'] = bo
    a_probe = float(np.median(bank.a))
    prng = np.random.default_rng(seed + 13)
    probes = [planted.plant_any(l.theta, model, d, a_ai, a_probe, fa, fb, fc, prng) for l in world.learners]
    pb = np.array(planted.PROBE_B)
    stats = {}
    for name, ords in orders.items():
        per_k = {}
        for k in CHECKPOINTS:
            ns, npb = planted.scored_before(k), planted.probes_before(k)
            hits, excess = [], []
            for i in range(n):
                it = ords[i][:ns]
                th = signature.eap(fa[it], fb[it], fc[it], table[i, it])
                corr = probes[i][0][:npb]
                hits.append(int(corr.sum()))
                excess.append(float(corr.mean() - p3pl(th, a_probe, pb[:npb], 0.0).mean()))
            per_k[str(k)] = dict(hits=hits, excess=excess)
        stats[name] = per_k
    return dict(model=model, a_ai=a_ai, d=d, seed=seed, n=n, a_probe=a_probe,
                theta=[l.theta for l in world.learners], truth=[l.truth for l in world.learners],
                probe_outsourced=[p[1].astype(int).tolist() for p in probes], stats=stats)


def main_run(stage, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = study.load_cfg(study_dir())
    t, pr = study.setup()
    before = study.state_hash(t)
    bank, q_ids = bank_from_fit.build(bank_from_fit.load_fit(study.FIT), c=0.0, seed=study.BANK_SEED)
    receipt = dict(stage=stage, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   planted_sha256=hashlib.sha256((REPO / 'src/plainproof/planted.py').read_bytes()).hexdigest(),
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cells=[])
    for idx, (model, a_ai, d) in enumerate(CELLS):
        path = out / f'cell-{idx:02d}.json.gz'
        if path.exists():
            raise SystemExit(f'{path} exists; never overwrite')
        t0 = time.time()
        r = run_cell(t, pr, bank, q_ids, cfg, model, a_ai, d, SEEDS[stage] + 100 * idx, N[stage])
        path.write_bytes(gzip.compress(json.dumps(r).encode()))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
        el = time.time() - t0
        receipt['cells'].append(dict(index=idx, seconds=round(el, 1), peak_rss_gib=round(peak, 3),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(receipt['cells'][-1]), flush=True)
        if peak > RSS_CAP_GIB or el > SECONDS_CAP:
            receipt['stopped'] = receipt['cells'][-1]
            (out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
            raise SystemExit('cap breached; stopped')
    assert study.state_hash(t) == before
    receipt['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1))


def study_dir():
    return heldout.STUDY


def main_summarize(out):
    out = Path(out)
    cells = {int(p.name[5:7]): json.loads(gzip.decompress(p.read_bytes())) for p in sorted(out.glob('cell-*.json.gz'))}
    rows = []
    for i, c in sorted(cells.items()):
        if c['d'] == 0.0:
            continue
        ref = next(x for x in cells.values() if x['model'] == c['model'] and x['a_ai'] == c['a_ai'] and x['d'] == 0.0)
        hth, oth = np.array(ref['theta']), np.array(c['theta'])
        share = float(np.mean(c['probe_outsourced']))
        for k in map(str, CHECKPOINTS):
            row = dict(cell=i, model=c['model'], a_ai=c['a_ai'], d=c['d'], k=int(k), probe_outsourced_share=share)
            for stat in ('hits', 'excess'):
                res = {}
                for name in ('sprt', 'random', 'bobcat', 'fixed'):
                    def auc_for(sel_h, sel_o):
                        if name == 'fixed':
                            v = [signature.auc(np.array(ref['stats'][f'fixed{s}'][k][stat])[sel_h],
                                               np.array(c['stats'][f'fixed{s}'][k][stat])[sel_o], +1) for s in study.FIXED_SEEDS]
                            return float(np.nanmean(v))
                        return signature.auc(np.array(ref['stats'][name][k][stat])[sel_h],
                                             np.array(c['stats'][name][k][stat])[sel_o], +1)
                    res[name] = dict(all=auc_for(np.ones(len(hth), bool), np.ones(len(oth), bool)),
                                     theta_above=auc_for(hth >= 0, oth >= 0), theta_below=auc_for(hth < 0, oth < 0))
                row[stat] = res
            rows.append(row)
    line = [r for r in rows if r['d'] == LINE['d'] and r['a_ai'] == LINE['a_ai'] and r['k'] == LINE['k']]
    verdict = {r['model']: {stat: {m: r[stat][m]['all'] >= LINE['auc'] for m in r[stat]} for stat in ('hits', 'excess')}
               for r in line if r['model'] in LINE['models']}
    summ = dict(rows=rows, line=LINE, at_line=line, crossing=verdict)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    for r in rows:
        if r['k'] == 15:
            print(r['model'][:4], r['a_ai'], r['d'], 'probe out share', round(r['probe_outsourced_share'], 3),
                  'hits', {m: (round(v['all'], 2), round(v['theta_above'], 2), round(v['theta_below'], 2)) for m, v in r['hits'].items() if m in ('sprt', 'bobcat')},
                  'excess', {m: (round(v['all'], 2), round(v['theta_above'], 2), round(v['theta_below'], 2)) for m, v in r['excess'].items()})


if __name__ == '__main__':
    if sys.argv[1] in ('pilot', 'run'):
        main_run(sys.argv[1], sys.argv[2])
    else:
        main_summarize(sys.argv[2])
