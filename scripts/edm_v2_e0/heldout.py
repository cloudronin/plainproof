"""EDM v2 held-out run under the sealed freeze (reports/edm-v2-e0/freeze.json, sha256 a0f8ce40...).

Written after the seal. It adds only: BOBCAT in chunks (memory; rows are independent in BOBCAT's policy and
adaptation, so chunking does not change any row), the seven signatures, the paired bootstrap with Holm, and
the summary. Paths, scoring and calibration come from the frozen study.py functions and the frozen maps.
Signature statistic 3 for a comparison at outsourcing level d is computed for both groups with that d (computing
the honest group at its own d = 0 would make every honest value 0).
Usage: python heldout.py run <out_dir> [--smoke]    (--smoke: development seeds, 60 learners, 2 cells)
       python heldout.py summarize <out_dir>
"""
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'edm_v2_e0'))
import study  # noqa: E402  (frozen; imports BOBCAT setup and the edm_v2 package)
from study import (FIXED_SEEDS, Iso, answer_table, load_cfg, needed, posterior_path, setup, sprt_paths,  # noqa: E402
                   state_hash, world_for)
import torch  # noqa: E402
from plainproof import bank_from_fit, signature, simulator  # noqa: E402

FREEZE = REPO / 'reports' / 'edm-v2-e0' / 'freeze.json'
FREEZE_SHA = 'a0f8ce404f2b5f7e671e2503a33b4554040e80714f6053553507e0972acdbcdc'
STUDY = Path(__import__('os').environ.get('REPRO_DATA', 'data'), 'edm-v2-e0/study')
CHUNK = 500
RSS_CAP_GIB, SECONDS_CAP = 2.0, 1800   # freeze.json resources
DS = (0.25, 0.5, 0.75, 1.0)


def bobcat_chunked(t, pr, fu_q, table, n_max):
    """Same steps as study.bobcat_paths, per chunk; also returns each learner's asked follow-up indices."""
    col = {int(q): j for j, q in enumerate(fu_q)}
    thetas, orders = [], []
    for s in range(0, len(table), CHUNK):
        tb = table[s:s + CHUNK]
        B, n = len(tb), pr.n_question
        avail = torch.zeros((B, n), dtype=torch.long); avail[:, torch.as_tensor(fu_q, dtype=torch.long)] = 1
        labels = torch.zeros((B, n), dtype=torch.long); mask = torch.zeros((B, n), dtype=torch.long)
        batch = {'input_labels': labels, 'input_mask': avail.clone()}
        th, acts = [], []
        for _ in range(n_max):
            with torch.no_grad():
                _, action = t.st_policy.policy((labels.float() - .5) * 2 * mask, avail)
            a = action.tolist()
            for i, c in enumerate(a):
                assert c in col and avail[i, c] == 1, 'BOBCAT left the bank'
                labels[i, c] = int(tb[i, col[c]])
            avail[torch.arange(B), action] = 0; mask[torch.arange(B), action] = 1
            batch['input_labels'] = labels
            th.append(study.adapt(t, batch, mask)[:, 0].numpy().copy())
            acts.append([col[c] for c in a])
        thetas += np.array(th).T.tolist()
        orders += np.array(acts).T.tolist()
    return thetas, orders


def run_cell(t, pr, bank, q_ids, probe_rows, cfg, maps, model, a_ai, d, seed, n):
    world = simulator.World(bank, n, cut=0.0, margin=0.25, h_share=0.3, a_ai=a_ai, model=model, d=d, seed=seed)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    fu_rows = bank.rows('followup')
    table = answer_table(world, m)
    truth = np.array([l.truth for l in world.learners])
    stop, ndec, sraw = sprt_paths(fa, fb, fc, table, cfg, full=True)
    # SPRT's order is the same for everyone; recover it from the frozen arm on learner 0.
    from plainproof.sprt import run_sprt_full
    sprt_order, _, _, _, _ = run_sprt_full(fa, fb, fc, cfg, lambda j: int(table[0, j]))
    orders = {'sprt': [sprt_order] * n}
    paths = {'sprt': sraw}
    for s in FIXED_SEEDS:
        o = list(np.random.default_rng(s).permutation(m)[:cfg.n_max])
        orders[f'fixed{s}'] = [o] * n
        paths[f'fixed{s}'] = [posterior_path(fa, fb, fc, o, table[i, o]) for i in range(n)]
    rng = np.random.default_rng(seed + 7)                       # same stream as study.all_paths
    ro = [list(rng.permutation(m)[:cfg.n_max]) for _ in range(n)]
    orders['random'] = ro
    paths['random'] = [posterior_path(fa, fb, fc, ro[i], table[i, ro[i]]) for i in range(n)]
    paths['bobcat'], orders['bobcat'] = bobcat_chunked(t, pr, q_ids[fu_rows], table, cfg.n_max)
    res = {}
    for name in paths:
        mp = maps['fixed' if name.startswith('fixed') else name]
        cal = [Iso.apply(mp, np.array(p)) for p in paths[name]]
        r = [needed(c, cfg.n_max) for c in cal]
        res[name] = dict(q=np.array([x[0] for x in r]), dec=np.array([-1 if x[1] is None else x[1] for x in r]),
                         cal=cal)
    # Signatures: 1-6 on the SPRT order, 7 per method on its own order (fixed: each of the 10 forms).
    sprt_global = [int(fu_rows[j]) for j in sprt_order]
    sig = {k: {} for k in signature.CHECKPOINTS}
    for k in signature.CHECKPOINTS:
        rows = []
        for i, l in enumerate(world.learners):
            st = signature.learner_stats(world.log[i], bank, sprt_global, k, 0.0, a_ai, res['sprt']['cal'][i], l.truth)
            asked = [r for r in world.log[i] if r['bank'] == 'followup']
            byid = {r['item']: r for r in asked}
            sel = [byid[j] for j in sprt_global[:k]]
            it = [r['item'] for r in sel]
            th = signature.eap(bank.a[it], bank.b[it], bank.c[it], [r['correct'] for r in sel])
            st['llr_at'] = {str(dd): signature.llr_outsourcing(bank.a[it], bank.b[it], bank.c[it],
                                                               [r['correct'] for r in sel], th, dd, a_ai) for dd in DS}
            st['probe_excess'] = {name: signature.probe_excess(world.log[i], bank,
                                                               [int(fu_rows[j]) for j in orders[name][i]], k, probe_rows)
                                  for name in orders}
            rows.append(st)
        sig[k] = rows
    return dict(model=model, a_ai=a_ai, d=d, seed=seed, n=n, truth=truth.tolist(), native_stop=stop,
                native_dec=[x if x in (0, 1) else -1 for x in ndec],
                q={k: v['q'].tolist() for k, v in res.items()}, dec={k: v['dec'].tolist() for k, v in res.items()},
                signature={str(k): v for k, v in sig.items()}, sprt_order=[int(x) for x in sprt_order])


def main_run(out, smoke):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(FREEZE.read_bytes()).hexdigest() != FREEZE_SHA:
        raise SystemExit('freeze file changed; stop')
    F = json.loads(FREEZE.read_text())
    for p, h in F['code'].items():
        if hashlib.sha256(__import__('repro_paths').original_bytes(REPO, p)).hexdigest() != h:
            raise SystemExit(f'frozen code changed: {p}')
    if hashlib.sha256((STUDY / 'calibration-v2.json').read_bytes()).hexdigest() != F['calibration']['sha256']:
        raise SystemExit('calibration changed; stop')
    maps = json.loads((STUDY / 'calibration-v2.json').read_bytes())
    cfg = load_cfg(STUDY)
    assert (cfg.delta, cfg.n_max) == (0.7, 20)
    t, pr = setup()
    before = state_hash(t)
    bank, q_ids = bank_from_fit.build(bank_from_fit.load_fit(study.FIT), c=0.0, seed=study.BANK_SEED)
    probe_rows, probe_ids = bank_from_fit.probes(bank, q_ids)
    assert probe_ids == F['secondary_signature']['probes']['q_ids']
    cells = F['cells'] if not smoke else [dict(index=0, model='uniform', a_ai=0.95, d=0.0, seed=2026102590),
                                         dict(index=2, model='uniform', a_ai=0.95, d=0.5, seed=2026102591)]
    n = F['n_per_cell'] if not smoke else 60
    receipt = dict(freeze_sha256=FREEZE_SHA, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   smoke=smoke, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cells=[])
    for c in cells:
        path = out / f"cell-{c['index']:02d}.json.gz"
        if path.exists():
            raise SystemExit(f'{path} exists; this runner never overwrites a cell')
        t0 = time.time()
        r = run_cell(t, pr, bank, q_ids, probe_rows, cfg, maps, c['model'], c['a_ai'], c['d'], c['seed'], n)
        path.write_bytes(gzip.compress(json.dumps(r).encode(), compresslevel=6))
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # macOS: bytes
        elapsed = time.time() - t0
        receipt['cells'].append(dict(index=c['index'], seconds=round(elapsed, 1),
                                     peak_rss_gib=round(peak, 3),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(receipt['cells'][-1]), flush=True)
        # Freeze caps (resources: 2 GiB RSS and 1,800 s per cell). Added 25 Oct 2026 after the held-out run
        # exceeded the RSS cap (2.16-2.23 GiB) because nothing enforced it. A breach now stops the run.
        if peak > RSS_CAP_GIB or elapsed > SECONDS_CAP:
            receipt['stopped'] = dict(cell=c['index'], peak_rss_gib=round(peak, 3), seconds=round(elapsed, 1),
                                      caps=dict(rss_gib=RSS_CAP_GIB, seconds=SECONDS_CAP))
            (out / 'receipt.json').write_text(json.dumps(receipt, indent=1))
            raise SystemExit(f"cap breached at cell {c['index']}: {peak:.3f} GiB, {elapsed:.0f} s; stopped")
    assert state_hash(t) == before, 'BOBCAT weights changed'
    receipt['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1))


def boot_ci(x, levels, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.array([x[rng.integers(0, n, n)].mean() for _ in range(reps)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return means, min(1.0, p)


def main_summarize(out):
    out = Path(out)
    F = json.loads(FREEZE.read_text())
    cells = {int(p.name.split('-')[1].split('.')[0]): json.loads(gzip.decompress(p.read_bytes())) for p in sorted(out.glob('cell-*.json.gz'))}
    methods = ['sprt', 'fixed', 'random', 'bobcat']
    table = []
    for i, c in sorted(cells.items()):
        truth = np.array(c['truth']); known = truth >= 0
        row = dict(index=i, model=c['model'], a_ai=c['a_ai'], d=c['d'], n=c['n'],
                   exploratory=c['a_ai'] != 0.95)
        for mth in methods:
            if mth == 'fixed':
                qs = np.mean([c['q'][f'fixed{s}'] for s in FIXED_SEEDS], axis=0)
                decs = [np.array(c['dec'][f'fixed{s}']) for s in FIXED_SEEDS]
            else:
                qs = np.array(c['q'][mth]); decs = [np.array(c['dec'][mth])]
            errs, unds, fm, fg = [], [], [], []
            for dc in decs:
                ok = (dc >= 0) & known
                errs.append(float(np.mean(dc[ok] != truth[ok])) if ok.any() else float('nan'))
                unds.append(float(np.mean(dc < 0)))
                fm.append(float(np.mean((dc == 1) & (truth == 0))))
                fg.append(float(np.mean((dc == 0) & (truth == 1))))
            row[mth] = dict(questions=float(qs.mean()), error=float(np.nanmean(errs)), undecided=float(np.mean(unds)),
                            false_mastery=float(np.mean(fm)), false_gap=float(np.mean(fg)))
        nd = np.array(c['native_dec']); ns = np.array(c['native_stop'])
        ok = (nd >= 0) & known
        row['sprt_native'] = dict(questions=float(ns.mean()), error=float(np.mean(nd[ok] != truth[ok])) if ok.any() else float('nan'),
                                  undecided=float(np.mean(nd < 0)))
        table.append(row)
    # Confirmatory: SPRT minus fixed / BOBCAT, paired per learner, Holm across 8.
    conf = []
    for i, c in sorted(cells.items()):
        if c['a_ai'] != 0.95 or c['d'] not in (0.0, 0.5):
            continue
        fixed_q = np.mean([c['q'][f'fixed{s}'] for s in FIXED_SEEDS], axis=0)
        for other, oq in (('fixed', fixed_q), ('bobcat', np.array(c['q']['bobcat']))):
            dlt = np.array(c['q']['sprt']) - oq
            means, p = boot_ci(dlt, None, seed=1000 + i)
            conf.append(dict(cell=i, model=c['model'], d=c['d'], contrast=f'sprt-{other}', mean=float(dlt.mean()),
                             p=p, boot=means))
    conf.sort(key=lambda r: r['p'])
    m = len(conf); prev = 0.0
    for rank, r in enumerate(conf):
        alpha = 0.05 / (m - rank)
        r['holm_p'] = max(prev, min(1.0, (m - rank) * r['p'])); prev = r['holm_p']
        lo, hi = np.quantile(r.pop('boot'), [alpha / 2, 1 - alpha / 2])
        r['holm_interval'] = [float(lo), float(hi)]; r['holm_level'] = 1 - alpha
    # Signatures: AUC of d = 0 against each d > 0, same model and AI accuracy.
    aucs = []
    for i, c in sorted(cells.items()):
        if c['d'] == 0.0:
            continue
        ref = next(x for x in cells.values() if x['model'] == c['model'] and x['a_ai'] == c['a_ai'] and x['d'] == 0.0)
        for k in map(str, signature.CHECKPOINTS):
            h, o = ref['signature'][k], c['signature'][k]
            row = dict(cell=i, model=c['model'], a_ai=c['a_ai'], d=c['d'], k=int(k))
            for stat in ('difficulty_correctness', 'lz', 'hw_minus_fu', 'fu_minus_transfer', 'sprt_confident_wrong'):
                row[stat] = signature.auc([x[stat] for x in h], [x[stat] for x in o], signature.DIRECTION[stat])
            row['llr_outsourcing'] = signature.auc([x['llr_at'][str(c['d'])] for x in h], [x['llr_at'][str(c['d'])] for x in o], +1)
            pe = {}
            for name in ('sprt', 'random', 'bobcat'):
                hv = [x['probe_excess'][name] for x in h]; ov = [x['probe_excess'][name] for x in o]
                pe[name] = dict(auc=signature.auc(hv, ov, +1), nan_honest=int(np.isnan(hv).sum()), nan_out=int(np.isnan(ov).sum()))
            fa = [signature.auc([x['probe_excess'][f'fixed{s}'] for x in h], [x['probe_excess'][f'fixed{s}'] for x in o], +1) for s in FIXED_SEEDS]
            pe['fixed'] = dict(auc=float(np.nanmean(fa)), auc_range=[float(np.nanmin(fa)), float(np.nanmax(fa))])
            row['probe_excess'] = pe
            aucs.append(row)
    summ = dict(freeze_sha256=FREEZE_SHA, table=table, confirmatory=conf, signature_auc=aucs,
                separation=signature.SEPARATION, note='a_ai 0.75 cells exploratory (calibration learners at 0.95)')
    (out / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(dict(cells=len(cells), confirmatory=[(r['model'], r['d'], r['contrast'], round(r['mean'], 3),
                      [round(x, 3) for x in r['holm_interval']], round(r['holm_p'], 4)) for r in conf]), indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        main_run(sys.argv[2], '--smoke' in sys.argv)
    else:
        main_summarize(sys.argv[2])
