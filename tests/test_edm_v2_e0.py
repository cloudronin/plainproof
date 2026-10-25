"""EDM v2 E0: SPRT arm against catIrt, exact stopping time, outsourcing simulator, item fit. Offline only."""
import csv
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from plainproof import fit, simulator, stopping
from plainproof.irt import fisher_info, p3pl
from plainproof.sprt import SPRTArm, SPRTConfig, run_sprt

FIX = Path(__file__).parent / 'fixtures' / 'edm_v2_catirt_sprt'


def _csv(name):
    with open(FIX / name, newline='') as f:
        return list(csv.DictReader(f))


# 1. SPRT arm replays catIrt 0.5.1's saved steps.

def test_sprt_replays_catirt_steps():
    cfg_row = _csv('config.csv')[0]
    assert cfg_row['catIrt'] == '0.5.1'
    cfg = SPRTConfig(cut=float(cfg_row['cut']), delta=float(cfg_row['delta']), alpha=float(cfg_row['alpha']),
                     beta=float(cfg_row['beta']), n_max=int(cfg_row['n_max']))
    bank = _csv('bank.csv')
    a, b, c = (np.array([float(r[k]) for r in bank]) for k in ('a', 'b', 'c'))
    resp = [[int(v) for v in r.values()] for r in _csv('resp.csv')]
    steps = _csv('steps.csv')
    terminal = {int(r['sim']): r for r in _csv('terminal.csv')}
    by_sim = {}
    for r in steps:
        by_sim.setdefault(int(r['sim']), []).append(r)
    checked = 0
    for sim, rows in by_sim.items():
        arm = run_sprt(a, b, c, cfg, lambda j: resp[sim][j])
        assert arm.state.items == [int(r['item']) for r in rows], f'sim {sim}: item order'
        for mine, theirs in zip(arm.state.log_ratios, rows):
            assert abs(mine - float(theirs['log_ratio'])) < 1e-9
            checked += 1
        t = terminal[sim]
        assert len(arm.state.items) == int(t['n'])
        if t['term'] == 'class':
            assert arm.state.decision == int(t['categ'])
        else:
            assert t['term'] == 'fixed' and arm.state.decision == 'truncated'
    assert checked == len(steps) and len(by_sim) == 25


def test_sprt_selects_max_information_at_cut_with_stable_ties():
    a = np.array([1.0, 2.0, 2.0, 0.5]); b = np.zeros(4); c = np.zeros(4)
    arm = SPRTArm(a, b, c, SPRTConfig(cut=0.0, n_max=4))
    assert arm.next_item() == 1                       # tie between rows 1 and 2 goes to the earlier row
    arm.answer(1, 1)
    assert arm.next_item() == 2


def test_truncation_is_never_a_classification():
    a = np.ones(5); b = np.zeros(5); c = np.zeros(5)
    arm = run_sprt(a, b, c, SPRTConfig(cut=0.0, delta=0.01, n_max=3), lambda j: j % 2)
    assert arm.state.decision == 'truncated' and len(arm.state.items) == 3


# 2. Exact expected stopping time, identical items.

def test_exact_matches_enumeration_small_horizon():
    w1, w0 = math.log(0.6 / 0.4), math.log(0.4 / 0.6)
    up, lo, n_max, p = 1.5, -1.5, 8, 0.63
    want = dict(expected_n=0.0, p_master=0.0, p_non_master=0.0, p_truncated=0.0)
    for seq in itertools.product((0, 1), repeat=n_max):
        prob = np.prod([p if r else 1 - p for r in seq])
        lr, stop = 0.0, None
        for n, r in enumerate(seq, 1):
            lr += w1 if r else w0
            if lr >= up or lr <= lo:
                stop = (n, 'p_master' if lr >= up else 'p_non_master')
                break
        if stop is None:
            want['p_truncated'] += prob; want['expected_n'] += n_max * prob
        else:
            want[stop[1]] += prob; want['expected_n'] += stop[0] * prob
    got = stopping.exact(p, w1, w0, up, lo, n_max)
    for k in want:
        assert got[k] == pytest.approx(want[k], abs=1e-12)


@pytest.mark.parametrize('d', [0.0, 0.5])
@pytest.mark.parametrize('a_ai', [0.95, 0.75])
def test_simulator_matches_exact_stopping_time(d, a_ai):
    item = dict(a=1.3, b=0.2, c=0.15)
    cfg = SPRTConfig(cut=0.0, delta=0.4, alpha=0.05, beta=0.05, n_max=30)
    theta0 = 0.3
    n = 30
    rng = np.random.default_rng(1)
    bank = simulator.Bank(np.full(n, item['a']), np.full(n, item['b']), np.full(n, item['c']),
                          np.zeros(n, dtype=int), np.array(['followup'] * n, dtype=object))
    world = simulator.World(bank, n_learners=3000, cut=0.0, margin=0.25, h_share=0.0, a_ai=a_ai,
                            model='uniform', d=d, seed=11, theta_mean=theta0, theta_sd=0.0)
    fa, fb, fc = world.followup_bank()
    lengths, master = [], []
    for l in world.learners:
        arm = run_sprt(fa, fb, fc, cfg, lambda j, i=l.id: world.respond(i, j))
        lengths.append(len(arm.state.items)); master.append(arm.state.decision == 1)
    w1, w0 = stopping.increments(item['a'], item['b'], item['c'], cfg.cut, cfg.delta)
    p_true = stopping.p_correct(theta0, item['a'], item['b'], item['c'], d, a_ai)
    ex = stopping.exact(p_true, w1, w0, cfg.upper, cfg.lower, cfg.n_max)
    se_n = np.std(lengths) / math.sqrt(len(lengths))
    se_m = math.sqrt(ex['p_master'] * (1 - ex['p_master']) / len(master)) + 1e-9
    assert abs(np.mean(lengths) - ex['expected_n']) < 4 * se_n
    assert abs(np.mean(master) - ex['p_master']) < 4 * se_m
    wd = stopping.wald(p_true, w1, w0, cfg.upper, cfg.lower)
    assert math.isfinite(wd['expected_n']) and 0 <= wd['p_master'] <= 1


def test_wald_close_to_exact_when_steps_are_small_and_untruncated():
    w1, w0 = math.log(0.52 / 0.48), math.log(0.48 / 0.52)
    up, lo = math.log(19), math.log(1 / 19)
    ex = stopping.exact(0.55, w1, w0, up, lo, n_max=3000)
    wd = stopping.wald(0.55, w1, w0, up, lo)
    assert ex['p_truncated'] < 1e-6
    assert wd['expected_n'] == pytest.approx(ex['expected_n'], rel=0.05)
    assert wd['p_master'] == pytest.approx(ex['p_master'], abs=0.02)


# 3. Simulator: banks, outsourcing models, no leak, d = 0 reproduces the item model.

def _bank(seed=0, families=8, per=40, held=(6, 7)):
    rng = np.random.default_rng(seed)
    n = families * per
    fam = np.repeat(np.arange(families), per)
    return simulator.split_bank(rng.lognormal(0, 0.3, n), rng.normal(0, 1, n), rng.uniform(0, 0.2, n),
                                fam, held, rng)


def test_banks_are_disjoint_and_held_out_families_are_transfer_only():
    bank = _bank()
    hw, fu, tr = (set(bank.rows(r)) for r in ('homework', 'followup', 'transfer'))
    assert not (hw & fu) and not (hw & tr) and not (fu & tr)
    assert hw | fu | tr == set(range(len(bank.a)))
    for f in (6, 7):
        assert set(bank.role[bank.family == f]) == {'transfer'}
    for f in range(6):
        assert {'homework', 'followup', 'transfer'} <= set(bank.role[bank.family == f])
    world = simulator.World(bank, 5, 0.0, 0.25, 0.3, 0.95, 'uniform', 0.5, seed=1)
    fa, _, _ = world.followup_bank()
    assert len(fa) == len(fu) and np.array_equal(fa, bank.a[sorted(fu)])


@pytest.mark.parametrize('d', [0.0, 0.25, 0.5, 0.75, 1.0])
def test_uncertainty_model_hits_the_dial_and_peaks_near_half(d):
    p = np.linspace(0.02, 0.98, 60)
    q = simulator.outsourcing_probs('uncertainty', d, p)
    assert q.mean() == pytest.approx(d, abs=1e-9)
    assert np.all((0 <= q) & (q <= 1))
    if 0 < d < 1:
        assert q[np.argmin(abs(p - 0.5))] == q.max()
        assert q[0] < q[np.argmin(abs(p - 0.5))]
    u = simulator.outsourcing_probs('uniform', d, p)
    assert np.all(u == d)


@pytest.mark.parametrize('model', simulator.OUTSOURCING_MODELS)
@pytest.mark.parametrize('a_ai', [0.95, 0.75])
def test_full_outsourcing_carries_no_signal(model, a_ai):
    bank = _bank(seed=3)
    world = simulator.World(bank, 2000, 0.0, 0.25, 0.3, a_ai, model, 1.0, seed=5)
    fa, fb, fc = world.followup_bank()
    cfg = SPRTConfig(cut=0.0, delta=0.4, n_max=20)
    dec = {0: [], 1: []}
    for l in world.learners:
        if l.truth == -1:
            continue
        arm = run_sprt(fa, fb, fc, cfg, lambda j, i=l.id: world.respond(i, j))
        dec[l.truth].append(arm.state.decision == 1)
    r1, r0 = np.mean(dec[1]), np.mean(dec[0])
    se = math.sqrt(r1 * (1 - r1) / len(dec[1]) + r0 * (1 - r0) / len(dec[0])) + 1e-9
    assert abs(r1 - r0) < 4 * se, 'a policy recovered understanding from fully outsourced answers: leak'


def test_no_outsourcing_reproduces_the_item_model():
    bank = _bank(seed=4)
    world = simulator.World(bank, 400, 0.0, 0.25, 0.0, 0.95, 'uniform', 0.0, seed=9, theta_mean=0.5, theta_sd=0.0)
    fa, fb, fc = world.followup_bank()
    js = np.arange(10)
    obs = np.array([[world.respond(l.id, j) for j in js] for l in world.learners]).mean(axis=0)
    want = p3pl(0.5, fa[js], fb[js], fc[js])
    assert np.all(abs(obs - want) < 4 * np.sqrt(want * (1 - want) / 400))


def test_outsourced_answers_ignore_understanding():
    bank = _bank(seed=6)
    for theta in (-2.0, 2.0):
        world = simulator.World(bank, 3000, 0.0, 0.25, 0.0, 0.75, 'uniform', 1.0, seed=2, theta_mean=theta, theta_sd=0.0)
        rate = np.mean([world.respond(l.id, 0) for l in world.learners])
        assert abs(rate - 0.75) < 4 * math.sqrt(0.75 * 0.25 / 3000)


def test_understanding_keeps_borderline_and_cohort_ignores_truth():
    t = simulator.understanding(np.array([-1.0, -0.1, 0.0, 0.1, 1.0]), cut=0.0, margin=0.2)
    assert t.tolist() == [0, -1, -1, -1, 1]
    bank = _bank(seed=7)
    world = simulator.World(bank, 50, 0.0, 0.25, 0.5, 0.95, 'uniform', 0.5, seed=3)
    before = [l.id for l in world.cohort(0.6)]
    for l in world.learners:
        l.truth = 1 - l.truth if l.truth >= 0 else l.truth
    assert [l.id for l in world.cohort(0.6)] == before


def test_arm_interface_sees_no_generator():
    bank = _bank(seed=8)
    world = simulator.World(bank, 3, 0.0, 0.25, 0.0, 0.95, 'uniform', 0.0, seed=1)
    fa, fb, fc = world.followup_bank()
    fa[:] = 99.0                                        # a copy: arms cannot write back into the generator
    assert not np.any(world.bank.a == 99.0)


# 4. Item fit recovers known parameters; the development split is by ID, not outcome.

def test_fit_2pl_recovers_parameters():
    rng = np.random.default_rng(12)
    n, m = 3000, 20
    a = rng.uniform(0.6, 2.0, m); b = rng.normal(0, 1, m)
    theta = rng.normal(0, 1, n)
    resp = (rng.random((n, m)) < p3pl(theta[:, None], a, b, 0.0)).astype(float)
    resp[rng.random((n, m)) < 0.3] = np.nan
    ah, bh, hist = fit.fit_2pl(resp)
    assert np.corrcoef(a, ah)[0, 1] > 0.9 and np.corrcoef(b, bh)[0, 1] > 0.98
    assert all(y >= x - 1e-6 * abs(x) for x, y in zip(hist, hist[1:]))


def test_dev_split_is_stable_and_about_half():
    ids = [str(i) for i in range(4000)]
    share = np.mean([fit.dev_learner(i) for i in ids])
    assert 0.45 < share < 0.55 and fit.dev_learner('64525') == fit.dev_learner('64525')


def test_fisher_info_matches_numeric_derivative():
    a, b, c, t, h = 1.4, 0.3, 0.2, 0.1, 1e-6
    p = p3pl(t, a, b, c)
    dp = (p3pl(t + h, a, b, c) - p3pl(t - h, a, b, c)) / (2 * h)
    assert fisher_info(t, a, b, c) == pytest.approx(dp ** 2 / (p * (1 - p)), rel=1e-6)


# 5. Full-length SPRT path, per-learner log, signature statistics (rulings of 25 Oct 2026).

from plainproof import signature  # noqa: E402
from plainproof.sprt import run_sprt_full  # noqa: E402


def test_full_sprt_path_matches_stopped_path_up_to_native_stop():
    bank = _bank(seed=10)
    world = simulator.World(bank, 60, 0.0, 0.25, 0.0, 0.95, 'uniform', 0.0, seed=4)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    table = np.array([[world.respond(i, j) for j in range(m)] for i in range(60)])
    cfg = SPRTConfig(cut=0.0, delta=0.7, n_max=20)
    for i in range(60):
        stopped = run_sprt(fa, fb, fc, cfg, lambda j: int(table[i, j]))
        items, _, lrs, stop, dec = run_sprt_full(fa, fb, fc, cfg, lambda j: int(table[i, j]))
        assert len(items) == 20 and len(set(items)) == 20
        n = len(stopped.state.items)
        assert items[:n] == stopped.state.items and np.allclose(lrs[:n], stopped.state.log_ratios)
        if stopped.state.decision in (0, 1):
            assert stop == n and dec == stopped.state.decision
        else:
            assert stop is None and dec == 'truncated'


@pytest.mark.parametrize('d,expect', [(0.0, 0.0), (1.0, 1.0)])
def test_log_keeps_required_fields_and_true_outsourced_flag(d, expect):
    bank = _bank(seed=11)
    world = simulator.World(bank, 20, 0.0, 0.25, 0.3, 0.95, 'uniform', d, seed=6)
    m = len(world.followup_bank()[0])
    for i in range(20):
        for j in range(m):
            world.respond(i, j)
    for i, rows in world.log.items():
        assert {'homework', 'followup', 'transfer'} == {r['bank'] for r in rows}
        for r in rows:
            assert all(f in r for f in signature.REQUIRED_LOG_FIELDS)
        fu = [r['outsourced'] for r in rows if r['bank'] == 'followup']
        assert len(fu) == m and np.mean(fu) == expect
        assert not any(r['outsourced'] for r in rows if r['bank'] == 'transfer')


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(3)
    h = rng.integers(0, 5, 40).astype(float); o = rng.integers(1, 6, 30).astype(float)
    want = np.mean([1.0 if y > x else 0.5 if y == x else 0.0 for x in h for y in o])
    assert signature.auc(h, o, +1) == pytest.approx(want)
    assert signature.auc(h, o, -1) == pytest.approx(1 - want)
    assert signature.auc(h, h, +1) == pytest.approx(0.5)


def test_llr_is_zero_without_outsourcing_and_signature_directions_are_fixed():
    assert signature.llr_outsourcing([1.0, 1.2], [0.0, 0.5], [0.0, 0.0], [1, 0], 0.2, 0.0, 0.95) == pytest.approx(0.0)
    assert set(signature.DIRECTION) == {'difficulty_correctness', 'lz', 'llr_outsourcing', 'hw_minus_fu',
                                        'fu_minus_transfer', 'sprt_confident_wrong', 'probe_excess'}
    assert signature.SEPARATION == 0.75 and signature.CHECKPOINTS == (5, 10, 15, 20)


def test_signature_statistics_compute_on_a_world_log():
    bank = _bank(seed=12)
    world = simulator.World(bank, 5, 0.0, 0.25, 0.3, 0.95, 'uniform', 0.5, seed=7)
    fa, fb, fc = world.followup_bank()
    m = len(fa)
    for i in range(5):
        for j in range(m):
            world.respond(i, j)
    order = [int(x) for x in bank.rows('followup')[:20]]
    s = signature.learner_stats(world.log[0], bank, order, 10, 0.5, 0.95, [0.5] * 20, world.learners[0].truth)
    six = [k for k in signature.DIRECTION if k != 'probe_excess']       # statistic 7 has its own per-method function
    assert set(six) <= set(s) and all(np.isfinite(s[k]) or k == 'sprt_confident_wrong' for k in six)


def test_probes_are_the_hardest_followups_and_excess_is_positive_under_full_outsourcing():
    from plainproof import bank_from_fit
    bank = _bank(seed=13)
    q = np.arange(len(bank.a)) + 1000
    rows, ids = bank_from_fit.probes(bank, q, count=5)
    fu = bank.rows('followup')
    assert set(rows) <= set(fu) and len(rows) == 5
    assert min(bank.b[rows]) >= max(bank.b[[j for j in fu if j not in rows]])
    assert ids == [int(q[j]) for j in rows]
    world = simulator.World(bank, 300, 0.0, 0.25, 0.0, 0.95, 'uniform', 1.0, seed=8, theta_mean=-0.5, theta_sd=0.0)
    m = len(world.followup_bank()[0])
    for i in range(300):
        for j in range(m):
            world.respond(i, j)
    order = sorted(fu, key=lambda j: -bank.b[j])       # an order that asks the probes first
    ex = [signature.probe_excess(world.log[i], bank, order, 10, rows) for i in range(300)]
    assert np.nanmean(ex) > 0.2
    assert np.isnan(signature.probe_excess(world.log[0], bank, [j for j in fu if j not in rows], 10, rows))
