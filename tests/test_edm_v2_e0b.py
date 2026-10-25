"""E0b planted probes: positions, outsourcing scale, separate random stream. Offline only."""
import numpy as np
import pytest

from plainproof import planted, simulator


def test_positions_count_every_item_asked():
    assert [planted.scored_before(k) for k in (5, 10, 15, 20)] == [4, 8, 12, 17]
    assert [planted.probes_before(k) for k in (3, 4, 9, 14, 20)] == [0, 1, 2, 3, 3]


@pytest.mark.parametrize('d', [0.25, 0.5, 0.75])
def test_scale_matches_the_frozen_outsourcing_model(d):
    p = np.linspace(0.05, 0.95, 50)
    s = planted.scale(d, p)
    assert np.allclose(np.minimum(1.0, s * simulator.g(p)), simulator.outsourcing_probs('uncertainty', d, p))


def test_probe_outsourcing_uniform_is_d_and_uncertainty_is_small_for_near_impossible_items():
    p_bank = np.linspace(0.2, 0.9, 50)
    assert np.all(planted.probe_outsourcing('uniform', 0.5, [0.01, 0.02, 0.03], p_bank) == 0.5)
    q = planted.probe_outsourcing('uncertainty', 0.5, [0.01, 0.02, 0.03], p_bank)
    assert np.all(q < 0.1)
    assert np.all(planted.probe_outsourcing('uncertainty', 1.0, [0.01], p_bank) == 1.0)


def test_planting_uses_its_own_stream():
    rng = np.random.default_rng(3)
    c, o = planted.plant(0.0, 'uniform', 1.0, 0.95, 1.15, np.ones(5), np.zeros(5), np.zeros(5), rng)
    assert o.all() and c.shape == (3,)


@pytest.mark.parametrize('d', [0.25, 0.5, 0.75])
def test_giveup_hits_the_dial_and_peaks_at_low_own_chance(d):
    p = np.linspace(0.02, 0.98, 60)
    q = planted.giveup_probs(d, p)
    assert q.mean() == pytest.approx(d, abs=1e-9) and np.all(np.diff(q) <= 1e-12)


def test_giveup_world_matches_frozen_world_when_nothing_is_outsourced():
    rng = np.random.default_rng(0)
    n = 60; fam = np.repeat(np.arange(6), 10)
    bank = simulator.split_bank(rng.lognormal(0, .3, n), rng.normal(0, 1, n), np.zeros(n), fam, (5,), rng)
    w0 = simulator.World(bank, 20, 0.0, 0.25, 0.3, 0.95, 'uniform', 0.0, seed=4)
    w1 = planted.make_world(bank, 20, 'giveup', 0.0, 0.95, 4)
    m = len(w0.followup_bank()[0])
    a = [[w0.respond(i, j) for j in range(m)] for i in range(20)]
    b = [[w1.respond(i, j) for j in range(m)] for i in range(20)]
    assert a == b and w1.model == 'giveup'
