"""E0b: planted hard probes (ruling 25 Oct 2026). New code; the frozen E0 modules are not changed.

Three synthetic probe items per learner, difficulty 2.5, 3.0 and 3.5, discrimination at the bank median, c = 0,
planted at combined positions 4, 9 and 14 of every method's path (positions count every item asked). Probes are
asked but unscored: their answers never enter SPRT, BOBCAT, the posterior, calibration or stopping.

Outsourcing of a probe uses the learner's own outsourcing model on the same dial d:
  uniform      probability d;
  uncertainty  min(1, s * g(p)), g(p) = 4 p (1 - p), with s the learner's scale solved on the follow-up bank alone
               (as simulator.outsourcing_probs does), so the bank's share stays exactly d.
Probe answers come from their own random stream, so scored answers are unchanged by planting.
"""
import numpy as np

from .irt import p3pl
from .simulator import g

PROBE_B = (2.5, 3.0, 3.5)
POSITIONS = (4, 9, 14)


def scale(d, p_bank):
    """The uncertainty model's per-learner scale s on the bank alone (same bisection as outsourcing_probs)."""
    w = g(np.asarray(p_bank, dtype=float))
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * w).mean() < d:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if np.minimum(1.0, mid * w).mean() < d:
            lo = mid
        else:
            hi = mid
    return hi


def probe_outsourcing(model, d, p_probe, p_bank):
    p_probe = np.asarray(p_probe, dtype=float)
    if model == 'uniform' or d in (0.0, 1.0):
        return np.full(p_probe.shape, float(d))
    return np.minimum(1.0, scale(d, p_bank) * g(p_probe))


def plant(theta, model, d, a_ai, a_probe, bank_a, bank_b, bank_c, rng):
    """One learner's three probe answers and true outsourced flags."""
    b = np.array(PROBE_B)
    p_own = p3pl(theta, a_probe, b, 0.0)
    q = probe_outsourcing(model, d, p_own, p3pl(theta, bank_a, bank_b, bank_c))
    out = rng.random(3) < q
    correct = np.where(out, rng.random(3) < a_ai, rng.random(3) < p_own)
    return correct.astype(int), out


def scored_before(k):
    """Number of scored items within the first k combined positions."""
    return k - sum(1 for p in POSITIONS if p <= k)


def probes_before(k):
    return sum(1 for p in POSITIONS if p <= k)


# Third outsourcing model, exploratory (ruling 25 Oct 2026): give-up. The outsourcing weight rises as the learner's
# own chance on the item falls, peaking at p = 0: w(p) = 1 - p, scaled per learner so the mean share over the
# follow-up bank equals d, like the other two models. Defined here so the frozen simulator is not changed.

def w_giveup(p):
    return 1.0 - np.asarray(p, dtype=float)


def _scale_for(weights, d):
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * weights).mean() < d:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if np.minimum(1.0, mid * weights).mean() < d:
            lo = mid
        else:
            hi = mid
    return hi


def giveup_probs(d, p_bank):
    if d in (0.0, 1.0):
        return np.full(np.shape(p_bank), float(d))
    w = w_giveup(p_bank)
    return np.minimum(1.0, _scale_for(w, d) * w)


def probe_outsourcing_any(model, d, p_probe, p_bank):
    if model == 'giveup':
        p_probe = np.asarray(p_probe, dtype=float)
        if d in (0.0, 1.0):
            return np.full(p_probe.shape, float(d))
        return np.minimum(1.0, _scale_for(w_giveup(p_bank), d) * w_giveup(p_probe))
    return probe_outsourcing(model, d, p_probe, p_bank)


def plant_any(theta, model, d, a_ai, a_probe, bank_a, bank_b, bank_c, rng):
    b = np.array(PROBE_B)
    p_own = p3pl(theta, a_probe, b, 0.0)
    q = probe_outsourcing_any(model, d, p_own, p3pl(theta, bank_a, bank_b, bank_c))
    out = rng.random(3) < q
    correct = np.where(out, rng.random(3) < a_ai, rng.random(3) < p_own)
    return correct.astype(int), out


def make_world(bank, n, model, d, a_ai, seed, h_share=0.3, cut=0.0, margin=0.25):
    """A frozen simulator.World for 'uniform'/'uncertainty'; for 'giveup', a subclass whose respond() uses the
    give-up probabilities with the same random draws and logging as the frozen respond()."""
    from .simulator import World
    if model != 'giveup':
        return World(bank, n, cut=cut, margin=margin, h_share=h_share, a_ai=a_ai, model=model, d=d, seed=seed)

    class GiveUpWorld(World):
        def respond(self, learner_id, followup_index):
            l = self.learners[learner_id]
            j = self._fu[followup_index]
            p_own = float(p3pl(l.theta, self.bank.a[j], self.bank.b[j], self.bank.c[j]))
            if learner_id not in self._q:
                fa, fb, fc = self.followup_bank()
                self._q[learner_id] = giveup_probs(self.d, p3pl(l.theta, fa, fb, fc))
            if self.rng.random() < self._q[learner_id][followup_index]:
                u, outsourced = int(self.rng.random() < self.a_ai), True
            else:
                u, outsourced = int(self.rng.random() < p_own), False
            self.log[learner_id].append(self._row(j, 'followup', u, outsourced))
            return u

    w = GiveUpWorld(bank, n, cut=cut, margin=margin, h_share=h_share, a_ai=a_ai, model='uniform', d=d, seed=seed)
    w.model = 'giveup'
    return w
