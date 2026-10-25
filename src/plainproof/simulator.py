"""Learner worlds for the outsourcing study. Truth lives here and never reaches an arm.

One concept per world. Latent ability theta is drawn per learner; understanding is defined from theta alone
(understands / gap / borderline), independent of every estimator. Items belong to families; within each
family, homework, follow-up and transfer items come from disjoint banks, and some families are held out
for transfer only.

Answering:
  own answer        correct with the 3PL probability P(theta) (guessing via c; slips via the upper curve)
  outsourced answer correct with probability a_ai, whatever the learner's understanding
Outsourcing models for follow-ups, on the same dial d in [0, 1]:
  uniform      every follow-up is outsourced with probability d
  uncertainty  item j is outsourced with probability min(1, s * g(p_j)), g(p) = 4 p (1 - p), which peaks at
               p = 0.5; s is solved per learner so the mean over that learner's follow-up bank equals d.
Homework is outsourced at its own share h (uniform). Transfer answers are always the learner's own.
"""
from dataclasses import dataclass

import numpy as np

from .irt import p3pl

OUTSOURCING_MODELS = ('uniform', 'uncertainty')


@dataclass(frozen=True)
class Bank:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    family: np.ndarray
    role: np.ndarray          # 'homework', 'followup', 'transfer'

    def rows(self, role):
        return np.flatnonzero(self.role == role)


def split_bank(a, b, c, family, held_out_families, rng, shares=(0.4, 0.4, 0.2)):
    """Assign roles. Held-out families are transfer-only; every other family is split three ways."""
    a, b, c, family = (np.asarray(x) for x in (a, b, c, family))
    role = np.empty(len(a), dtype=object)
    for f in np.unique(family):
        idx = np.flatnonzero(family == f)
        if f in set(held_out_families):
            role[idx] = 'transfer'
            continue
        idx = rng.permutation(idx)
        n_hw = int(round(shares[0] * len(idx)))
        n_fu = int(round(shares[1] * len(idx)))
        role[idx[:n_hw]] = 'homework'
        role[idx[n_hw:n_hw + n_fu]] = 'followup'
        role[idx[n_hw + n_fu:]] = 'transfer'
    return Bank(a.astype(float), b.astype(float), c.astype(float), family, role)


def understanding(theta, cut, margin):
    """1 understands, 0 gap, -1 borderline (kept as its own category, never forced to a side)."""
    theta = np.asarray(theta)
    out = np.full(theta.shape, -1)
    out[theta >= cut + margin] = 1
    out[theta <= cut - margin] = 0
    return out


def g(p):
    return 4.0 * p * (1.0 - p)


def outsourcing_probs(model, d, p_own):
    """Per-item outsourcing probabilities for one learner's follow-up bank. Their mean equals d."""
    p_own = np.asarray(p_own, dtype=float)
    if not 0 <= d <= 1:
        raise ValueError('d')
    if model == 'uniform':
        return np.full(p_own.shape, float(d))
    if model != 'uncertainty':
        raise ValueError('outsourcing model')
    if d == 0:
        return np.zeros(p_own.shape)
    if d == 1:
        return np.ones(p_own.shape)
    w = g(p_own)
    if not np.any(w > 0):
        raise ValueError('no item near p = 0.5 can carry the share')
    lo, hi = 0.0, 1.0
    while np.minimum(1.0, hi * w).mean() < d:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError('share d not reachable')
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if np.minimum(1.0, mid * w).mean() < d:
            lo = mid
        else:
            hi = mid
    return np.minimum(1.0, hi * w)


@dataclass
class Learner:
    id: int
    theta: float
    truth: int                 # 1 understands, 0 gap, -1 borderline
    homework_score: float
    transfer_score: float


class World:
    """One simulated population. Arms get only respond(); truth and theta stay here."""

    def __init__(self, bank, n_learners, cut, margin, h_share, a_ai, model, d, seed,
                 theta_mean=0.0, theta_sd=1.0):
        if model not in OUTSOURCING_MODELS:
            raise ValueError('outsourcing model')
        if not (0 <= h_share <= 1 and 0 <= a_ai <= 1 and 0 <= d <= 1):
            raise ValueError('share/accuracy')
        self.bank, self.cut, self.margin = bank, cut, margin
        self.a_ai, self.model, self.d = a_ai, model, d
        self.rng = np.random.default_rng(seed)
        theta = self.rng.normal(theta_mean, theta_sd, n_learners)
        truth = understanding(theta, cut, margin)
        hw, tr = bank.rows('homework'), bank.rows('transfer')
        self.learners = []
        self.log = {}      # truth-side per-learner log; never passed to an arm (same RNG draws as before)
        for i, t in enumerate(theta):
            p_hw = p3pl(t, bank.a[hw], bank.b[hw], bank.c[hw])
            out = self.rng.random(len(hw)) < h_share
            correct = np.where(out, self.rng.random(len(hw)) < a_ai, self.rng.random(len(hw)) < p_hw)
            p_tr = p3pl(t, bank.a[tr], bank.b[tr], bank.c[tr])
            tr_correct = self.rng.random(len(tr)) < p_tr
            transfer = tr_correct.mean() if len(tr) else float('nan')
            hw_score = float(correct.mean()) if len(hw) else float('nan')
            self.learners.append(Learner(i, float(t), int(truth[i]), hw_score, float(transfer)))
            self.log[i] = ([self._row(j, 'homework', c, o) for j, c, o in zip(hw, correct, out)]
                           + [self._row(j, 'transfer', c, False) for j, c in zip(tr, tr_correct)])
        self._fu = bank.rows('followup')
        self._q = {}

    def _row(self, j, bank_name, correct, outsourced):
        return dict(item=int(j), difficulty=float(self.bank.b[j]), discrimination=float(self.bank.a[j]),
                    bank=bank_name, family=int(self.bank.family[j]), correct=int(correct), outsourced=bool(outsourced))

    def cohort(self, min_homework):
        """'Comparable homework success': learners at or above a homework score, selected without truth."""
        return [l for l in self.learners if l.homework_score >= min_homework]

    def followup_bank(self):
        """Item parameters an arm may see: follow-up items only, never truth, theta or the generator."""
        f = self._fu
        return self.bank.a[f].copy(), self.bank.b[f].copy(), self.bank.c[f].copy()

    def respond(self, learner_id, followup_index):
        """One follow-up answer (0/1). followup_index indexes followup_bank()."""
        l = self.learners[learner_id]
        j = self._fu[followup_index]
        p_own = float(p3pl(l.theta, self.bank.a[j], self.bank.b[j], self.bank.c[j]))
        if learner_id not in self._q:
            fa, fb, fc = self.followup_bank()
            self._q[learner_id] = outsourcing_probs(self.model, self.d, p3pl(l.theta, fa, fb, fc))
        if self.rng.random() < self._q[learner_id][followup_index]:
            u, outsourced = int(self.rng.random() < self.a_ai), True
        else:
            u, outsourced = int(self.rng.random() < p_own), False
        self.log[learner_id].append(self._row(j, 'followup', u, outsourced))
        return u
