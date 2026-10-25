"""Wald SPRT classification at one cut score, with item selection by Fisher information at the cut.

Matches catIrt 0.5.1 (termSPRT, itChoose with select='UW-FI', at='bounds', one bound):
  log ratio = logLik(theta = cut + delta) - logLik(theta = cut - delta)
  upper = log((1 - beta) / alpha) -> master (1); lower = log(beta / (1 - alpha)) -> non-master (0).
  Next item: largest expected information at the cut among unused items; ties go to the earliest bank row
  (R's order(decreasing=TRUE) is stable).
Truncation at n_max is its own outcome ('truncated'), never a classification success. The theta-vs-cut
fallback that catIrt applies at truncation is kept separately (fallback_decision) for replay only.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .irt import fisher_info, loglik


@dataclass
class SPRTConfig:
    cut: float
    delta: float = 0.1
    alpha: float = 0.05
    beta: float = 0.05
    n_max: int = 50
    n_min: int = 1

    def __post_init__(self):
        if not (0 < self.alpha < 1 and 0 < self.beta < 1 and self.alpha + self.beta < 1):
            raise ValueError('alpha/beta')
        if self.delta <= 0 or self.n_max < 1 or not 1 <= self.n_min <= self.n_max:
            raise ValueError('delta/n')

    @property
    def upper(self):
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self):
        return math.log(self.beta / (1 - self.alpha))


@dataclass
class SPRTState:
    items: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    log_ratios: list = field(default_factory=list)
    decision: object = None   # 1 master, 0 non-master, 'truncated', or None while running


class SPRTArm:
    """One learner's sequential test over a fixed item bank (rows of a, b, c)."""

    def __init__(self, a, b, c, config: SPRTConfig):
        self.a, self.b, self.c = (np.asarray(x, dtype=float) for x in (a, b, c))
        if not (self.a.shape == self.b.shape == self.c.shape) or self.a.ndim != 1:
            raise ValueError('bank shape')
        self.cfg = config
        self.info_at_cut = fisher_info(config.cut, self.a, self.b, self.c)
        self.state = SPRTState()

    def next_item(self):
        if self.state.decision is not None:
            raise RuntimeError('test already ended')
        return self.next_item_ignoring_decision()

    def next_item_ignoring_decision(self):
        used = set(self.state.items)
        best, best_info = None, -math.inf
        for j, v in enumerate(self.info_at_cut):
            if j not in used and v > best_info:
                best, best_info = j, v
        if best is None:
            raise RuntimeError('bank exhausted')
        return best

    def log_ratio(self):
        it = self.state.items
        u = self.state.responses
        a, b, c = self.a[it], self.b[it], self.c[it]
        return (loglik(u, self.cfg.cut + self.cfg.delta, a, b, c)
                - loglik(u, self.cfg.cut - self.cfg.delta, a, b, c))

    def answer(self, item, response):
        if response not in (0, 1):
            raise ValueError('response must be 0 or 1')
        s = self.state
        s.items.append(int(item))
        s.responses.append(int(response))
        lr = self.log_ratio()
        s.log_ratios.append(lr)
        n = len(s.items)
        if n >= self.cfg.n_min:
            if lr >= self.cfg.upper:
                s.decision = 1
            elif lr <= self.cfg.lower:
                s.decision = 0
        if s.decision is None and (n >= self.cfg.n_max or n >= len(self.a)):
            s.decision = 'truncated'
        return s.decision

    def p_mastery(self):
        """Reporting score for matched-coverage ranking: logistic of the log ratio
        (posterior of 'cut + delta' against 'cut - delta' at equal prior). Not a calibrated probability."""
        lr = self.state.log_ratios[-1] if self.state.log_ratios else 0.0
        return 1.0 / (1.0 + math.exp(-lr))


def run_sprt(a, b, c, config, respond):
    """Run one test. respond(item_index) -> 0/1. Returns the final state."""
    arm = SPRTArm(a, b, c, config)
    while arm.state.decision is None:
        j = arm.next_item()
        arm.answer(j, respond(j))
    return arm


def run_sprt_full(a, b, c, config, respond):
    """Run the fixed SPRT order to config.n_max without stopping (ruling 25 Oct 2026).

    Returns (items, responses, log_ratios, native_stop, native_decision): native_stop is the first step at which
    the boundary fired (None if never within n_max), native_decision its classification (1/0) or 'truncated'.
    The same items, in the same order, as run_sprt up to the native stop.
    """
    arm = SPRTArm(a, b, c, config)
    native_stop, native_decision = None, 'truncated'
    for _ in range(min(config.n_max, len(arm.a))):
        j = arm.next_item_ignoring_decision()
        arm.state.items.append(int(j))
        arm.state.responses.append(int(respond(j)))
        lr = arm.log_ratio()
        arm.state.log_ratios.append(lr)
        n = len(arm.state.items)
        if native_stop is None and n >= config.n_min and (lr >= config.upper or lr <= config.lower):
            native_stop, native_decision = n, (1 if lr >= config.upper else 0)
    return arm.state.items, arm.state.responses, arm.state.log_ratios, native_stop, native_decision
