"""Expected SPRT stopping time when every item is identical (one fixed Bernoulli coin per response).

With uniform outsourcing share d and AI accuracy a_ai, each follow-up is correct with probability
    p_true = (1 - d) * P(theta) + d * a_ai,
independently, while the SPRT still scores answers with the item model at cut +/- delta. The log ratio after
k correct of n answers is k * w1 + (n - k) * w0, so the walk lives on (n, k) and the exact law of the stopping
time, truncation included, is a finite-horizon calculation. Wald's approximation (no overshoot, no
truncation) is reported beside it for reference; it is not used as the test truth.
"""
import math

import numpy as np
from scipy.optimize import brentq

from .irt import p3pl


def increments(a, b, c, cut, delta):
    p_hi, p_lo = p3pl(cut + delta, a, b, c), p3pl(cut - delta, a, b, c)
    return math.log(p_hi / p_lo), math.log((1 - p_hi) / (1 - p_lo))   # (correct, incorrect)


def p_correct(theta, a, b, c, d, a_ai):
    if not (0 <= d <= 1 and 0 <= a_ai <= 1):
        raise ValueError('d/a_ai')
    return (1 - d) * float(p3pl(theta, a, b, c)) + d * a_ai


def exact(p_true, w1, w0, upper, lower, n_max, n_min=1):
    """Exact stopping law. Returns expected length and the probabilities of each outcome."""
    alive = np.zeros(n_max + 1)          # alive[k]: probability of k correct so far and not yet stopped
    alive[0] = 1.0
    e_n = 0.0
    p_master = p_non = 0.0
    for n in range(1, n_max + 1):
        nxt = np.zeros(n_max + 1)
        nxt[1:n + 1] += alive[:n] * p_true
        nxt[0:n] += alive[:n] * (1 - p_true)
        k = np.arange(n + 1)
        lr = k * w1 + (n - k) * w0
        if n >= n_min:
            up = (lr >= upper) & (nxt[:n + 1] > 0)
            dn = (lr <= lower) & (nxt[:n + 1] > 0)
            pm, pn = nxt[:n + 1][up].sum(), nxt[:n + 1][dn].sum()
            p_master += pm
            p_non += pn
            e_n += n * (pm + pn)
            nxt[:n + 1][up | dn] = 0.0
        alive = nxt
    p_trunc = alive.sum()
    e_n += n_max * p_trunc
    return dict(expected_n=e_n, p_master=p_master, p_non_master=p_non, p_truncated=p_trunc)


def wald(p_true, w1, w0, upper, lower):
    """Wald's approximation without truncation: P(master) from the OC function, E[N] from Wald's identity."""
    mu = p_true * w1 + (1 - p_true) * w0
    A, B = upper, lower
    if abs(mu) < 1e-12:
        var = p_true * w1 ** 2 + (1 - p_true) * w0 ** 2
        return dict(expected_n=-A * B / var, p_master=-B / (A - B))
    f = lambda h: p_true * math.exp(h * w1) + (1 - p_true) * math.exp(h * w0) - 1
    # Non-zero root h of E[exp(h Z)] = 1 has the sign opposite to mu.
    lo, hi = (1e-9, 1.0) if mu < 0 else (-1.0, -1e-9)
    while f(hi if mu < 0 else lo) <= 0:
        if mu < 0:
            hi *= 2
        else:
            lo *= 2
    h = brentq(f, lo, hi)
    p_master = (1 - math.exp(h * B)) / (math.exp(h * A) - math.exp(h * B))
    return dict(expected_n=(p_master * A + (1 - p_master) * B) / mu, p_master=p_master)
