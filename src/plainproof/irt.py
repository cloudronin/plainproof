"""Three-parameter logistic model with D = 1, as catIrt's brm: P = c + (1 - c) / (1 + exp(-a (theta - b)))."""
import numpy as np


def p3pl(theta, a, b, c):
    """Probability of a correct answer. Broadcasts over theta and item arrays."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def fisher_info(theta, a, b, c):
    """Expected Fisher information, catIrt FI.brm(type='expected'): p'(theta)^2 / (p q)."""
    p = p3pl(theta, a, b, c)
    e = np.exp(-a * (theta - b))
    dp = (1.0 - c) * a * e / (1.0 + e) ** 2
    return dp ** 2 / (p * (1.0 - p))


def loglik(u, theta, a, b, c):
    """Sum of Bernoulli log-likelihoods for responses u (0/1) at one theta."""
    p = p3pl(theta, np.asarray(a), np.asarray(b), np.asarray(c))
    u = np.asarray(u, dtype=float)
    return float(np.sum(u * np.log(p) + (1.0 - u) * np.log(1.0 - p)))
