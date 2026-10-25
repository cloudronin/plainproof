"""Two-parameter logistic fit (c = 0) by marginal maximum likelihood (EM, Gauss-Hermite quadrature).

Used to anchor simulated item parameters in real responses (ASSIST2009 development learners only).
Missing responses are allowed. Guessing is not fitted: c is set per item in the simulator and stated
as an assumption, because sparse open-response data do not identify it.
"""
import hashlib

import numpy as np


def dev_learner(user_id, salt='edm-v2-e0'):
    """Fixed development/held-out split by hashed learner ID (half each); never by outcomes."""
    h = hashlib.sha256(f'{salt}:{user_id}'.encode()).digest()
    return h[0] % 2 == 0


def fit_2pl(resp, n_quad=31, iters=200, tol=1e-6, prior_sd_loga=0.5):
    """resp: learners x items array of 0/1 with NaN for missing. Returns (a, b, log-likelihood history)."""
    resp = np.asarray(resp, dtype=float)
    obs = ~np.isnan(resp)
    u = np.nan_to_num(resp)
    x, w = np.polynomial.hermite_e.hermegauss(n_quad)
    w = w / w.sum()
    n_items = resp.shape[1]
    a, b = np.ones(n_items), np.zeros(n_items)
    hist = []
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a[None, :] * (x[:, None] - b[None, :])))          # quad x items
        lp, lq = np.log(np.clip(p, 1e-12, 1)), np.log(np.clip(1 - p, 1e-12, 1))
        ll = (u * obs) @ lp.T + ((1 - u) * obs) @ lq.T                          # learners x quad
        m = ll.max(axis=1, keepdims=True)
        post = np.exp(ll - m) * w[None, :]
        marg = post.sum(axis=1, keepdims=True)
        hist.append(float((np.log(marg) + m).sum()))
        post /= marg
        n_k = post.T @ obs                                                       # expected attempts
        r_k = post.T @ (u * obs)                                                 # expected corrects
        for j in range(n_items):                                                  # Newton step per item
            for _ in range(5):
                pj = 1 / (1 + np.exp(-a[j] * (x - b[j])))
                res = r_k[:, j] - n_k[:, j] * pj
                ww = n_k[:, j] * pj * (1 - pj)
                dz_da, dz_db = x - b[j], -a[j]
                la = np.log(a[j])
                g = np.array([(res * dz_da).sum() * a[j] - la / prior_sd_loga ** 2, (res * dz_db).sum()])
                H = -np.array([[(ww * dz_da ** 2).sum() * a[j] ** 2 + 1 / prior_sd_loga ** 2,
                                (ww * dz_da * dz_db).sum() * a[j]],
                               [(ww * dz_da * dz_db).sum() * a[j], (ww * dz_db ** 2).sum()]])
                step = np.linalg.solve(H, g)
                la, b[j] = la - step[0], b[j] - step[1]
                a[j] = float(np.clip(np.exp(la), 0.05, 6.0))
                b[j] = float(np.clip(b[j], -6, 6))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2]):
            break
    return a, b, hist
