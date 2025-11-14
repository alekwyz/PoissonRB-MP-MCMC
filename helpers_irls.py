# helpers_irls.py
import numpy as np

def poisson_map_irls(XX, y, alpha=100.0, tol=1e-8, iters=100):
    """Return (theta_hat, cov_hat) where cov_hat ≈ (X^T W X + alpha^{-1}I)^{-1} at MAP."""
    n, d = XX.shape
    lam = (y.mean() + 1e-6)
    th = np.zeros(d)                    # init
    eye = np.eye(d)
    for _ in range(iters):
        eta = XX @ th
        lam = np.exp(np.clip(eta, -20, 20))
        W = lam                         # diag weights
        z = eta + (y - lam) / (lam + 1e-12)
        XtW = XX.T * W
        H = XtW @ XX + (1.0/alpha) * eye
        g = XtW @ z
        # Solve H th = g (Newton step)
        th_new = np.linalg.solve(H, g)
        if np.linalg.norm(th_new - th) < tol * (1 + np.linalg.norm(th)):
            th = th_new
            break
        th = th_new
    cov = np.linalg.inv(H)
    return th, cov
