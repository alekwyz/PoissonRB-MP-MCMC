# mh_tuning.py
import numpy as np
from mh_sampler import RW_BayesianPoissonReg

def tune_mh_stepsize(XX, y, x0, alpha, target=0.23, grid=None, seed=123, iters=3000):
    if grid is None:
        grid = np.logspace(-3, -0.5, 10)  # adjust if needed
    best = None
    np.random.seed(seed)
    for s in grid:
        mh = RW_BayesianPoissonReg(N_iter=iters, StepSize=s, x0=x0, XX=XX, y=y, alpha=alpha)
        acc = mh.GetAcceptRate()
        score = abs(acc - target)
        if best is None or score < best[0]:
            best = (score, s, acc)
    return {"StepSize": best[1], "AcceptRate": best[2]}
