
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class PoissonSimConfig:
    # number of observations
    n: int = 500
    # number of raw covariates (excl. intercept)
    D: int = 8
    # target E[y] roughly (controls intercept shift)
    mean_rate: float = 2.
    # desired sd of linear predictor for numerical stability
    eta_sd: float = 0.6
    # RNG seed (None = random)
    seed: Optional[int] = 0
    x_dist: str = "normal"
    # z-score X columns
    standardize: bool = True
    check_full_rank: bool = True

class PoissonDataLoad:
    """
    Simulate and package a Poisson GLM dataset (log link).
    Y ~ Poisson( lambda_i ),  log lambda_i = beta0 + X_i·beta
    """
    def __init__(self, cfg: PoissonSimConfig, beta_true: Optional[np.ndarray] = None):
        rng = np.random.default_rng(cfg.seed)

        # Generate covariates X (n x D)
        if cfg.x_dist == "normal":
            X = rng.normal(size=(cfg.n, cfg.D))
        elif cfg.x_dist == "uniform":
            X = rng.uniform(-1, 1, size=(cfg.n, cfg.D))
        else:
            raise ValueError("x_dist must be 'normal' or 'uniform'")

        # z-score for better conditioning
        if cfg.standardize:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, ddof=1, keepdims=True)
            sd[sd == 0.0] = 1.0
            X = (X - mu) / sd

        # Choose true coefficients
        if beta_true is None:
            # Slightly sparse-ish signal by default
            beta_large = rng.uniform(0.5, 1.0, size=min(3, cfg.D))
            beta_small = rng.uniform(0.0, 0.3, size=cfg.D - len(beta_large))
            beta_true = np.concatenate([beta_large, beta_small])
            rng.shuffle(beta_true)
        beta_true = beta_true.reshape(cfg.D,)

        # Build linear predictor (pre-scale) and stabilize it
        eta_raw = X @ beta_true  # n-vector before intercept/centering
        # Rescale eta to have a controlled spread (keeps exp(eta) numerically tame)
        s = np.std(eta_raw)
        if s > 0:
            eta = (cfg.eta_sd / s) * eta_raw
        else:
            eta = eta_raw.copy()

        # Center and shift intercept so that mean rate ~ mean_rate
        # mean of log-rate = log(mean_rate) - 0.5*var(log-rate) approx for Poisson log-link
        eta = eta - eta.mean()
        beta0 = np.log(cfg.mean_rate) - 0.5 * (np.std(eta) ** 2)
        eta = beta0 + eta

        # Simulate counts
        # Guard against extreme lambda that can overflow exp(.) or induce numerical issues
        lam = np.exp(np.clip(eta, a_min=-8.0, a_max=8.0))  # [~0.0003, ~2980]
        y = rng.poisson(lam)

        # Make sure we didn't generate a degenerate sample (e.g., all zeros)
        if np.all(y == 0):
            raise RuntimeError("All y are zero; tweak mean_rate / eta_sd / beta_true to avoid degeneracy.")

        # Create design matrix with intercept
        XX = np.concatenate([np.ones((cfg.n, 1)), X], axis=1)
        self.XX = XX
        # responses
        self.t = y.astype(float)
        # m = n, d = D+1
        self.m, self.d = self.XX.shape
        self.beta_true = np.concatenate([[beta0], beta_true])

        # Optional checks relevant to the algorithms
        # - With a Gaussian prior N(0, alpha I), the posterior is continuous & proper for finite data.
        # - Independence kernels in RB/ARB satisfy the “independence” assumption if you use fixed (mu, Sigma).
        # - We ensure the design is full column rank (helps with mode/Hessian computations & stability).
        if cfg.check_full_rank:
            if np.linalg.matrix_rank(self.XX) < self.d:
                raise RuntimeError("Design matrix not full rank; regenerate or adjust standardization/seed.")

        # Keep some diagnostics handy
        self.diag = {
            "y_mean": float(y.mean()),
            "y_max": int(y.max()),
            "eta_mean": float(eta.mean()),
            "eta_sd": float(eta.std()),
            "lambda_range": (float(lam.min()), float(lam.max())),
        }

    def GetDimension(self):      # includes intercept
        return self.d
    def GetDesignMatrix(self):
        return self.XX
    def GetNumOfSamples(self):
        return self.m
    def GetResponses(self):
        return self.t
    def GetTrueBeta(self):
        return self.beta_true
    def GetDiag(self):
        return self.diag
