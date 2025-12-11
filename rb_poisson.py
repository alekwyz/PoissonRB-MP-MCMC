
# rb_poisson.py
import numpy as np
from scipy.stats import norm

class RB_BayesianPoissonReg:
    def __init__(self, N, StepSize, PowerOfTwo, x0, InitMean, InitCov,
                 XX, y, alpha=100.0, Stream="iid"):
        """
        RB multiple-proposal MCMC for Poisson GLM with log link.
        Prior: theta ~ N(0, alpha I)
        Proposals: Gaussian independence around InitMean, InitCov (scaled by StepSize).
        """
        m, d = XX.shape
        self.N = N
        self.d = d
        self.AcceptVals = []
        self.xVals = [x0]

        # diagnostics (init)
        self.max_weight_history = []  # track max(P) per iter (peakedness)
        self.entropy_history = []  # track -sum P log P per iter (diversity)

        # seeds for uniforms in R^{d+1}
        from Seed import SeedGen
        xs = SeedGen(d + 1, PowerOfTwo, Stream)

        ApprMean = InitMean
        ApprCov  = InitCov
        CholAppr = np.linalg.cholesky(ApprCov)
        InvAppr  = np.linalg.inv(ApprCov)

        # precompute prior precision
        PrecPrior = np.eye(d) / alpha

        # weighted per-iter posterior mean (RB)
        NumOfIter = int(int((2**PowerOfTwo - 1) / (d + 1)) * (d + 1) / N)
        self.WeightedSum = np.zeros((NumOfIter, d))

        xI = self.xVals[0]

        for n in range(NumOfIter):
            U = xs[n*N:(n+1)*N, :]

            # proposals (Gaussian via inverse-CDF to keep parity with your code)
            Y = ApprMean + norm.ppf(U[:, :d], 0.0, StepSize) @ CholAppr

            # prepend current state
            # (N+1,d)
            Proposals = np.vstack([xI, Y])
            # (m, N+1)
            eta = XX @ Proposals.T

            # log posterior (up to const)
            # prior
            LogPrior = -0.5 * (Proposals @ PrecPrior * Proposals).sum(axis=1)
            # Poisson log-lik: sum_i [ y_i * eta_i - exp(eta_i) ]  (drop log y! constant)
            LogLik = y @ eta - np.exp(eta).sum(axis=0)
            LogPost = LogPrior + LogLik

            # product of transition densities κ-tilde
            D = Proposals - ApprMean
            LogK_ni = -0.5 * (D @ (InvAppr / (StepSize**2)) * D).sum(axis=1)
            LogKs = LogK_ni.sum() - LogK_ni

            # normalized RB weights
            L = LogPost + LogKs
            mx = L.max()
            P = np.exp(L - (mx + np.log1p(np.exp(L - mx).sum() - 1.0)))  # log-sum-exp

            #Track whether MP-MCMC weights are collapsing; this guides tuning.
            maxw = float(P.max())
            H = float(-(P * np.log(P + 1e-16)).sum())
            self.max_weight_history.append(maxw)
            self.entropy_history.append(H)

            # RB mean for this iteration
            self.WeightedSum[n, :] = (P[:, None] * Proposals).sum(axis=0)

            # sample N indices using last uniform coordinate
            cdf = np.cumsum(P)
            idx = np.searchsorted(cdf, U[:, d])
            x_new = Proposals[idx]
            self.xVals.append(x_new.copy())

            # acceptance diagnostic
            self.AcceptVals.append(1.0 - P[idx])

            # propagate last state
            xI = Proposals[idx[-1]]

    def GetSamples(self, BurnIn=0):
        return np.concatenate(self.xVals[1:], axis=0)[BurnIn:, :]

    def GetAcceptRate(self, BurnIn=0):
        return np.concatenate(self.AcceptVals)[BurnIn:].mean()

    def Get_MeanEstimate(self, N, BurnIn=0):
        return self.WeightedSum[int(BurnIn/N):].mean(axis=0)

    def GetDiagnostics(self):
        # tolerate older objects that might not have these attrs
        maxw = getattr(self, "max_weight_history", [])
        H = getattr(self, "entropy_history", [])
        return {"maxw": np.asarray(maxw), "entropy": np.asarray(H)}

    def Get_RBIterMeans(self):
        """
        Return per-iteration Rao-Blackwellised mean contributions:
        array of shape (L, d), where row ℓ is sum_i w_i^{(ℓ)} y_i^{(ℓ)}.
        """
        return self.WeightedSum


