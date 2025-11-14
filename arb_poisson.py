
import numpy as np
from scipy.stats import norm

class ARB_BayesianPoissonReg:
    def __init__(self, N, StepSize, PowerOfTwo, InitMean, InitCov,
                 XX, y, alpha=100.0, Stream="iid", WeightIn=0):
        """
        Adaptive RB multiple-proposal MCMC for Poisson GLM with log link.
        Adapt (mu, Sigma) using running RB means/covs (like your ARB logistic class).
        """
        m, d = XX.shape
        self.N = N
        self.d = d
        self.AcceptVals = []
        self.xVals = [InitMean.copy()]

        # diagnostics (init)
        self.max_weight_history = []  # track max(P) per iter (peakedness)
        self.entropy_history = []  # track -sum P log P per iter (diversity)

        from Seed import SeedGen
        xs = SeedGen(d + 1, PowerOfTwo, Stream)

        # iterations
        NumOfIter = int(int((2**PowerOfTwo - 1) / (d + 1)) * (d + 1) / N)

        # prior precision
        PrecPrior = np.eye(d) / alpha

        # RB storage (+M to incorporate past weight if desired)
        M = int(WeightIn / N) + 1
        self.WeightedSum = np.zeros((NumOfIter + M, d))
        self.WeightedCov = np.zeros((NumOfIter + M, d, d))
        self.WeightedSum[:M] = InitMean
        self.WeightedCov[:M] = InitCov

        # adaptive proposal params
        ApprMean = InitMean.copy()
        ApprCov  = InitCov.copy()
        CholAppr = np.linalg.cholesky(ApprCov)
        InvAppr  = np.linalg.inv(ApprCov)

        xI = self.xVals[0]

        for n in range(NumOfIter):
            U = xs[n*N:(n+1)*N, :]

            # proposals
            Y = ApprMean + norm.ppf(U[:, :d], 0.0, StepSize) @ CholAppr
            # (N+1,d)
            Proposals = np.vstack([xI, Y])
            # (m, N+1)
            eta = XX @ Proposals.T

            # log post
            LogPrior = -0.5 * (Proposals @ (np.eye(d)/alpha) * Proposals).sum(axis=1)
            LogLik   = y @ eta - np.exp(eta).sum(axis=0)
            LogPost  = LogPrior + LogLik

            # kappa-tilde
            D = Proposals - ApprMean
            LogK_ni = -0.5 * (D @ (InvAppr / (StepSize**2)) * D).sum(axis=1)
            LogKs = LogK_ni.sum() - LogK_ni

            # RB weights
            L = LogPost + LogKs
            mx = L.max()
            P = np.exp(L - (mx + np.log1p(np.exp(L - mx).sum() - 1.0)))

            #Track whether MP-MCMC weights are collapsing; this guides tuning.
            maxw = float(P.max())
            H = float(-(P * np.log(P + 1e-16)).sum())
            self.max_weight_history.append(maxw)
            self.entropy_history.append(H)

            # RB mean this iter
            self.WeightedSum[n+M] = (P[:, None] * Proposals).sum(axis=0)

            # adapt mean (running average of RB means)
            ApprMean = self.WeightedSum[:n+M+1].mean(axis=0)

            # RB covariance this iter
            C = Proposals - ApprMean
            outer = C[:, :, None] * C[:, None, :]
            self.WeightedCov[n+M] = (P[:, None, None] * outer).sum(axis=0)

            # adapt cov after enough samples
            if n > 2 * d / N:
                ApprCov = self.WeightedCov[:n+M+1].mean(axis=0)
                # ensure PD (add a tiny jitter if needed)
                eps = 1e-8
                try:
                    CholAppr = np.linalg.cholesky(ApprCov)
                except np.linalg.LinAlgError:
                    ApprCov = ApprCov + eps * np.eye(d)
                    CholAppr = np.linalg.cholesky(ApprCov)
                InvAppr = np.linalg.inv(ApprCov)

            # sample N indices
            cdf = np.cumsum(P)
            idx = np.searchsorted(cdf, U[:, d])
            x_new = Proposals[idx]
            self.xVals.append(x_new.copy())
            self.AcceptVals.append(1.0 - P[idx])

            xI = Proposals[idx[-1]]

    def GetSamples(self, BurnIn=0):
        return np.concatenate(self.xVals[1:], axis=0)[BurnIn:, :]

    def GetAcceptRate(self, BurnIn=0):
        return np.concatenate(self.AcceptVals)[BurnIn:].mean()

    def Get_MeanEstimate(self, N, BurnIn=0):
        return self.WeightedSum[int(BurnIn/N):].mean(axis=0)

    def Get_CovEstimate(self, N, BurnIn=0):
        return self.WeightedCov[int(BurnIn/N):].mean(axis=0)

    def GetDiagnostics(self):
        # tolerate older objects that might not have these attrs
        maxw = getattr(self, "max_weight_history", [])
        H = getattr(self, "entropy_history", [])
        return {"maxw": np.asarray(maxw), "entropy": np.asarray(H)}

