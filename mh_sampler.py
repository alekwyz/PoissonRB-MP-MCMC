import numpy as np


class RW_BayesianPoissonReg:
    def __init__(self, N_iter, StepSize, x0, XX, y, alpha=100.0):
        """
        Standard Random Walk Metropolis-Hastings (Single Chain).

        Parameters:
        -----------
        N_iter : int
            Total number of iterations.
            For a fair comparison with RB-MP-MCMC (L iters, N proposals),
            you should set N_iter = L * N.
        StepSize : float
            Standard deviation of the Gaussian random walk proposal.
        x0 : array
            Initial starting point.
        XX : array
            Design matrix.
        y : array
            Response vector.
        alpha : float
            Prior variance scalar (beta ~ N(0, alpha*I)).
        """
        self.XX = XX
        self.y = y
        self.alpha = alpha
        self.StepSize = StepSize
        self.d = XX.shape[1]

        # Storage
        self.xVals = []
        self.xVals.append(x0.copy())
        self.AcceptCount = 0

        # Current State Tracking
        self.curr_x = x0.copy()
        self.curr_log_post = self._log_posterior(self.curr_x)

        # --- Run Chain ---
        # In a real loop, you might want progress bars, but we keep it simple here
        for _ in range(N_iter):
            self._step()

    def _log_posterior(self, theta):
        """
        Calculates unnormalized Log Posterior: Log Prior + Log Likelihood
        """
        # 1. Log Prior: -0.5 * theta^T * (1/alpha * I) * theta
        log_prior = -0.5 * np.sum(theta ** 2) / self.alpha

        # 2. Log Likelihood: y * eta - exp(eta)
        # eta = X * theta
        eta = self.XX @ theta

        # Clip eta for numerical stability (matching your project's data gen constraints)
        # This prevents exp(eta) from overflowing
        eta = np.clip(eta, -20, 20)

        # Poisson log-likelihood (ignoring log(y!) constant)
        lam = np.exp(eta)
        log_lik = np.sum(self.y * eta - lam)

        return log_prior + log_lik

    def _step(self):
        """
        Performs one Metropolis-Hastings step
        """
        # 1. Propose: Random Walk (Symmetric Gaussian)
        # x' = x + epsilon, where epsilon ~ N(0, StepSize^2)
        proposal = self.curr_x + np.random.normal(0, self.StepSize, size=self.d)

        # 2. Calculate Acceptance Probability
        prop_log_post = self._log_posterior(proposal)

        # Log Ratio = log( P(x') / P(x) ) + log( q(x|x') / q(x'|x) )
        # Since proposal is symmetric Gaussian, the q terms cancel out.
        log_alpha = prop_log_post - self.curr_log_post

        # 3. Accept/Reject
        # Accept if log(u) < log_alpha
        if np.log(np.random.rand()) < log_alpha:
            self.curr_x = proposal
            self.curr_log_post = prop_log_post
            self.AcceptCount += 1

        # 4. Store sample (we store the current state regardless of acceptance)
        self.xVals.append(self.curr_x.copy())

    def GetSamples(self, BurnIn=0):
        """
        Returns the chain samples after removing burn-in.
        """
        return np.array(self.xVals)[BurnIn:, :]

    def GetAcceptRate(self, BurnIn=0):
        """
        Returns the global acceptance rate.
        (BurnIn argument kept for API compatibility, though usually we calc global rate)
        """
        if len(self.xVals) <= 1:
            return 0.0
        return self.AcceptCount / (len(self.xVals) - 1)

    def Get_MeanEstimate(self, N=1, BurnIn=0):
        """
        Returns the posterior mean estimate.

        Args:
            N: Ignored (kept for compatibility with RB-MP-MCMC API which uses N proposals).
            BurnIn: Number of initial samples to discard.
        """
        samples = self.GetSamples(BurnIn)
        return np.mean(samples, axis=0)

    def GetDiagnostics(self):
        """
        Returns empty diagnostics for compatibility with visualize_results.py
        (MH doesn't have 'entropy' or 'max_weight' like RB methods do).
        """
        return {"maxw": [], "entropy": []}