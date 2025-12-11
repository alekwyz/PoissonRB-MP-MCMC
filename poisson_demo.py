# poisson_demo.py
import numpy as np
import visualize_results
from poisson_data import PoissonSimConfig, PoissonDataLoad
from rb_poisson import RB_BayesianPoissonReg
from arb_poisson import ARB_BayesianPoissonReg
from typing import Optional
from helpers_irls import poisson_map_irls
from mh_sampler import RW_BayesianPoissonReg
from paper_metrics_poisson import paper_style_experiment, plot_paper_style, variance_reduction_factors
from mh_tuning import tune_mh_stepsize
from paper_metrics_poisson import plot_fig4_style



np.random.seed(5331)
# data
cfg = PoissonSimConfig(n=1200, D=10, mean_rate=3.0, eta_sd=0.7, seed=7)
dat = PoissonDataLoad(cfg)
XX = dat.GetDesignMatrix()
y  = dat.GetResponses()

n, d = XX.shape
perm = np.random.default_rng(0).permutation(n)
train, test = perm[:n//2], perm[n//2:]
XXtr, ytr = XX[train], y[train]
XXte, yte = XX[test],  y[test]

# Prior & warm start proposal
alpha = 100.0
InitMean, InitCov = poisson_map_irls(XXtr, ytr, alpha=alpha)
x0 = InitMean.copy()

# MP-MCMC parameters
N          = 20
StepSize   = 1.0
PowerOfTwo = 14
BurnIn     = 5000

# Tune MH step size (paper tunes RW-MH acceptance ~0.20–0.25)
tuned = tune_mh_stepsize(
    XX=XXtr, y=ytr, x0=x0, alpha=alpha,
    target=0.23,
    grid=np.logspace(-3, -0.5, 12),
    iters=3000,
    seed=123
)
MH_StepSize = tuned["StepSize"]
print("Tuned MH step size:", MH_StepSize, "pilot accept:", tuned["AcceptRate"])

# PAPER-STYLE EXPERIMENT
#  - 25 independent MCMC simulations
#  - cost axis n = L*N likelihood evaluations (post burn-in)
#  - empirical variance of posterior mean estimator vs n
#  - variance reduction factors
out = paper_style_experiment(
    XXtr, ytr,
    alpha=alpha,
    N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo,
    burnin_samples=BurnIn,
    R=25,
    mh_stepsize=MH_StepSize
)

plot_fig4_style(out)
print("Variance reduction factors:", variance_reduction_factors(out))



# one diagnostic run for traces/predictive checks (Appendix)
# Running RW-MH Baseline for {MH_Budget} iterations
L = int(int((2**PowerOfTwo - 1) / (d + 1)) * (d + 1) / N)
print(f"[Diag] RB/ARB iterations L = {L}")
MH_Budget = L * N

# Run tuned RW-MH (single long chain)
mh = RW_BayesianPoissonReg(
    N_iter=MH_Budget,
    StepSize=MH_StepSize,
    x0=x0,
    XX=XXtr,
    y=ytr,
    alpha=alpha
)

# --- Run RB ---
rb = RB_BayesianPoissonReg(
    N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo, x0=x0,
    InitMean=InitMean, InitCov=InitCov,
    XX=XXtr, y=ytr, alpha=alpha, Stream="iid"
)
rb_samps = rb.GetSamples(BurnIn=BurnIn)
rb_mean  = rb.Get_MeanEstimate(N, BurnIn=BurnIn)
rb_sel_diag = rb.GetAcceptRate(BurnIn=BurnIn)  # weight-selection diagnostic (NOT MH accept)

# --- Run ARB ---
arb = ARB_BayesianPoissonReg(
    N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo,
    InitMean=InitMean, InitCov=InitCov,
    XX=XXtr, y=ytr, alpha=alpha, Stream="iid", WeightIn=0
)
arb_samps = arb.GetSamples(BurnIn=BurnIn)
arb_mean  = arb.Get_MeanEstimate(N, BurnIn=BurnIn)
arb_sel_diag = arb.GetAcceptRate(BurnIn=BurnIn)  # weight-selection diagnostic (NOT MH accept)

print("\n[Diag] Weight-selection diagnostics (1 - selected weight):")
print("RB  mean(1 - selected weight): ", rb_sel_diag)
print("ARB mean(1 - selected weight): ", arb_sel_diag)

print("\n[Diag] Posterior mean (first 5 coords):")
print("RB :", np.round(rb_mean[:5], 3))
print("ARB:", np.round(arb_mean[:5], 3))


# --- Predictive sanity checks (Appendix only) ---
def mean_loglik(XX, y, theta):
    eta = XX @ theta
    eta = np.clip(eta, -20, 20)
    return (y * eta - np.exp(eta)).mean()

print("\n[Diag] Plug-in mean log-likelihood:")
print("RB  train:", mean_loglik(XXtr, ytr, rb_mean))
print("ARB train:", mean_loglik(XXtr, ytr, arb_mean))
print("RB  test :", mean_loglik(XXte, yte, rb_mean))
print("ARB test :", mean_loglik(XXte, yte, arb_mean))

lam_rb  = np.exp(np.clip(XXte @ rb_mean, -20, 20))
lam_arb = np.exp(np.clip(XXte @ arb_mean, -20, 20))
yrep_rb  = np.random.default_rng(1).poisson(lam_rb)
yrep_arb = np.random.default_rng(2).poisson(lam_arb)
print("\n[Diag] Posterior predictive mean check (test):")
print("Observed mean:", yte.mean())
print("RB  rep mean :", yrep_rb.mean())
print("ARB rep mean :", yrep_arb.mean())

# ============================================================
# EXTRA VISUALS REQUESTED BY FEEDBACK:
# trace plots + convergence + evolution of ARB proposal mean/cov
# ============================================================

import matplotlib.pyplot as plt

def trace_plot(samples_dict, idx_list=(0, 1, 2), thin=10, burn=0):
    plt.figure(figsize=(10, 6))
    for idx in idx_list:
        plt.clf()
        for name, s in samples_dict.items():
            ss = s[burn::thin, idx]
            plt.plot(ss, label=name, alpha=0.7, linewidth=1)
        plt.title(f"Trace plot for coefficient {idx}")
        plt.xlabel("Iteration (thinned)")
        plt.ylabel(f"beta[{idx}]")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Trace plots: compare MH vs RB vs ARB
chains = {
    "RW-MH": mh.GetSamples(BurnIn=BurnIn),
    "RB": rb_samps,
    "ARB": arb_samps,
}
trace_plot(chains, idx_list=(0, 1, 2), thin=10, burn=0)

# --- ARB adaptation diagnostics: proposal mean/cov evolution ---
# We reconstruct the proposal mean/cov sequence from ARB's stored arrays.
# ARB.WeightedSum[t] contains the RB estimate at iteration t (post init block).
# The running average of WeightedSum is exactly what ARB uses as ApprMean.
M = getattr(arb, "M", 1)  # we strongly recommend setting self.M in ARB __init__
Wsum = arb.WeightedSum[M:, :]        # drop initialization block
Wcov = arb.WeightedCov[M:, :, :]     # same

# running proposal centers mu_t
tgrid = np.arange(1, Wsum.shape[0] + 1)[:, None]
mu_t = np.cumsum(Wsum, axis=0) / tgrid

# reference = final center (proxy for limit)
mu_ref = mu_t[-1]
mu_err = np.linalg.norm(mu_t - mu_ref, axis=1)

plt.figure(figsize=(9, 5))
plt.semilogy(mu_err + 1e-16)
plt.title("ARB proposal mean evolution: ||mu_t - mu_final||")
plt.xlabel("Iteration")
plt.ylabel("Distance (log scale)")
plt.tight_layout()
plt.show()

# proposal covariance evolution: compare Frobenius distance to final
Sigma_t = np.cumsum(Wcov, axis=0) / np.arange(1, Wcov.shape[0] + 1)[:, None, None]
Sigma_ref = Sigma_t[-1]
Sigma_err = np.array([np.linalg.norm(Sigma_t[k] - Sigma_ref, ord="fro") for k in range(Sigma_t.shape[0])])

plt.figure(figsize=(9, 5))
plt.semilogy(Sigma_err + 1e-16)
plt.title("ARB proposal covariance evolution: ||Sigma_t - Sigma_final||_F")
plt.xlabel("Iteration")
plt.ylabel("Frobenius distance (log scale)")
plt.tight_layout()
plt.show()

visualize_results.main(
    rb_sampler=rb,
    arb_sampler=arb,
    rb_fixed_sampler=None,
    true_beta=dat.GetTrueBeta(),
    xx_test=XXte,
    y_test=yte,
    N_proposals=N,
    mh_sampler=mh
)