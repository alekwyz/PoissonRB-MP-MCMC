#paper_metrics_poisson.py
import numpy as np
import matplotlib.pyplot as plt

from rb_poisson import RB_BayesianPoissonReg
from arb_poisson import ARB_BayesianPoissonReg
from mh_sampler import RW_BayesianPoissonReg
from helpers_irls import poisson_map_irls

def running_mean(x):  # x: (T,d)
    cs = np.cumsum(x, axis=0)
    t = np.arange(1, x.shape[0] + 1)[:, None]
    return cs / t

def empirical_variance_over_reps(theta_hats):
    var_dim = np.var(theta_hats, axis=0, ddof=1)   # (T,d)
    var_scalar = var_dim.mean(axis=1)              # (T,)

    var_per_rep = ((theta_hats - theta_hats.mean(axis=0))**2).mean(axis=2)  # (R,T)

    R = theta_hats.shape[0]
    # <-- KEY CHANGE: divide by sqrt(R)
    sd_scalar = var_per_rep.std(axis=0, ddof=1) / np.sqrt(R)

    return var_scalar, sd_scalar



def run_parallel_mh(N_chains, L, burnin, step, x0, XX, y, alpha, seed0):
    """
    Paper-style baseline: N independent chains, keep L samples after burn-in,
    total cost n = L*N likelihood evals (post burn-in).
    Returns series of posterior mean estimates as function of ℓ=1..L:
    theta_hat(ℓ) = average over all chains and first ℓ post-burnin samples.
    """
    d = x0.shape[0]
    chain_means = np.zeros((N_chains, L, d))
    for j in range(N_chains):
        np.random.seed(seed0 + 10_000*j)   # independence across chains
        mh = RW_BayesianPoissonReg(
            N_iter=burnin + L, StepSize=step, x0=x0, XX=XX, y=y, alpha=alpha
        )
        samps = mh.GetSamples(BurnIn=burnin)[:L]  # (L,d)
        chain_means[j] = running_mean(samps)      # (L,d)
    # pool chains: average chain running means (equiv to pooling samples)
    return chain_means.mean(axis=0)               # (L,d)

def paper_style_experiment(XXtr, ytr, alpha=100.0,
                           N=20, StepSize=1.0, PowerOfTwo=14,
                           burnin_samples=5000,
                           R=25,
                           mh_stepsize=0.01):
    """
    Returns empirical variance curves for RB, ARB, and MH (parallel baseline),
    using paper-style replication across R independent runs.
    """
    # Initialize proposal from Laplace warm start (fixed across reps)
    InitMean, InitCov = poisson_map_irls(XXtr, ytr, alpha=alpha)
    x0 = InitMean.copy()

    d = x0.shape[0]
    # RB/ARB iterations implied by your seed schedule:
    L = int(int((2**PowerOfTwo - 1) / (d + 1)) * (d + 1) / N)

    # Convert burn-in in samples to burn-in in iterations for RB/ARB series
    burnin_iter = int(np.ceil(burnin_samples / N))
    L_post = L - burnin_iter
    if L_post <= 10:
        raise ValueError("Burn-in too large relative to L; reduce burnin_samples or increase PowerOfTwo.")

    rb_hats  = np.zeros((R, L_post, d))
    arb_hats = np.zeros((R, L_post, d))
    mh_hats  = np.zeros((R, L_post, d))

    for r in range(R):
        # control randomness for fair replicate comparisons
        np.random.seed(1000 + r)

        rb = RB_BayesianPoissonReg(
            N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo, x0=x0,
            InitMean=InitMean, InitCov=InitCov,
            XX=XXtr, y=ytr, alpha=alpha, Stream="iid"
        )
        rb_iter = rb.Get_RBIterMeans()[burnin_iter:]         # (L_post,d)
        rb_hats[r] = running_mean(rb_iter)

        np.random.seed(2000 + r)
        arb = ARB_BayesianPoissonReg(
            N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo,
            InitMean=InitMean, InitCov=InitCov,
            XX=XXtr, y=ytr, alpha=alpha, Stream="iid", WeightIn=0
        )
        arb_iter = arb.Get_RBIterMeans()[burnin_iter:]
        arb_hats[r] = running_mean(arb_iter)

        # MH baseline: N independent chains, L_post samples after burn-in
        mh_hats[r] = run_parallel_mh(
            N_chains=N, L=L_post, burnin=burnin_samples,
            step=mh_stepsize, x0=x0, XX=XXtr, y=ytr, alpha=alpha,
            seed0=3000 + r
        )

    rb_var,  rb_sd  = empirical_variance_over_reps(rb_hats)
    arb_var, arb_sd = empirical_variance_over_reps(arb_hats)
    mh_var,  mh_sd  = empirical_variance_over_reps(mh_hats)

    # Cost axis after burn-in: n = (ℓ * N) likelihood evaluations
    n_cost = N * np.arange(1, L_post + 1)

    out = {
        "n_cost": n_cost,
        "RB":  (rb_var, rb_sd),
        "ARB": (arb_var, arb_sd),
        "MH":  (mh_var, mh_sd),
        "L_post": L_post
    }
    return out

def plot_paper_style(out):
    n = out["n_cost"]
    plt.figure(figsize=(9,6))

    for name in ["MH", "RB", "ARB"]:
        mean, sd = out[name]
        plt.loglog(n, mean, label=name)
        # paper uses ±3 sd bands across 25 simulations :contentReference[oaicite:9]{index=9}
        plt.fill_between(n, np.maximum(mean - 3*sd, 1e-16), mean + 3*sd, alpha=0.15)

    plt.xlabel("Number of likelihood evaluations $n = LN$")
    plt.ylabel("Empirical variance of posterior mean estimator")
    plt.title("Empirical variance vs cost")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def variance_reduction_factors(out):
    # Table-2 style: factor = Var(MH) / Var(method) at final cost :contentReference[oaicite:10]{index=10}
    mh_final = out["MH"][0][-1]
    rb_final = out["RB"][0][-1]
    arb_final = out["ARB"][0][-1]
    return {
        "RB_vs_MH": mh_final / rb_final,
        "ARB_vs_MH": mh_final / arb_final
    }


def plot_fig4_style(out, methods=("MH", "RB", "ARB")):
    """
    Poisson analogue of Schwedes & Calderhead Fig. 4:

    - One panel: empirical variance vs number of likelihood evaluations (log-log)
    - Curves for MH, RB-MP-MCMC, ARB-MP-MCMC
    - Error bars at a handful of cost points, showing ± 3 SD across 25 runs
    - A reference n^{-1} line
    """

    n_cost = out["n_cost"]   # (T,)
    L_post = out["L_post"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # choose ~7 log-spaced indices for error bars, skipping very early noisy points
    idx = np.unique(
        np.linspace(L_post // 5, L_post - 1, 7, dtype=int)
    )

    # style choices roughly mimicking Fig. 4
    style = {
        "MH":  {"color": "C0", "ls": "--", "marker": "o"},
        "RB":  {"color": "C2", "ls": "-",  "marker": "s"},
        "ARB": {"color": "C3", "ls": "-.", "marker": "^"},
    }

    # --------------------------
    # variance vs cost curves
    # --------------------------
    for name in methods:
        mean, sd = out[name]

        st = style.get(name, {"color": None, "ls": "-", "marker": "o"})

        # continuous line
        ax.loglog(n_cost, mean,
                  label=name,
                  color=st["color"],
                  linestyle=st["ls"],
                  linewidth=1.8)

        # error bars (± 3 SD) at selected points
        ax.errorbar(
            n_cost[idx], mean[idx],
            yerr=3 * sd[idx],
            fmt=st["marker"],
            markersize=4,
            capsize=3,
            linestyle="none",
            color=st["color"],
            alpha=0.9,
        )

    # --------------------------
    # n^{-1} reference line
    # --------------------------
    # anchor at first RB point in the error-bar region
    ref_name = "RB" if "RB" in methods else methods[0]
    ref_mean, _ = out[ref_name]
    k0 = idx[0]
    n0 = n_cost[k0]
    v0 = ref_mean[k0]

    n_ref = np.array([n0, n_cost[-1]])
    y_ref = v0 * (n0 / n_ref)  # proportional to 1/n

    ax.loglog(n_ref, y_ref, "k-", linewidth=1.2, label=r"$n^{-1}$")

    # === CROP REGION HERE ===
    xmin = n_cost[idx[0]] * 0.9  # a bit to the left of first error bar
    xmax = n_cost[idx[-1]] * 1.1  # a bit to the right of last one
    ax.set_xlim(xmin, xmax)

    # axes / labels
    ax.set_xlabel(r"Number of likelihood evaluations $n = LN$")
    ax.set_ylabel("Variance")
    ax.set_title("Empirical variance estimates (Poisson MP–MCMC)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
