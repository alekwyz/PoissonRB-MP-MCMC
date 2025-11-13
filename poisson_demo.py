
import numpy as np
from poisson_data import PoissonSimConfig, PoissonDataLoad
from rb_poisson import RB_BayesianPoissonReg
from arb_poisson import ARB_BayesianPoissonReg

# data
cfg = PoissonSimConfig(n=1200, D=10, mean_rate=3.0, eta_sd=0.7, seed=7)
dat = PoissonDataLoad(cfg)
XX = dat.GetDesignMatrix(); y = dat.GetResponses()
n, d = XX.shape

# train / test split (like paper’s train/test comparisons)
perm = np.random.default_rng(0).permutation(n)
train, test = perm[:n//2], perm[n//2:]
XXtr, ytr = XX[train], y[train]
XXte, yte = XX[test],  y[test]

# prior & proposals
alpha     = 100.0
InitMean  = np.zeros(d)
InitCov   = np.eye(d)
x0        = InitMean.copy()

# MP-MCMC params (pick similar magnitude as your logistic runs)
N         = 20
StepSize  = 1.0
PowerOfTwo= 14
BurnIn    = 5000

# run RB (fixed proposal)
rb = RB_BayesianPoissonReg(
    N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo, x0=x0,
    InitMean=InitMean, InitCov=InitCov,
    XX=XXtr, y=ytr, alpha=alpha, Stream="iid"
)
rb_samps = rb.GetSamples(BurnIn=BurnIn)
rb_mean  = rb.Get_MeanEstimate(N, BurnIn=BurnIn)
rb_acc   = rb.GetAcceptRate(BurnIn=BurnIn)

# run ARB (adaptive proposal)
arb = ARB_BayesianPoissonReg(
    N=N, StepSize=StepSize, PowerOfTwo=PowerOfTwo,
    InitMean=InitMean, InitCov=InitCov,
    XX=XXtr, y=ytr, alpha=alpha, Stream="iid", WeightIn=0
)
arb_samps = arb.GetSamples(BurnIn=BurnIn)
arb_mean  = arb.Get_MeanEstimate(N, BurnIn=BurnIn)
arb_acc   = arb.GetAcceptRate(BurnIn=BurnIn)

# outputs

def mean_loglik(XX, y, theta):
    eta = XX @ theta
    return (y * eta - np.exp(eta)).mean()  # drop log(y!) const for comparison

print("RB accept rate:",  rb_acc)
print("ARB accept rate:", arb_acc)
print("RB posterior mean (first 5):",  np.round(rb_mean[:5], 3))
print("ARB posterior mean (first 5):", np.round(arb_mean[:5], 3))

# train/test average log-likelihood (posterior mean plug-in)
print("RB  train mean loglik:", mean_loglik(XXtr, ytr, rb_mean))
print("ARB train mean loglik:", mean_loglik(XXtr, ytr, arb_mean))
print("RB  test  mean loglik:", mean_loglik(XXte, yte, rb_mean))
print("ARB test  mean loglik:", mean_loglik(XXte, yte, arb_mean))

# posterior predictive check at the mean:
lam_rb  = np.exp(XXte @ rb_mean);  yrep_rb  = np.random.default_rng(1).poisson(lam_rb)
lam_arb = np.exp(XXte @ arb_mean); yrep_arb = np.random.default_rng(2).poisson(lam_arb)
print("PP check (means)  RB:",  yte.mean(), yrep_rb.mean())
print("PP check (means) ARB:",  yte.mean(), yrep_arb.mean())
