# visualize_results.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

# Set a style similar to the paper snippet you provided
plt.style.use('seaborn-v0_8-whitegrid')


def calculate_mse_series(samples, true_param):
    """
    Calculates the Mean Squared Error (MSE) of the running posterior mean
    against the true parameter (or a high-quality reference).

    Returns: Array of MSE values over iterations.
    """
    # 1. Calculate cumulative sum of samples
    cumsum = np.cumsum(samples, axis=0)

    # 2. Divide by step number to get running mean: (1, 2, ..., T)
    steps = np.arange(1, samples.shape[0] + 1).reshape(-1, 1)
    running_means = cumsum / steps

    # 3. Calculate Squared Error for each step: ||theta_hat - theta_true||^2
    # We average across dimensions (D)
    squared_errors = np.mean((running_means - true_param) ** 2, axis=1)

    return squared_errors


def plot_trace_comparison(chains, param_indices=[0, 1], truth=None):
    """
    Plots trace plots for specific dimensions to visualize mixing.
    """
    fig, axes = plt.subplots(len(param_indices), 1, figsize=(10, 6), sharex=True)
    if len(param_indices) == 1: axes = [axes]

    for i, ax in zip(param_indices, axes):
        for name, samples in chains.items():
            # Thinning for visual clarity if chain is huge
            ax.plot(samples[:, i], label=name, alpha=0.7, linewidth=1)

        if truth is not None:
            ax.axhline(truth[i], color='black', linestyle='--', label='True Beta')

        ax.set_ylabel(f'Beta_{i}')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('MCMC Iterations')
    fig.suptitle('Trace Plots: Visualizing Mixing Efficiency')
    plt.tight_layout()
    plt.show()


def plot_adaptation_diagnostics(arb_sampler, true_mean=None):
    """
    Professor's Request: Visualize evolution of proposal mean/covariance.

    This reconstructs the history of the RB-Weighted Mean (which drives the adaptation)
    to show how it converges toward the true posterior mode/mean.
    """
    # Reconstruct the running weighted mean from the sampler history
    # The ARB class stores 'WeightedSum'. We need to turn this into a mean.
    # Note: WeightedSum[n] is the sum up to iter n.

    history_sum = arb_sampler.WeightedSum
    # Avoid division by zero for the 0-th index if initialized empty
    steps = np.arange(1, history_sum.shape[0] + 1).reshape(-1, 1)

    # This is the sequence of "Centers" the adaptive proposal used
    proposal_centers = history_sum / steps

    # Calculate distance of Proposal Center from Truth (or final estimate)
    if true_mean is None:
        # Use the final estimate as proxy for truth if truth not provided
        reference = proposal_centers[-1]
    else:
        reference = true_mean

    # Euclidean distance || mu_t - mu_true ||
    diff_norm = np.linalg.norm(proposal_centers - reference, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff_norm, color='purple', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$|| \mu_{proposal}^{(t)} - \mu_{true} ||$ (Log Scale)')
    ax.set_xlabel('Iterations')
    ax.set_title('Evolution of Proposal Mean Accuracy\n(Why Adaptation Works)')

    # Add annotation explaining the plot
    text = ("Decreasing distance indicates the\n"
            "proposal is centering on the\n"
            "true posterior mass.")
    ax.text(0.7, 0.8, text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_mse_convergence_paper_style(chains, true_param, likelihood_evals_per_iter):
    """
    Replicates the Log-Log MSE plot from the Schwedes & Calderhead paper.

    likelihood_evals_per_iter: Dict mapping algorithm name to cost per step.
       RB/ARB: cost = N (number of proposals)
       RW-MH: cost = 1
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'RB (Fixed)': 'red', 'ARB (Adaptive)': 'blue', 'RW-MH': 'gray', 'Warm Start': 'green'}
    markers = {'RB (Fixed)': 'x', 'ARB (Adaptive)': 'o', 'RW-MH': '.', 'Warm Start': 's'}

    for name, samples in chains.items():
        # Calculate MSE history
        mse_series = calculate_mse_series(samples, true_param)

        # Calculate X-axis (Number of Likelihood Evaluations)
        cost = likelihood_evals_per_iter.get(name, 1)
        x_axis = np.arange(1, len(mse_series) + 1) * cost

        # Plot Log-Log
        ax.loglog(x_axis, mse_series, label=name, color=colors.get(name, 'black'),
                  marker=markers.get(name, ''), markevery=0.1, linewidth=1.5)

    # Add the theoretical 1/n reference line (Standard MCMC rate)
    # We anchor it to the last point of the best chain
    x_ref = np.array([x_axis[0], x_axis[-1]])
    # Simple heuristic to place the line
    y_ref = mse_series[0] * (x_ref[0] / x_ref) * 0.1  # scaled down to sit below
    # ax.loglog(x_ref, y_ref, 'k--', label=r'O($n^{-1}$)')

    ax.set_xlabel('Number of Likelihood Evaluations (Cost)')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Convergence of Estimators')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_improved_summary_table(samplers_dict, xx_test, y_test):
    """
    Generates a readable pandas DataFrame with explicit definitions.
    """
    results = []

    for name, sampler in samplers_dict.items():
        # 1. Acceptance Rate
        # Note: ARB/RB store full history, calculate mean
        acc_rate = sampler.GetAcceptRate()

        # 2. Diversity (Entropy) - The professor wants to see this clearly
        # We calculate the mean entropy of the weights over the run
        # Retrieve diagnostics if available
        diag = sampler.GetDiagnostics()
        if len(diag['entropy']) > 0:
            avg_entropy = np.mean(diag['entropy'])
            # Normalized entropy (0 to 1) is easier to interpret: H / log(N)
            # Assuming N is stored in sampler
            N_proposals = getattr(sampler, 'N', 1)  # Default to 1 if MH
            norm_diversity = avg_entropy / np.log(N_proposals) if N_proposals > 1 else 1.0
        else:
            avg_entropy = np.nan
            norm_diversity = np.nan

        # 3. Posterior Predictive Check (Test Set)
        # Mean estimate of Beta
        beta_hat = sampler.Get_MeanEstimate(N=1)  # N=1 effectively gets mean of chain
        # Predicted Lambda
        eta = xx_test @ beta_hat
        lam = np.exp(np.clip(eta, -20, 20))
        pp_mean_y = np.mean(lam)

        # 4. Log Likelihood (Test)
        # sum (y*eta - exp(eta))
        log_lik_test = np.mean(y_test * eta - lam)

        results.append({
            "Method": name,
            "Accept Rate": f"{acc_rate:.3f}",
            "Diversity (Norm Entropy)": f"{norm_diversity:.3f}",
            "Test Log-Likelihood": f"{log_lik_test:.3f}",
            "PP Mean (Test)": f"{pp_mean_y:.3f}"
        })

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("IMPROVED RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    print("DEFINITIONS:")
    print(
        "1. Diversity (Norm Entropy): Measures weight collapse. 0.0 = All weight on 1 sample (Bad). 1.0 = Equal weights (Ideal).")
    print("2. Test Log-Likelihood: Higher is better. Measures fit on unseen data.")
    print("3. PP Mean: Posterior Predictive Mean count. Should match empirical Test Mean.")
    print("=" * 60 + "\n")


# ==========================================================
# MAIN EXECUTION HELPER
# 要加新的chain在这里加详见以下abc 3步
# ==========================================================
def main(rb_sampler, arb_sampler, rb_fixed_sampler, true_beta, xx_test, y_test, N_proposals, mh_sampler=None):
    """
    Call this function from your poisson_demo.py script.

    Args:
        rb_sampler: The 'Warm Start' RB object
        arb_sampler: The 'Adaptive' ARB object
        rb_fixed_sampler: The 'Naive' RB object (if you ran it)
        true_beta: The true coefficients used to generate data
        xx_test, y_test: Test data for PP checks
        N_proposals: The 'N' used in MP-MCMC
        mh_sampler: (Optional) The RW-MH object if you have run it
    """

    # 1. Aggregate Chains for Plotting
    # We assume GetSamples() returns the (Samples x D) array
    chains = {
        'ARB (Adaptive)': arb_sampler.GetSamples(BurnIn=0),
        'Warm Start RB': rb_sampler.GetSamples(BurnIn=0),
        # a：mh_sampler.GetSamples(BurnIn=0),
    }

    samplers_obj = {
        'ARB (Adaptive)': arb_sampler,
        'Warm Start RB': rb_sampler
        # b:  mh_sampler
    }

    if rb_fixed_sampler is not None:
        chains['RB (Fixed)'] = rb_fixed_sampler.GetSamples(BurnIn=0)
        samplers_obj['RB (Fixed)'] = rb_fixed_sampler

    # ==========================================================
    # FUTURE WORK: RW-MH INTEGRATION
    # When you finish the RW-MH code I gave you, pass the object here.
    # ==========================================================
    if mh_sampler is not None:
        chains['RW-MH'] = mh_sampler.GetSamples(BurnIn=0)
        samplers_obj['RW-MH'] = mh_sampler
        mh_cost = 1
    else:
        mh_cost = 1

    # 2. Generate Trace Plots
    # Plotting index 0 (Intercept) and index 1 (First Coeff)
    print("Generating Trace Plots...")
    plot_trace_comparison(chains, param_indices=[0, 1], truth=true_beta)

    # 3. Generate Adaptation Diagnostics
    # Only for ARB
    print("Generating Adaptation Evolution Plot...")
    plot_adaptation_diagnostics(arb_sampler, true_mean=true_beta)

    # 4. Generate MSE Convergence (Paper Style)
    print("Generating MSE Convergence Plot...")
    # Define cost: RB/ARB cost N evals per sample. MH costs 1.
    costs = {
        'ARB (Adaptive)': N_proposals,
        'Warm Start RB': N_proposals,
        'RB (Fixed)': N_proposals,
        'RW-MH': 1
        # c: 'RW-MH Baseline': 1,  # Cost is 1 per step
        # 'My New Sampler': 1  # If single chain
    }
    plot_mse_convergence_paper_style(chains, true_beta, costs)

    # 5. Generate Table
    generate_improved_summary_table(samplers_obj, xx_test, y_test)


if __name__ == "__main__":
    print("This is a module. Import it in poisson_demo.py or run that file instead.")
