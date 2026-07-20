"""
02a_snpe_inference.py
=====================

ALTERNATIVE STEP To Direct Ranking (Path B, part 1 of 2).

After this, we can go to predictive top n code to find the top viable samples


------------
Trains a normalizing flow (SNPE) on the feasibility simulation cloud to
approximate p(theta | features). Conditions on the WT cohort's mean
feature vector and samples 10,000 parameter sets from the posterior.

Compared to direct ranking (Path A):
  - Uses ALL viable sims as training data, not just the closest
  - Learns continuous parameter distributions and joint correlations
  - Gives proper credible intervals and constraint strengths
  - Can collapse onto a wrong corner if the model can't match cohort

Use this if the model fits the cohort well. If the feasibility check
shows large mismatches, prefer direct ranking (02_direct_ranking.py).

Dependencies
------------
Requires `sbi` and `torch`:
    pip install sbi

Output
------
  SNPE_posterior_<COHORT>.csv      : 10k posterior samples (linear scale)
  SNPE_marginals_<COHORT>.png      : prior vs posterior per parameter
  SNPE_pairwise_<COHORT>.png       : joint posterior (hexbin)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
except ImportError:
    raise SystemExit("Missing dependency. Install with:  pip install sbi")

# =============================================================================
# CONFIG
# =============================================================================
SAVE_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
# COHORT_FILTER = 'WT'
COHORT_FILTER = 'KO'


# Must match the priors used in 01_feasibility_check.py
PRIOR = {
    'gNa':   ('log', 30.0,  450.0),
    'gKv1':  ('log', 0.5,   50.0),
    'gKv3':  ('log', 20.0,  300.0),
    'gCa':   ('log', 0.5,   50.0),
    'gSK':   ('log', 0.1,   50.0),
    'Btot':  ('lin', 10.0,  120.0),
    'gleak': ('lin', 0.5,   5.0),
}
PARAM_COLS = list(PRIOR.keys())
PARAM_LOG  = {p for p, (s, _, _) in PRIOR.items() if s == 'log'}

# Features that SNPE conditions on. Same as direct_ranking's distance set
# (no tau_f_s — model rarely produces a clean exponential decay).
FEATURES = [
    'spike_count', 'adapt_idx', 'discharge_time_s',
    'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms',
]

N_POSTERIOR_SAMPLES = 10000


sim_df  = pd.read_csv(SAVE_DIR / "feasibility_sims_ko.csv")
real_df = pd.read_csv(SAVE_DIR / f"feasibility_real_{COHORT_FILTER.lower()}.csv")
print(f"Loaded {len(sim_df)} sims and {len(real_df)} real {COHORT_FILTER} cells.")

# Keep only viable sims with all features defined
sim = sim_df.dropna(subset=['spike_count']).copy()
sim = sim[sim['spike_count'] >= 2]
sim = sim.dropna(subset=FEATURES + PARAM_COLS).reset_index(drop=True)
print(f"Viable sims with all features defined: {len(sim)}")


# Log-priors were sampled in log space, so the network should also see them
# in log space. We convert back to linear after sampling the posterior.
theta_cols = []
for p in PARAM_COLS:
    v = sim[p].values.astype(np.float32)
    if p in PARAM_LOG:
        v = np.log10(v)
    theta_cols.append(v)
theta = torch.tensor(np.column_stack(theta_cols), dtype=torch.float32)
x     = torch.tensor(sim[FEATURES].values, dtype=torch.float32)

print(f"theta shape: {tuple(theta.shape)}, x shape: {tuple(x.shape)}")


lows, highs = [], []
for p in PARAM_COLS:
    scale, lo, hi = PRIOR[p]
    if p in PARAM_LOG:
        lo, hi = np.log10(lo), np.log10(hi)
    lows.append(lo); highs.append(hi)
prior = BoxUniform(low=torch.tensor(lows,  dtype=torch.float32),
                   high=torch.tensor(highs, dtype=torch.float32))

print("\n🧠 Training SNPE on the feasibility cloud...")
inference = SNPE(prior=prior)
inference = inference.append_simulations(theta, x)
density_estimator = inference.train(show_train_summary=True)
posterior = inference.build_posterior(density_estimator)
print("✅ Training complete.")


x_obs = torch.tensor(real_df[FEATURES].mean().values, dtype=torch.float32)
print(f"\n📌 Conditioning on {COHORT_FILTER} cohort mean:")
for f, v in zip(FEATURES, x_obs.numpy()):
    print(f"   {f:18s}  {v:.4f}")

print(f"\n🎯 Sampling {N_POSTERIOR_SAMPLES} posterior draws...")
samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs).numpy()

# Back to linear scale
samples_lin = samples.copy()
for i, p in enumerate(PARAM_COLS):
    if p in PARAM_LOG:
        samples_lin[:, i] = 10 ** samples[:, i]
posterior_df = pd.DataFrame(samples_lin, columns=PARAM_COLS)
out_csv = SAVE_DIR / f"SNPE_posterior_{COHORT_FILTER}.csv"
posterior_df.to_csv(out_csv, index=False)
print(f"💾 Saved to {out_csv}")


fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, p in zip(axes.flat, PARAM_COLS):
    prior_vals = sim[p].values
    post_vals  = posterior_df[p].values
    use_log = p in PARAM_LOG
    if use_log:
        prior_vals = np.log10(prior_vals)
        post_vals  = np.log10(post_vals)
        xlabel = f"log10({p})"
    else:
        xlabel = p
    bins = np.linspace(prior_vals.min(), prior_vals.max(), 40)
    ax.hist(prior_vals, bins=bins, density=True, alpha=0.35, color='gray',
            label='Prior')
    ax.hist(post_vals,  bins=bins, density=True, alpha=0.7,  color='purple',
            label='SNPE posterior')
    ax.axvline(np.median(post_vals), color='indigo', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel); ax.set_title(p); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes.flat[-1].axis('off')
fig.suptitle(f"SNPE marginal posteriors ({COHORT_FILTER} cohort mean)",
             fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR / f"SNPE_marginals_{COHORT_FILTER}.png", dpi=150)
plt.show()
plt.close()


labels = [f"log10({p})" if p in PARAM_LOG else p for p in PARAM_COLS]
n = len(PARAM_COLS)
fig, axes = plt.subplots(n, n, figsize=(2.0*n, 2.0*n))
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            ax.hist(samples[:, i], bins=40, color='purple', alpha=0.6, density=True)
        elif j < i:
            ax.hexbin(samples[:, j], samples[:, i], gridsize=25,
                      cmap='Purples', mincnt=1)
        else:
            ax.axis('off'); continue
        if i == n - 1: ax.set_xlabel(labels[j], fontsize=8)
        if j == 0:     ax.set_ylabel(labels[i], fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle(f"SNPE pairwise posterior ({COHORT_FILTER})", fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR / f"SNPE_pairwise_{COHORT_FILTER}.png", dpi=150)
plt.show()
plt.close()


print(f"\n{'='*72}")
print(f"SNPE POSTERIOR SUMMARY (N={N_POSTERIOR_SAMPLES})")
print(f"{'='*72}")
print(f"{'param':10s} {'median':>10s} {'p5':>10s} {'p95':>10s}   prior_range")
print('-' * 72)
for p in PARAM_COLS:
    v = posterior_df[p]
    med, lo, hi = v.median(), v.quantile(0.05), v.quantile(0.95)
    p_lo, p_hi = PRIOR[p][1], PRIOR[p][2]
    print(f"{p:10s} {med:>10.3f} {lo:>10.3f} {hi:>10.3f}   [{p_lo:g}, {p_hi:g}]")

# Constraint strength = 1 - posterior_width / prior_width (log scale for log params)
print(f"\n📐 Constraint strength (1 - posterior_width / prior_width):")
for p in PARAM_COLS:
    pri_lo, pri_hi = PRIOR[p][1], PRIOR[p][2]
    pv = posterior_df[p].values
    if p in PARAM_LOG:
        pri_w  = np.log10(pri_hi) - np.log10(pri_lo)
        post_w = np.log10(np.quantile(pv, 0.95)) - np.log10(np.quantile(pv, 0.05))
    else:
        pri_w  = pri_hi - pri_lo
        post_w = np.quantile(pv, 0.95) - np.quantile(pv, 0.05)
    c = 1.0 - post_w / pri_w
    tag = ("tightly constrained"   if c > 0.7 else
           "moderately constrained" if c > 0.4 else
           "weakly constrained"     if c > 0.15 else
           "degenerate")
    print(f"   {p:8s}  {100*c:5.1f}%   ({tag})")
print('=' * 72)
