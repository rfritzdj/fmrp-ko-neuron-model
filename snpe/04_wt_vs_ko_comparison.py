"""
04_wt_vs_ko_comparison.py
=========================

STEP 4 of the pipeline (Path 2 comparison).

Compares WT and KO representative parameter distributions per conductance.
For each parameter, plots WT vs KO histograms overlaid and computes a
Mann-Whitney U test to check whether the distributions differ significantly.

Input
-----
  WT_representative_params_direct.csv (or _SNPE.csv)
  KO_representative_params_direct.csv (or _SNPE.csv)

Output
------
  WT_vs_KO_parameter_comparison.png     : 7-panel violin/histogram plot
  WT_vs_KO_statistics.csv               : per-parameter test statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')

# Which files to compare. Use SNPE outputs by changing '_direct' to '_SNPE'.
WT_CSV = DATA_DIR / "WT_representative_params_direct.csv"
KO_CSV = DATA_DIR / "KO_representative_params_direct.csv"


# WT_CSV = DATA_DIR / "WT_representative_params_SNPE.csv"
# KO_CSV = DATA_DIR / "KO_representative_params_SNPE.csv"

PARAM_COLS = ['gNa', 'gKv1', 'gKv3', 'gCa', 'gSK', 'Btot', 'gleak']
PARAM_LOG  = {'gNa', 'gKv1', 'gKv3', 'gCa', 'gSK'}   # plot these on log axis

OUT_FIG   = DATA_DIR / "WT_vs_KO_parameter_comparison.png"
OUT_STATS = DATA_DIR / "WT_vs_KO_statistics.csv"

# =============================================================================
# 1. LOAD
# =============================================================================
if not WT_CSV.exists():
    raise FileNotFoundError(f"{WT_CSV} not found. Run 02_direct_ranking.py with COHORT_FILTER='WT' first.")
if not KO_CSV.exists():
    raise FileNotFoundError(f"{KO_CSV} not found. Run 02_direct_ranking.py with COHORT_FILTER='KO' first.")

wt = pd.read_csv(WT_CSV)
ko = pd.read_csv(KO_CSV)
print(f"Loaded {len(wt)} WT sets and {len(ko)} KO sets.")

# =============================================================================
# 2. STATISTICS PER PARAMETER
# =============================================================================
rows = []
for p in PARAM_COLS:
    wt_vals = wt[p].dropna().values
    ko_vals = ko[p].dropna().values

    # Mann-Whitney U (nonparametric — doesn't assume normality)
    stat, pval = mannwhitneyu(wt_vals, ko_vals, alternative='two-sided')

    # Effect size (rank-biserial correlation, related to U)
    n1, n2 = len(wt_vals), len(ko_vals)
    r = 1.0 - (2.0 * stat) / (n1 * n2)

    # Percent difference of medians
    wt_med, ko_med = np.median(wt_vals), np.median(ko_vals)
    pct_change = 100.0 * (ko_med - wt_med) / wt_med if wt_med != 0 else np.nan

    rows.append({
        'parameter': p,
        'WT_median': wt_med,
        'KO_median': ko_med,
        'WT_p5':  np.quantile(wt_vals, 0.05),
        'WT_p95': np.quantile(wt_vals, 0.95),
        'KO_p5':  np.quantile(ko_vals, 0.05),
        'KO_p95': np.quantile(ko_vals, 0.95),
        'pct_change_KO_vs_WT': pct_change,
        'mann_whitney_U': stat,
        'p_value': pval,
        'effect_size_r': r,
    })

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT_STATS, index=False)

# Print table
print(f"\n{'='*90}")
print(f"WT vs KO PARAMETER COMPARISON  (Mann-Whitney U, N_WT={len(wt)}, N_KO={len(ko)})")
print(f"{'='*90}")
print(f"{'param':8s} {'WT median':>10s} {'KO median':>10s} {'% change':>10s} "
      f"{'p-value':>10s} {'effect (r)':>12s}  {'sig':>8s}")
print("-" * 90)
for _, r in stats_df.iterrows():
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
    print(f"{r['parameter']:8s} {r['WT_median']:>10.3f} {r['KO_median']:>10.3f} "
          f"{r['pct_change_KO_vs_WT']:>9.1f}% {r['p_value']:>10.4g} "
          f"{r['effect_size_r']:>12.3f}  {sig:>8s}")
print(f"{'='*90}")
print(f"💾 Saved to {OUT_STATS}\n")

# =============================================================================
# 3. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, p in zip(axes.flat, PARAM_COLS):
    wt_vals = wt[p].dropna().values
    ko_vals = ko[p].dropna().values
    use_log = p in PARAM_LOG

    if use_log:
        wt_plot = np.log10(wt_vals)
        ko_plot = np.log10(ko_vals)
        xlabel  = f"log10({p})"
    else:
        wt_plot = wt_vals
        ko_plot = ko_vals
        xlabel  = p

    lo = min(wt_plot.min(), ko_plot.min())
    hi = max(wt_plot.max(), ko_plot.max())
    pad = 0.1 * (hi - lo) if hi > lo else 0.1
    bins = np.linspace(lo - pad, hi + pad, 25)

    ax.hist(wt_plot, bins=bins, density=True, alpha=0.55, color='steelblue',
            edgecolor='black', linewidth=0.5, label=f'WT (n={len(wt_vals)})')
    ax.hist(ko_plot, bins=bins, density=True, alpha=0.55, color='indianred',
            edgecolor='black', linewidth=0.5, label=f'KO (n={len(ko_vals)})')

    ax.axvline(np.median(wt_plot), color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(np.median(ko_plot), color='indianred', linestyle='--', linewidth=1.5)

    # Significance annotation
    row = stats_df[stats_df['parameter'] == p].iloc[0]
    if   row['p_value'] < 0.001: sig = '***'
    elif row['p_value'] < 0.01:  sig = '**'
    elif row['p_value'] < 0.05:  sig = '*'
    else:                          sig = 'ns'

    ax.set_title(f"{p}   ({sig}, p={row['p_value']:.3g})", fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes.flat[-1].axis('off')
fig.suptitle('WT vs KO parameter distributions (representative sets)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200)
plt.show()
plt.close()

print(f"✅ Done. Figure saved to {OUT_FIG}")
