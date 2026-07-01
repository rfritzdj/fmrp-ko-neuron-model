"""
02_direct_ranking.py
====================

STEP 2 of the pipeline.

What it does
------------
Loads the simulation cloud from step 1 and ranks each simulation by how
close its features are to the real cohort's mean. The top N "best" sets
are saved as a representative parameter cloud for the genotype.

This bypasses SNPE entirely. We tried SNPE; it gave a tighter posterior
but it collapsed onto a corner where shape features matched perfectly
while firing rate / discharge time were systematically off. Direct
ranking on the raw feasibility sims avoids that collapse — it can pick
sets matching firing properties even if shape features are slightly off.

Distance metric
---------------
For each simulation, the distance is the root-mean-squared z-score
across the chosen features, where z = (sim - cohort_mean) / cohort_SEM.

Missing values (e.g. tau_f undefined for non-adapting cells) are treated
as "MISSING_PENALTY_SEM SEMs away" rather than skipped. This stops sims
that fail to define features from getting an unfair advantage.

Features used for ranking exclude tau_f_s by default (the model rarely
produces a clean exponential frequency decay, so requiring it would push
the ranking toward odd corners).

Output
------
  <COHORT>_representative_params_direct.csv : top-N rows of params + features
  direct_ranking_predictive.png             : full pool / top-N / real cohort
  direct_ranking_traces.png                 : V traces of top 6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem
import pyabf
import warnings

from pvin_model import solve_pvin_rk4, make_y0, warm_up_jit

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
ROOT_DIR = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/')
ABF_DIR  = ROOT_DIR / 'Abf Traces'

COHORT_FILTER = 'WT'              # WT or KO; must match step 1 output
N_REPRESENTATIVE = 30             # how many top sets to keep
N_TRACES_TO_PLOT = 6              # how many example V traces
MISSING_PENALTY_SEM = 3.0         # penalty when a feature is NaN

# Features that drive the ranking. tau_f_s excluded; mean_freq / latency /
# isi_cv held out for validation (not used in fit).
DISTANCE_FEATURES = [
    'spike_count', 'adapt_idx', 'discharge_time_s',
    'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms',
]

ALL_PLOT_FEATURES = [
    'spike_count', 'mean_freq_Hz', 'adapt_idx', 'tau_f_s',
    'latency_ms', 'isi_cv', 'discharge_time_s',
    'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms',
]

PARAM_COLS = ['gNa', 'gKv1', 'gKv3', 'gCa', 'gSK', 'Btot', 'gleak']
SIM_VLEAK = -72.0
ACTIVE_SWEEP_IDX = 8

# =============================================================================
# 1. LOAD CACHED DATA
# =============================================================================
sim_df  = pd.read_csv(DATA_DIR / "feasibility_sims.csv")
real_df = pd.read_csv(DATA_DIR / f"feasibility_real_{COHORT_FILTER.lower()}.csv")
print(f"Loaded {len(sim_df)} sims and {len(real_df)} real {COHORT_FILTER} cells.")

# Filter to viable sims (must spike ≥3 times, must have all required features)
sim = sim_df.dropna(subset=['spike_count']).copy()
sim = sim[sim['spike_count'] >= 3].copy()
required = ['spike_count', 'adapt_idx', 'discharge_time_s',
            'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms']
required = [f for f in required if f in sim.columns]
sim = sim.dropna(subset=required).reset_index(drop=True)
print(f"Viable sims (≥3 spikes, all required features): {len(sim)}")

# =============================================================================
# 2. COMPUTE COHORT TARGET
# =============================================================================
target = {f: {'mean': float(np.nanmean(real_df[f].values)),
              'sem':  float(sem(real_df[f].dropna()))}
          for f in DISTANCE_FEATURES if real_df[f].notna().sum() >= 2}

print(f"\nCohort target (mean ± SEM):")
for f in target:
    t = target[f]
    print(f"   {f:18s}  {t['mean']:8.3f} ± {t['sem']:6.3f}")

# =============================================================================
# 3. NAN-AWARE Z-DISTANCE OVER ENTIRE VIABLE POOL
# =============================================================================
# Stack each sim's features into a matrix, divide by cohort SEMs, replace
# any NaN z-scores with the missing penalty, then take RMS across features.
feature_vals = np.column_stack([sim[f].values for f in target])
means_arr    = np.array([target[f]['mean'] for f in target])
sems_arr     = np.array([max(target[f]['sem'], 1e-9) for f in target])

z = (feature_vals - means_arr) / sems_arr
z = np.where(np.isnan(z), MISSING_PENALTY_SEM, z)
sim['z_distance'] = np.sqrt(np.nanmean(z ** 2, axis=1))

# Diagnostic: where does the full pool sit vs cohort?
print(f"\nFull viable pool stats:")
print(f"   {'feature':18s}  {'pool mean':>10s}  {'range':>22s}  {'cohort':>10s}")
for f in DISTANCE_FEATURES:
    if f not in sim.columns: continue
    v = sim[f]
    print(f"   {f:18s}  {v.mean():>10.3f}  "
          f"[{v.min():>7.3f}, {v.max():>7.3f}]  {target[f]['mean']:>10.3f}")

# =============================================================================
# 4. KEEP TOP-N
# =============================================================================
top = sim.nsmallest(N_REPRESENTATIVE, 'z_distance').reset_index(drop=True)
print(f"\nTop {len(top)} representative {COHORT_FILTER} parameter sets")
print(f"(z range: {top['z_distance'].min():.2f}–{top['z_distance'].max():.2f})")
print(top[['z_distance'] + PARAM_COLS].to_string(index=True))

out_csv = DATA_DIR / f"{COHORT_FILTER}_representative_params_direct.csv"
top.to_csv(out_csv, index=False)
print(f"\n💾 Saved to {out_csv}")

# =============================================================================
# 5. FEATURE REPRODUCTION REPORT
# =============================================================================
print(f"\n{'='*72}")
print(f"FEATURE REPRODUCTION (top-{len(top)})")
print(f"{'='*72}")
print(f"{'feature':18s}  {'cohort mean±SEM':>18s}  {'top-N mean±std':>20s}")
print("-" * 72)
for f in ALL_PLOT_FEATURES:
    if f not in real_df.columns or real_df[f].notna().sum() < 3: continue
    real_m = real_df[f].mean(); real_s = sem(real_df[f].dropna())
    sim_m  = top[f].mean();     sim_s  = top[f].std()
    tag = "[fit]" if f in DISTANCE_FEATURES else "[held]"
    print(f"{f:18s}  {real_m:8.3f}±{real_s:6.3f}  {sim_m:8.3f}±{sim_s:6.3f}  {tag}")
print(f"{'='*72}\n")

# =============================================================================
# 6. PREDICTIVE CHECK (3-layer histograms: full pool / top-N / cohort)
# =============================================================================
plot_feats = [f for f in ALL_PLOT_FEATURES
              if f in sim.columns and real_df[f].notna().sum() >= 3]
n_cols = 4
n_rows = int(np.ceil(len(plot_feats) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.2*n_rows))
for ax, f in zip(axes.flat, plot_feats):
    real_vals = real_df[f].dropna().values
    pool_vals = sim[f].dropna().values
    top_vals  = top[f].dropna().values
    if len(top_vals) == 0: continue
    lo, hi = np.nanpercentile(np.concatenate([real_vals, top_vals]), [1, 99])
    bins = np.linspace(lo, hi, 30)
    ax.hist(pool_vals, bins=bins, density=True, alpha=0.25, color='gray',
            label=f'Pool (N={len(pool_vals)})')
    ax.hist(top_vals,  bins=bins, density=True, alpha=0.55, color='seagreen',
            label=f'Top {len(top)}')
    ax.hist(real_vals, bins=bins, density=True, alpha=0.7,  color='darkorange',
            label=f'Real (N={len(real_vals)})')
    tag = " [fit]" if f in DISTANCE_FEATURES else " [held]"
    ax.set_title(f + tag, fontsize=9)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for j in range(len(plot_feats), n_rows*n_cols):
    axes.flat[j].axis('off')
fig.suptitle(f"Direct ranking: top-{len(top)} vs cohort ({COHORT_FILTER})",
             fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / "direct_ranking_predictive.png", dpi=150)
plt.show()
plt.close()

# =============================================================================
# 7. RE-SIMULATE TOP 6 FOR V-TRACE PLOTS
# =============================================================================
print(f"\n📈 Re-simulating top {N_TRACES_TO_PLOT} for V-trace plots...")
meta = pd.read_excel(ROOT_DIR / "EPHYS_data_astrocytes.xlsx",
                     converters={'Date': str, 'Code': str})
meta.columns = meta.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
cohort_meta = meta[(meta['Input'] == 'step') &
                   (meta['Self_eval'] != 'omit') &
                   (meta['Mouse_info'] == COHORT_FILTER)]
abf_path = next(ABF_DIR / r.Date / f"{r.Code}.abf"
                for r in cohort_meta.itertuples()
                if (ABF_DIR / r.Date / f"{r.Code}.abf").exists())
abf = pyabf.ABF(str(abf_path))
abf.setSweep(ACTIVE_SWEEP_IDX)
tspan_ms = np.ascontiguousarray((1000.0*abf.sweepX)-(1000.0*abf.sweepX[0]),
                                dtype=np.float64)
I_exp    = np.ascontiguousarray(abf.sweepC, dtype=np.float64)
stim_idx = np.where(I_exp > 1e-3)[0]
STIM_ON  = float(tspan_ms[stim_idx[0]])
STIM_OFF = float(tspan_ms[stim_idx[-1]])
warm_up_jit(tspan_ms, I_exp, SIM_VLEAK)

n_to_plot = min(N_TRACES_TO_PLOT, len(top))
fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True, sharey=True)
for ax, (i, row) in zip(axes.flat, top.head(n_to_plot).iterrows()):
    y0 = make_y0(SIM_VLEAK, row['gCa'])
    V = solve_pvin_rk4(y0, tspan_ms, I_exp,
                       row['gNa'], row['gKv1'], row['gKv3'],
                       row['gCa'], row['gSK'], row['gleak'],
                       row['Btot'], SIM_VLEAK)
    ax.plot(tspan_ms / 1000.0, V, color='seagreen', linewidth=0.6)
    ax.axvline(STIM_ON / 1000.0,  color='gray', alpha=0.3)
    ax.axvline(STIM_OFF / 1000.0, color='gray', alpha=0.3)
    ax.set_title(f"#{i+1}  z={row['z_distance']:.2f}  "
                 f"gNa={row['gNa']:.0f} gKv3={row['gKv3']:.0f}",
                 fontsize=9)
for ax in axes[-1, :]: ax.set_xlabel('Time (s)')
for ax in axes[:, 0]:  ax.set_ylabel('V (mV)')
fig.suptitle(f"Top-{n_to_plot} representative {COHORT_FILTER} parameter sets",
             fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / "direct_ranking_traces.png", dpi=150)
plt.show()
plt.close()

print(f"\n✅ Done.")
