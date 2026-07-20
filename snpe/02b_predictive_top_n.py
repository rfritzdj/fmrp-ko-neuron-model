"""
02b_predictive_top_n.py
=======================

This step goes after the SNPE posterior calculation

------------
Loads the SNPE posterior produced by 02a, samples N parameter sets,
re-simulates each, and picks the top-K closest to the cohort mean.


SNPE returns a continuous posterior (N samples). The channel-
perturbation script needs discrete representative sets to sweep. This
script bridges that gap, and validates that the posterior samples
actually produce cohort-like behaviour when re-simulated.

If the top-K is far from the cohort mean (e.g. spike_count clustered at
90 when cohort = 55), the SNPE posterior has collapsed to a wrong corner.
That's the sign to switch to Path A (direct ranking).

Output
------
  <COHORT>_representative_params_SNPE.csv  : top-K param sets (compatible
                                              with 03_channel_perturbation.py)
  SNPE_predictive_<COHORT>.png             : feature histograms
  SNPE_traces_<COHORT>.png                 : top-6 V traces
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import sem
import pyabf
import warnings

from pvin_model import solve_pvin_rk4, make_y0, warm_up_jit
from features import compute_stats, STAT_NAMES

warnings.filterwarnings('ignore')

DATA_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
ROOT_DIR = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/')
ABF_DIR  = ROOT_DIR / 'Abf Traces'

# COHORT_FILTER = 'WT'
COHORT_FILTER = 'KO'

N_SAMPLES_FROM_POSTERIOR = 300   # re-simulate this many posterior draws
N_REPRESENTATIVE         = 30    # keep this many as "top" sets
N_TRACES_TO_PLOT         = 6
N_JOBS                   = 8
MISSING_PENALTY_SEM      = 3.0

ACTIVE_SWEEP_IDX = 8
SIM_VLEAK = -72.0

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
_tspan_ms = np.ascontiguousarray((1000.0 * abf.sweepX) - (1000.0 * abf.sweepX[0]),
                                 dtype=np.float64)
_I_exp    = np.ascontiguousarray(abf.sweepC, dtype=np.float64)
stim_idx  = np.where(_I_exp > 1e-3)[0]
_STIM_ON_MS  = float(_tspan_ms[stim_idx[0]])
_STIM_OFF_MS = float(_tspan_ms[stim_idx[-1]])
print(f"📂 Canonical protocol: stim {_STIM_ON_MS:.0f}–{_STIM_OFF_MS:.0f} ms")
warm_up_jit(_tspan_ms, _I_exp, SIM_VLEAK)


def simulate_with_trace(theta):
    """Return (V, features) for one parameter dict."""
    try:
        y0 = make_y0(SIM_VLEAK, theta['gCa'])
        V = solve_pvin_rk4(y0, _tspan_ms, _I_exp,
                           theta['gNa'], theta['gKv1'], theta['gKv3'],
                           theta['gCa'], theta['gSK'], theta['gleak'],
                           theta['Btot'], SIM_VLEAK)
        return V, compute_stats(V, _tspan_ms, _STIM_ON_MS, _STIM_OFF_MS)
    except Exception:
        return np.full_like(_tspan_ms, np.nan), {k: np.nan for k in STAT_NAMES}

def simulate_features_only(theta):
    return simulate_with_trace(theta)[1]


posterior_df = pd.read_csv(DATA_DIR / f"SNPE_posterior_{COHORT_FILTER}.csv")
print(f"\nLoaded {len(posterior_df)} SNPE posterior samples.")

rng = np.random.default_rng(0)
idx = rng.choice(len(posterior_df),
                 size=min(N_SAMPLES_FROM_POSTERIOR, len(posterior_df)),
                 replace=False)
draws = posterior_df.iloc[idx].reset_index(drop=True)

results = Parallel(n_jobs=N_JOBS, verbose=2)(
    delayed(simulate_features_only)(draws.iloc[i].to_dict())
    for i in range(len(draws)))
sim_full = pd.concat([draws.reset_index(drop=True),
                      pd.DataFrame(results)], axis=1)


real_df = pd.read_csv(DATA_DIR / f"feasibility_real_{COHORT_FILTER.lower()}.csv")
target = {f: {'mean': float(np.nanmean(real_df[f].values)),
              'sem':  float(sem(real_df[f].dropna()))}
          for f in DISTANCE_FEATURES if real_df[f].notna().sum() >= 2}

required = [f for f in ['spike_count', 'adapt_idx', 'discharge_time_s',
                         'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms']
            if f in target]
valid = sim_full.dropna(subset=required).copy()
valid = valid[valid['spike_count'] >= 3].copy()

feature_vals = np.column_stack([valid[f].values for f in target])
means_arr    = np.array([target[f]['mean'] for f in target])
sems_arr     = np.array([max(target[f]['sem'], 1e-9) for f in target])
z = (feature_vals - means_arr) / sems_arr
z = np.where(np.isnan(z), MISSING_PENALTY_SEM, z)
valid['z_distance'] = np.sqrt(np.nanmean(z ** 2, axis=1))

top = valid.nsmallest(N_REPRESENTATIVE, 'z_distance').reset_index(drop=True)
print(f"\nTop {len(top)} SNPE-derived representative sets "
      f"(z range: {top['z_distance'].min():.2f}–{top['z_distance'].max():.2f})")

out_csv = DATA_DIR / f"{COHORT_FILTER}_representative_params_SNPE.csv"
top.to_csv(out_csv, index=False)

for f in ALL_PLOT_FEATURES:
    if f not in real_df.columns or real_df[f].notna().sum() < 3: continue
    real_m = real_df[f].mean(); real_s = sem(real_df[f].dropna())
    sim_m  = top[f].mean();     sim_s  = top[f].std()
    tag = "[fit]" if f in DISTANCE_FEATURES else "[held]"
    print(f"{f:18s}  {real_m:8.3f}±{real_s:6.3f}  {sim_m:8.3f}±{sim_s:6.3f}  {tag}")



plot_feats = [f for f in ALL_PLOT_FEATURES
              if f in sim_full.columns and real_df[f].notna().sum() >= 3]
n_cols = 4; n_rows = int(np.ceil(len(plot_feats) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.2*n_rows))
for ax, f in zip(axes.flat, plot_feats):
    real_vals = real_df[f].dropna().values
    sim_vals  = sim_full[f].dropna().values
    top_vals  = top[f].dropna().values
    if len(sim_vals) == 0: continue
    lo, hi = np.nanpercentile(np.concatenate([real_vals, sim_vals]), [1, 99])
    bins = np.linspace(lo, hi, 30)
    ax.hist(sim_vals,  bins=bins, density=True, alpha=0.3,  color='gray',
            label=f'Posterior sim (N={len(sim_vals)})')
    ax.hist(top_vals,  bins=bins, density=True, alpha=0.55, color='purple',
            label=f'Top {len(top)}')
    ax.hist(real_vals, bins=bins, density=True, alpha=0.7,  color='darkorange',
            label=f'Real (N={len(real_vals)})')
    tag = " [fit]" if f in DISTANCE_FEATURES else " [held]"
    ax.set_title(f + tag, fontsize=9)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for j in range(len(plot_feats), n_rows*n_cols):
    axes.flat[j].axis('off')
fig.suptitle(f"SNPE predictive check vs {COHORT_FILTER} cohort", fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / f"SNPE_predictive_{COHORT_FILTER}.png", dpi=150)
plt.show()


n_to_plot = min(N_TRACES_TO_PLOT, len(top))

trace_results = Parallel(n_jobs=N_JOBS, verbose=0)(
    delayed(simulate_with_trace)(top.iloc[i].to_dict()) for i in range(n_to_plot))

fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex=True, sharey=True)
for ax, (V, _), (i, row) in zip(axes.flat, trace_results, top.iterrows()):
    ax.plot(_tspan_ms / 1000.0, V, color='purple', linewidth=0.6)
    ax.axvline(_STIM_ON_MS / 1000.0,  color='gray', alpha=0.3)
    ax.axvline(_STIM_OFF_MS / 1000.0, color='gray', alpha=0.3)
    ax.set_title(f"#{i+1}  z={row['z_distance']:.2f}  "
                 f"gNa={row['gNa']:.0f} gKv3={row['gKv3']:.0f}",
                 fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for ax in axes[-1, :]: ax.set_xlabel('Time (s)')
for ax in axes[:, 0]:  ax.set_ylabel('V (mV)')
fig.suptitle(f"Top-{n_to_plot} SNPE-derived parameter sets ({COHORT_FILTER})",
             fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / f"SNPE_traces_{COHORT_FILTER}.png", dpi=150)
plt.show()

