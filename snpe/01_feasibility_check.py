"""
01_feasibility_check.py
=======================

STEP 1 of the pipeline.

What it does
------------
1. Loads a canonical current-injection protocol from one real ABF file
2. Draws 20,000 parameter sets from a broad log-uniform prior
3. Simulates each, computes 10 summary features
4. Extracts the same features from every real WT cell
5. Saves both to CSV for downstream analysis

Output
------
  feasibility_sims.csv     : 20k rows of (params + features)
  feasibility_real_wt.csv  : N rows of features per real cell
  feasibility_marginals.png: histograms of sim vs real per feature
  feasibility_pairwise.png : pairwise scatter of 5 key features

Runtime: ~3-8 minutes with N_JOBS=8 (~16 GB RAM).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import pyabf
import warnings

from pvin_model import solve_pvin_rk4, make_y0, warm_up_jit
from features import compute_stats, STAT_NAMES

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
ROOT_DIR  = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/')
ABF_DIR   = ROOT_DIR / 'Abf Traces'
SAVE_DIR  = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES        = 20000  # ~12% of these will be viable spiking sims
N_JOBS           = 8      # parallel workers
ACTIVE_SWEEP_IDX = 8      # sweep index in the ABFs that has the 200 pA step
SIM_VLEAK        = -72.0  # leak reversal for simulator initial conditions

# Mouse_info filter: 'WT' for the WT cohort, change to 'KO' for KO
# COHORT_FILTER = 'WT'
COHORT_FILTER = 'KO'

# Prior over the 7 model parameters. 'log' = log-uniform, 'lin' = uniform.
# These ranges were chosen to broadly cover plausible PV-cell values.
PRIOR = {
    'gNa':   ('log', 30.0,  450.0),   # main Na conductance
    'gKv1':  ('log', 0.5,   50.0),    # slow K
    'gKv3':  ('log', 20.0,  300.0),   # fast K (narrow spikes)
    'gCa':   ('log', 0.5,   50.0),    # Ca conductance
    'gSK':   ('log', 0.1,   50.0),    # SK channel (slow AHP)
    'Btot':  ('lin', 10.0,  120.0),   # total Ca buffer
    'gleak': ('lin', 0.5,   5.0),     # passive leak
}

# =============================================================================
# 1. LOAD CANONICAL PROTOCOL
# =============================================================================
# All real cells in this dataset were recorded with the same step protocol on
# sweep 8 (200 pA × 1 s). We load that protocol from one ABF and use it as
# the input current for every simulation, so the simulated cohort is directly
# comparable to the real cohort.

def load_canonical_protocol():
    meta = pd.read_excel(ROOT_DIR / "EPHYS_data_astrocytes.xlsx",
                         converters={'Date': str, 'Code': str})
    meta.columns = meta.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
    cohort = meta[(meta['Input'] == 'step') &
                  (meta['Self_eval'] != 'omit') &
                  (meta['Mouse_info'] == COHORT_FILTER)]

    abf_path = None
    for r in cohort.itertuples():
        p = ABF_DIR / r.Date / f"{r.Code}.abf"
        if p.exists():
            abf_path = p
            break
    if abf_path is None:
        raise FileNotFoundError(f"No {COHORT_FILTER} ABF found.")

    abf = pyabf.ABF(str(abf_path))
    abf.setSweep(ACTIVE_SWEEP_IDX)
    t_ms = (1000.0 * abf.sweepX) - (1000.0 * abf.sweepX[0])
    I = abf.sweepC.copy()
    stim_idx = np.where(I > 1e-3)[0]
    stim_on  = float(t_ms[stim_idx[0]])
    stim_off = float(t_ms[stim_idx[-1]])
    print(f"📂 Canonical protocol: {abf_path.name}, stim {stim_on:.0f}–{stim_off:.0f} ms, "
          f"dt {t_ms[1]-t_ms[0]:.3f} ms")
    return (np.ascontiguousarray(t_ms, dtype=np.float64),
            np.ascontiguousarray(I,    dtype=np.float64),
            stim_on, stim_off, cohort)

_tspan_ms, _I_exp, _STIM_ON_MS, _STIM_OFF_MS, _meta = load_canonical_protocol()
warm_up_jit(_tspan_ms, _I_exp, SIM_VLEAK)

# =============================================================================
# 2. SAMPLE FROM PRIOR
# =============================================================================
def sample_prior(n, seed=0):
    """Draw n parameter sets from the log/linear-uniform prior."""
    rng = np.random.default_rng(seed)
    samples = {}
    for name, (scale, lo, hi) in PRIOR.items():
        if scale == 'log':
            samples[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            samples[name] = rng.uniform(lo, hi, n)
    return pd.DataFrame(samples)

print(f"\n📦 Sampling {N_SAMPLES} parameter sets from prior...")
theta_df = sample_prior(N_SAMPLES, seed=0)

# =============================================================================
# 3. SIMULATE EACH PARAM SET, EXTRACT FEATURES
# =============================================================================
def simulate_one(theta):
    """Run one simulation with parameters `theta` (dict) and return its features."""
    try:
        y0 = make_y0(SIM_VLEAK, theta['gCa'])
        V = solve_pvin_rk4(y0, _tspan_ms, _I_exp,
                           theta['gNa'], theta['gKv1'], theta['gKv3'],
                           theta['gCa'], theta['gSK'], theta['gleak'],
                           theta['Btot'], SIM_VLEAK)
        return compute_stats(V, _tspan_ms, _STIM_ON_MS, _STIM_OFF_MS)
    except Exception:
        return {k: np.nan for k in STAT_NAMES}

print(f"\n⚡ Simulating {N_SAMPLES} param sets in parallel (n_jobs={N_JOBS})...")
# Batched to avoid joblib pickling out-of-memory with very large N
BATCH = 2000
all_results = []
for start in range(0, N_SAMPLES, BATCH):
    end = min(start + BATCH, N_SAMPLES)
    batch = Parallel(n_jobs=N_JOBS)(
        delayed(simulate_one)(theta_df.iloc[i].to_dict()) for i in range(start, end))
    all_results.extend(batch)
    print(f"   {end}/{N_SAMPLES} done")
sim_df = pd.concat([theta_df.reset_index(drop=True),
                    pd.DataFrame(all_results)], axis=1)

# =============================================================================
# 4. EXTRACT FEATURES FROM REAL CELLS
# =============================================================================
print(f"\n🧬 Extracting features from real {COHORT_FILTER} cohort...")
rows = []
for r in tqdm(_meta.itertuples(), total=len(_meta), desc=f"Real {COHORT_FILTER}"):
    path = ABF_DIR / r.Date / f"{r.Code}.abf"
    if not path.exists(): continue
    try:
        abf = pyabf.ABF(str(path))
        abf.setSweep(ACTIVE_SWEEP_IDX)
        V = abf.sweepY; I = abf.sweepC
        t_ms = (1000.0 * abf.sweepX) - (1000.0 * abf.sweepX[0])
        stim = np.where(I > 0)[0]
        if len(stim) == 0: continue
        stats = compute_stats(V, t_ms, float(t_ms[stim[0]]), float(t_ms[stim[-1]]))
        stats['Code'] = r.Code
        rows.append(stats)
    except Exception:
        continue
real_df = pd.DataFrame(rows)
print(f"   Got {len(real_df)} real cells.")

# =============================================================================
# 5. SAVE EVERYTHING
# =============================================================================
sim_df.to_csv(SAVE_DIR / "feasibility_sims_ko.csv", index=False)
real_df.to_csv(SAVE_DIR / f"feasibility_real_{COHORT_FILTER.lower()}.csv", index=False)
print(f"\n💾 Saved sims and real cohort to {SAVE_DIR}")

# =============================================================================
# 6. DIAGNOSTIC PLOTS
# =============================================================================
spiking = sim_df.dropna(subset=['spike_count']).copy()
spiking = spiking[spiking['spike_count'] >= 2]
print(f"   Viable sims (≥2 spikes): {len(spiking)} / {N_SAMPLES} "
      f"({100*len(spiking)/N_SAMPLES:.1f}%)")

# Per-feature: how often is a simulation inside the cohort's 5th–95th band?
print(f"\n🎯 Per-feature overlap (sims falling in real {COHORT_FILTER} 5–95% range):")
for s in STAT_NAMES:
    if s not in real_df.columns or real_df[s].notna().sum() < 3:
        continue
    lo, hi = real_df[s].quantile(0.05), real_df[s].quantile(0.95)
    in_band = ((spiking[s] >= lo) & (spiking[s] <= hi)).mean()
    print(f"   {s:18s}  real [{lo:7.3f}, {hi:7.3f}]   sims in band: {100*in_band:5.1f}%")

# Marginal histograms
plot_stats = [s for s in STAT_NAMES if real_df[s].notna().sum() >= 3]
n_cols = 3
n_rows = int(np.ceil(len(plot_stats) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4*n_rows))
for ax, stat in zip(axes.flat, plot_stats):
    sim_vals = spiking[stat].dropna()
    real_vals = real_df[stat].dropna()
    if len(sim_vals) == 0: continue
    all_v = np.concatenate([sim_vals.values, real_vals.values])
    bins = np.linspace(*np.nanpercentile(all_v, [1, 99]), 40)
    ax.hist(sim_vals,  bins=bins, density=True, alpha=0.4, color='gray',
            label=f'Sim (N={len(sim_vals)})')
    ax.hist(real_vals, bins=bins, density=True, alpha=0.7, color='darkorange',
            label=f'Real (N={len(real_vals)})')
    ax.set_title(stat); ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for j in range(len(plot_stats), n_rows*n_cols):
    axes.flat[j].axis('off')
fig.suptitle(f"Feasibility check: simulated cloud vs {COHORT_FILTER} cohort",
             fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR / "feasibility_marginals_ko.png", dpi=150)
plt.show()
plt.close()

print(f"\n✅ Done.")
