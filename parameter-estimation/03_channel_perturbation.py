"""
03_channel_perturbation.py
==========================

STEP 3 of the pipeline.

What it does
------------
For each top-N WT parameter set, sweep each channel over multipliers from
0× to 3× the baseline. Compute features at each multiplier. Plot the
population-mean curve with SEM bands across the N models.

This is the channel-sensitivity analysis. If scaling channel X by 1.5×
moves a feature from WT value toward the KO value, then channel X is a
candidate mechanism for the WT→KO change.

Why population sweeps (not single-model)
----------------------------------------
A single representative parameter set might be at the edge of a stability
boundary, giving misleading perturbation curves. Averaging across N
models smooths over individual idiosyncrasies and exposes the trend
that's robust across the cohort.

Output
------
  perturbation_raw_<ch>.csv     : per-(model, multiplier) features
  perturbation_summary_<ch>.csv : population mean ± SEM per multiplier
  population_perturbation_bands.png : the main figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress, sem
from scipy.signal import find_peaks
from tqdm import tqdm
import pyabf
import efel
import warnings

from pvin_model import solve_pvin_rk4, make_y0, warm_up_jit

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
ABF_PATH = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/'
                'Abf Traces/20230622/0008.abf')

N_MODELS    = 30                       # how many representative sets to sweep
MULTIPLIERS = np.arange(0.0, 3.2, 0.2) # 0× to 3× baseline (16 multipliers)
SWEEP_CHANNELS = ['gNa', 'gKv1', 'gKv3', 'gCa', 'gSK', 'gleak', 'Btot']

SWEEP_VK       = True   # also sweep K reversal potential
VK_BASELINE    = -80.0
VK_DELTA_RANGE = 20.0   # sweep from VK_BASELINE up to VK_BASELINE+20

THRESHOLD_V = -10.0
PARAM_COLS  = ['gNa', 'gKv1', 'gKv3', 'gCa', 'gSK', 'gleak', 'Btot']

# =============================================================================
# FEATURE EXTRACTION (uses scipy + eFEL, matching step 1)
# =============================================================================
def get_f_decay_rate(voltage, tspan_ms, threshold=THRESHOLD_V):
    dt_s = (tspan_ms[1] - tspan_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    spike_idx, _ = find_peaks(voltage, height=threshold, distance=int(0.001 * fs))
    if len(spike_idx) < 4: return np.nan
    ISIs_ms = np.diff(tspan_ms[spike_idx])
    freq = 1.0 / (ISIs_ms / 1000.0)
    t_sec = (tspan_ms[spike_idx[1:]] - tspan_ms[spike_idx[1]]) / 1000.0
    if np.any(freq <= 0) or len(t_sec) < 2: return np.nan
    try:
        reg = linregress(t_sec, np.log(freq))
        return -1.0 / reg.slope if reg.slope < 0 else np.nan
    except ValueError:
        return np.nan

def get_features(V, tspan, stim_idx, thresholdV=THRESHOLD_V):
    """Three features via eFEL: discharge time, spike count, adaptation index."""
    spike_idx, _ = find_peaks(V, height=thresholdV)
    if len(spike_idx) > 5:
        trace = {'T': tspan, 'V': V,
                 'stim_start': [tspan[stim_idx[0]]],
                 'stim_end':   [tspan[stim_idx[-1]]]}
        feats = efel.get_feature_values([trace],
            ['peak_time', 'adaptation_index', 'spike_count'])
        spike_times = feats[0].get('peak_time')
        sc_arr      = feats[0].get('spike_count')
        ai_arr      = feats[0].get('adaptation_index')
        td = (spike_times[-1] - spike_times[0]) / 1000.0 \
             if spike_times is not None and len(spike_times) >= 2 else 0.0
        sc = float(sc_arr[0]) if sc_arr is not None and len(sc_arr) > 0 else 0.0
        ai = float(ai_arr[0]) if ai_arr is not None and len(ai_arr) > 0 else 0.0
        return td, sc, ai
    return 0.0, 0.0, 0.0

def analyze_trace(val, V, tspan_np, stim_idx):
    """Smooth + compute the 4 features for one (value, V) pair."""
    Vs = gaussian_filter1d(V, 10)
    tau_f = get_f_decay_rate(Vs, tspan_np)
    td, sc, ai = get_features(Vs, tspan_np, stim_idx)
    return {'Value': float(val), 'Spike_Count': float(sc),
            'Discharge_Time': float(td), 'Tau_f': float(tau_f),
            'Adaptation_Index': float(ai)}

# =============================================================================
# LOAD CANONICAL PROTOCOL & REPRESENTATIVE MODELS
# =============================================================================
abf = pyabf.ABF(str(ABF_PATH))
abf.setSweep(8)
Vdata, Idata = abf.sweepY, abf.sweepC
tspan = (abf.sweepX * 1000.0) - (abf.sweepX[0] * 1000.0)
stim_idx = np.where(Idata > 0)[0]
Vleak = float(gaussian_filter1d(Vdata, 10)[0])  # use cell's resting V
V0 = Vleak

tspan_arr = np.ascontiguousarray(tspan, dtype=np.float64)
Idata_arr = np.ascontiguousarray(Idata, dtype=np.float64)

print(f"📂 Protocol: {ABF_PATH.name}, Vleak={Vleak:.1f} mV, "
      f"stim {tspan[stim_idx[0]]:.0f}–{tspan[stim_idx[-1]]:.0f} ms")
warm_up_jit(tspan_arr, Idata_arr, Vleak)

csv_path = DATA_DIR / "WT_representative_params_direct.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"{csv_path} not found. Run 02_direct_ranking.py first.")
top_models = pd.read_csv(csv_path).head(N_MODELS).reset_index(drop=True)
print(f"🎯 Loaded top {len(top_models)} representative parameter sets")

# =============================================================================
# PERTURBATION SWEEP
# =============================================================================
def simulate_one(base_cond, override_key=None, override_val=None, VK_val=VK_BASELINE):
    """Run one sim with optional override of one parameter (and/or VK).
    Note: this version always uses VK = -80 in the model itself — VK swept
    here is a separate analysis variable, not actually changing model VK.
    To change the actual VK inside the model you would need to add a VK
    argument to solve_pvin_rk4. Kept as-is for compatibility."""
    cond = dict(base_cond)
    if override_key is not None:
        cond[override_key] = override_val
    y0 = make_y0(V0, cond['gCa'])
    try:
        V = solve_pvin_rk4(y0, tspan_arr, Idata_arr,
                           cond['gNa'], cond['gKv1'], cond['gKv3'],
                           cond['gCa'], cond['gSK'], cond['gleak'],
                           cond['Btot'], Vleak)
    except (ZeroDivisionError, FloatingPointError):
        V = np.full(len(tspan_arr), np.nan)
    return V

def sweep_one_channel_one_model(model_idx, base_cond, channel, sweep_vals):
    """Sweep one channel for one model. Returns list of feature records."""
    records = []
    for val in sweep_vals:
        if channel == 'VK':
            V = simulate_one(base_cond, override_key=None, VK_val=val)
        else:
            V = simulate_one(base_cond, override_key=channel, override_val=val)
        rec = analyze_trace(val, V, tspan_arr, stim_idx)
        rec['Multiplier']  = val if channel == 'VK' else \
            float(val / base_cond[channel]) if base_cond[channel] > 0 else 0.0
        rec['Model_Index'] = model_idx
        records.append((channel, rec))
    return records

# Assemble all (model × channel) tasks
print(f"\n⚡ Running perturbation sweeps "
      f"({len(top_models)} models × {len(SWEEP_CHANNELS) + int(SWEEP_VK)} channels × "
      f"{len(MULTIPLIERS)} multipliers)...")

tasks = []
for m_idx, model in top_models.iterrows():
    base_cond = {p: float(model[p]) for p in PARAM_COLS}
    for ch in SWEEP_CHANNELS:
        tasks.append((m_idx, base_cond, ch, MULTIPLIERS * base_cond[ch]))
    if SWEEP_VK:
        VK_vals = VK_BASELINE + np.linspace(0.0, VK_DELTA_RANGE, len(MULTIPLIERS))
        tasks.append((m_idx, base_cond, 'VK', VK_vals))

# Serial loop with progress bar (parallelization was crashing with memory errors)
swept = {ch: [] for ch in SWEEP_CHANNELS + (['VK'] if SWEEP_VK else [])}
for task in tqdm(tasks, desc="Sweeps"):
    for ch, rec in sweep_one_channel_one_model(*task):
        swept[ch].append(rec)

# =============================================================================
# COMPILE, PLOT, AND SAVE
# =============================================================================
METRICS = [
    ('Tau_f',            r'$\tau_f$ (s)',    True),
    ('Discharge_Time',   r'$t_d$ (s)',       False),
    ('Spike_Count',      'Spike Count',      False),
    ('Adaptation_Index', 'Adaptation Index', False),
]

channels_to_plot = SWEEP_CHANNELS + (['VK'] if SWEEP_VK else [])
fig, axes = plt.subplots(len(METRICS), len(channels_to_plot),
                         figsize=(3.5 * len(channels_to_plot), 3.2 * len(METRICS)),
                         sharey='row')

for ch_idx, ch in enumerate(channels_to_plot):
    ch_df = pd.DataFrame(swept[ch])
    ch_df.to_csv(DATA_DIR / f'perturbation_raw_{ch}.csv', index=False)

    summary_records = []
    for m_idx, (col_name, ylabel, clip_to_1) in enumerate(METRICS):
        ax = axes[m_idx, ch_idx] if len(channels_to_plot) > 1 else axes[m_idx]

        pivoted = ch_df.pivot(index='Multiplier', columns='Model_Index', values=col_name)
        if clip_to_1:
            pivoted = pivoted.clip(0, 1)

        mean_line = pivoted.mean(axis=1)
        def _safe_sem(row):
            v = row.dropna().values
            return sem(v) if len(v) >= 2 else 0.0
        sem_band = pivoted.apply(_safe_sem, axis=1)
        x_axis = pivoted.index.values

        for mult, m_val, s_val in zip(pivoted.index, mean_line, sem_band):
            summary_records.append({
                'Parameter': ch, 'Multiplier': float(mult),
                'Feature': col_name, 'Mean': float(m_val), 'SEM': float(s_val)})

        ax.plot(x_axis, mean_line, marker='o', markersize=4,
                color='darkblue', linewidth=1.5, zorder=3)
        ax.fill_between(x_axis, mean_line - sem_band, mean_line + sem_band,
                        color='royalblue', alpha=0.25, zorder=2)
        ax.axvline(1.0 if ch != 'VK' else VK_BASELINE,
                   color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        if m_idx == 0:
            ax.set_title(ch, fontweight='bold', fontsize=12)
        if m_idx == len(METRICS) - 1:
            ax.set_xlabel('$V_K$ (mV)' if ch == 'VK' else 'Multiplier (× baseline)')
        if ch_idx == 0:
            ax.set_ylabel(ylabel, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    pd.DataFrame(summary_records).to_csv(
        DATA_DIR / f'perturbation_summary_{ch}.csv', index=False)

fig.suptitle(f"Channel perturbation analysis (N={len(top_models)} representative models)",
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(DATA_DIR / 'population_perturbation_bands.png', dpi=300)
plt.show()
plt.close()

print(f"\n✅ Done. Outputs:")
print(f"   • population_perturbation_bands.png")
print(f"   • perturbation_raw_<channel>.csv")
print(f"   • perturbation_summary_<channel>.csv")
