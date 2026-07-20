"""
03_channel_perturbation.py
==========================

By this time, the parameters are defined

Loads the top-N WT representative parameter sets, sweeps each channel over
multipliers from 0× to 4× baseline (Btot scaled by 1/3 since its buffering
saturates quickly), and plots population mean ± SEM across the N models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress, sem
from scipy.signal import find_peaks
from numba import njit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import efel
import pyabf


CSV_PATH  = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/WT_representative_params_SNPE.csv')
ABF_PATH  = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/Abf Traces/20230622/0008.abf')
SAVE_DIR  = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/results/')

N_MODELS       = 30                      # how many top WT sets to sweep
MULTIPLIERS    = np.arange(0.0, 4.2, 0.2)  # 0× to 4× baseline
BTOT_SCALE     = 1.0 / 3.0               # Btot compression factor
SKIP_GLEAK     = True                     # exclude gleak from sweep because of instability
SWEEP_VK       = True
VK_BASELINE    = -80.0
VK_DELTA_RANGE = 20.0
THRESHOLD_V    = -10.0

OUT_RAW_CSV = 'channel_perturbation_raw.csv'
OUT_FIG     = 'channel_perturbation_bands.png'


VNa, VK, VCa = 58.0, -80.0, 68.0
Cm, pgamma, KD, F = 30.0, 0.01, 0.1, 0.0964853321
mArea, d, Car = 3000.0, 0.1, 0.07

Vm, Sm = -20.0, -7.0
Aah, Sah, Vah = 0.0025, 10.0, 18.4
Abh, Sbh, Vbh = 0.094, -5.5, -31.0
Aan1, Van1, San1 = 0.002, -36.0, -9.0
Abn1, Vbn1, Sbn1 = 0.017, -36.75, 6.785
Aan3, Van3, San3 = 3.2, 96.0, -12.6
Abn3, Vbn3, Sbn3 = 0.34, -36.0, 13.965
Va, Sa = 3.5, -11.4
nk, ksk = 5.0, 0.8

@njit(cache=True)
def vtrap(dV, S):
    x = dV / S
    if np.abs(x) < 1e-6:
        return -S * (1.0 - x / 2.0)
    else:
        return dV / (1.0 - np.exp(x))

@njit(cache=True)
def PVIN_HH_deriv(y, t, Iapp, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local):
    V, h, n1, n3, ca2i = y[0], y[1], y[2], y[3], y[4]
    mmax = 1.0 / (1.0 + np.exp((V - Vm) / Sm))
    ah = Aah / np.exp((V - Vah) / Sah)
    bh = Abh * vtrap(V - Vbh, Sbh)
    INa = gNa * (mmax ** 3) * h * (V - VNa)

    an1 = Aan1 * vtrap(V - Van1, San1)
    bn1 = Abn1 / np.exp((V - Vbn1) / Sbn1)
    IKv1 = gKv1 * (n1 ** 4) * (V - VK_local)
    an3 = Aan3 * vtrap(V - Van3, San3)
    bn3 = Abn3 / np.exp((V - Vbn3) / Sbn3)
    IKv3 = gKv3 * (n3 ** 2) * (V - VK_local)

    amax = 1.0 / (1.0 + np.exp((V - Va) / Sa))
    ICa = gCa * (amax ** 2) * (V - VCa)

    k_sk_denom = ksk ** nk + ca2i ** nk
    if np.abs(k_sk_denom) < 1e-12:
        k_sk = 0.5
    else:
        k_sk = (ca2i ** nk) / k_sk_denom
    ISK = gSK * k_sk * (V - VK_local)
    Ileak = gleak * (V - Vleak)

    dVdt = (-Ileak - INa - IKv1 - IKv3 - ICa - ISK + Iapp) / Cm
    dhdt = ah * (1.0 - h) - bh * h

    an1_sum = an1 + bn1
    if an1_sum < 1e-12:
        an1max = 0.5
        tau_n1max = 1e12
    else:
        an1max = an1 / an1_sum
        tau_n1max = 1.0 / an1_sum
    dn1dt = (an1max - n1) / tau_n1max
    dn3dt = an3 * (1.0 - n3) - bn3 * n3
    dCadt = (-ICa / (2.0 * F * mArea * d) - pgamma * (ca2i - Car)) / (1.0 + Bt / KD)

    return np.array([dVdt, dhdt, dn1dt, dn3dt, dCadt])

@njit(cache=True)
def solve_pvin_rk4(y0, tspan, Idata, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local):
    n_steps = len(tspan)
    dt = tspan[1] - tspan[0]
    Y = np.zeros((n_steps, 5))
    Y[0] = y0
    y_current = np.copy(y0)
    V_DIVERGENCE_BOUND = 1000.0

    for i in range(1, n_steps):
        t0 = tspan[i - 1]
        t1 = tspan[i]
        t_half = t0 + 0.5 * dt
        I0, I1 = Idata[i - 1], Idata[i]
        I_half = 0.5 * (I0 + I1)

        k1 = PVIN_HH_deriv(y_current, t0, I0, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local)
        k2 = PVIN_HH_deriv(y_current + 0.5 * dt * k1, t_half, I_half, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local)
        k3 = PVIN_HH_deriv(y_current + 0.5 * dt * k2, t_half, I_half, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local)
        k4 = PVIN_HH_deriv(y_current + dt * k3, t1, I1, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak, VK_local)

        y_current = y_current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if np.abs(y_current[0]) > V_DIVERGENCE_BOUND or np.isnan(y_current[0]):
            for j in range(i, n_steps):
                Y[j, :] = np.nan
            return Y[:, 0], Y
        Y[i] = y_current
    return Y[:, 0], Y

@njit(cache=True)
def hmax(V0):
    ah = Aah / np.exp((V0 - Vah) / Sah)
    bh = Abh * vtrap(V0 - Vbh, Sbh)
    denom = ah + bh
    return 0.5 if denom < 1e-12 else ah / denom

@njit(cache=True)
def n1max_fn(V0):
    an1 = Aan1 * vtrap(V0 - Van1, San1)
    bn1 = Abn1 / np.exp((V0 - Vbn1) / Sbn1)
    denom = an1 + bn1
    return 0.5 if denom < 1e-12 else an1 / denom

@njit(cache=True)
def n3max_fn(V0):
    an3 = Aan3 * vtrap(V0 - Van3, San3)
    bn3 = Abn3 / np.exp((V0 - Vbn3) / Sbn3)
    denom = an3 + bn3
    return 0.5 if denom < 1e-12 else an3 / denom

@njit(cache=True)
def ca2i0(V0, gCa):
    amax = 1.0 / (1.0 + np.exp((V0 - Va) / Sa))
    ICa = gCa * amax ** 2 * (V0 - VCa)
    return (ICa / (2.0 * F * mArea * d * pgamma)) + Car

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def get_f_decay_rate(voltage, tspan_ms, threshold=THRESHOLD_V):
    dt_s = (tspan_ms[1] - tspan_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    spike_idx, _ = find_peaks(voltage, height=threshold, distance=int(0.001 * fs))
    if len(spike_idx) < 4:
        return np.nan
    ISIs_ms = np.diff(tspan_ms[spike_idx])
    freq = 1.0 / (ISIs_ms / 1000.0)
    t_sec = (tspan_ms[spike_idx[1:]] - tspan_ms[spike_idx[1]]) / 1000.0
    if np.any(freq <= 0) or len(t_sec) < 2:
        return np.nan
    try:
        reg = linregress(t_sec, np.log(freq))
        if reg.slope >= 0:
            return np.nan
        tau_f = -1.0 / reg.slope
        return tau_f if 0.005 < tau_f < 10.0 else np.nan
    except ValueError:
        return np.nan

def get_features(V, tspan, stim_idx, thresholdV=THRESHOLD_V):
    spike_train_features = ['peak_time', 'adaptation_index', 'spike_count']
    spike_idx, _ = find_peaks(V, height=thresholdV)
    if len(spike_idx) > 5:
        trace = {
            'T': tspan, 'V': V,
            'stim_start': [tspan[stim_idx[0]]], 'stim_end': [tspan[stim_idx[-1]]]
        }
        features = efel.get_feature_values([trace], spike_train_features)
        spike_times = features[0].get('peak_time', None)
        td = (spike_times[-1] - spike_times[0]) / 1000.0
        spike_counts = (features[0].get('spike_count', None))[0]
        adaptIdx = (features[0].get('adaptation_index', None))[0]
        return td, spike_counts, adaptIdx
    else:
        return 0.0, 0.0, 0.0

def analyze_traces_to_dataframe(label_vals, tspan_np, trajectories, stim_idx, thresholdV=THRESHOLD_V):
    records = []
    for i, val in enumerate(label_vals):
        V = gaussian_filter1d(trajectories[i], 10)
        tau_f = get_f_decay_rate(V, tspan_np, threshold=thresholdV)
        td, spike_count, adapt_idx = get_features(V, tspan_np, stim_idx, thresholdV)
        records.append({
            'Value': float(val), 'Spike_Count': float(spike_count),
            'Discharge_Time': float(td), 'Tau_f': float(tau_f),
            'Adaptation_Index': float(adapt_idx)
        })
    return pd.DataFrame(records)

def evaluate_single_sweep(task_info):
    cell_idx, ch_name, ch_key, sweep_vals, multipliers, base_cond, y0, Vleak, VK_base, tspan_arr, Idata_arr, stim_idx = task_info
    records_V = []
    for val in sweep_vals:
        cond = dict(base_cond)
        if ch_key == 'VK':
            V, _ = solve_pvin_rk4(y0, tspan_arr, Idata_arr,
                                   cond['gNa'], cond['gKv1'], cond['gKv3'], cond['gCa'],
                                   cond['gSK'], cond['gleak'], cond['Btot'], Vleak, val)
        else:
            cond[ch_key] = val
            V, _ = solve_pvin_rk4(y0, tspan_arr, Idata_arr,
                                   cond['gNa'], cond['gKv1'], cond['gKv3'], cond['gCa'],
                                   cond['gSK'], cond['gleak'], cond['Btot'], Vleak, VK_base)
        records_V.append(V)

    df = analyze_traces_to_dataframe(sweep_vals, tspan_arr, records_V, stim_idx)
    df['Channel'] = ch_name
    df['Multiplier'] = multipliers
    df['Cell_ID'] = cell_idx
    return df


# 1. Load top-N WT parameters
population_df = pd.read_csv(CSV_PATH).head(N_MODELS)

# 2. Load canonical protocol
abf = pyabf.ABF(str(ABF_PATH))
abf.setSweep(8)
Vdata, Idata, tspan = abf.sweepY, abf.sweepC, abf.sweepX * 1000.0
tspan = tspan - tspan[0]
stim_idx = np.where(Idata > 0)[0]
Vdata2 = gaussian_filter1d(Vdata, 10)

V0 = Vdata2[0]
Vleak = Vdata2[0]
VK_base = VK_BASELINE

tspan_arr = np.ascontiguousarray(tspan, dtype=np.float64)
Idata_arr = np.ascontiguousarray(Idata, dtype=np.float64)


# Warm up JIT
print("Compiling Numba functions...")
dummy_y0 = np.array([V0, 0.5, 0.5, 0.5, 0.07])
_ = solve_pvin_rk4(dummy_y0, tspan_arr, Idata_arr,
                   300.0, 15.0, 180.0, 8.0, 10.0, 3.0, 80.0, Vleak, VK_base)

# 3. Assemble tasks
tasks = []
panel_list = [
    ('gNa', 'gNa'), ('gKv1', 'gKv1'), ('gKv3', 'gKv3'),
    ('gCa', 'gCa'), ('gSK', 'gSK'), ('gleak', 'gleak'),
]
if SKIP_GLEAK:
    panel_list = [p for p in panel_list if p[0] != 'gleak']

for cell_idx, row in population_df.iterrows():
    base_cond = {
        'gNa': row['gNa'], 'gKv1': row['gKv1'], 'gKv3': row['gKv3'],
        'gCa': row['gCa'], 'gSK': row['gSK'], 'gleak': row['gleak'],
        'Btot': row['Btot']
    }
    y0 = np.array([V0, hmax(V0), n1max_fn(V0), n3max_fn(V0), ca2i0(V0, base_cond['gCa'])])

    for ch_name, ch_key in panel_list:
        sweep_vals = MULTIPLIERS * base_cond[ch_key]
        tasks.append((cell_idx, ch_name, ch_key, sweep_vals, MULTIPLIERS,
                      base_cond, y0, Vleak, VK_base, tspan_arr, Idata_arr, stim_idx))

    # Btot: compressed range because buffering has a different direction of perturbations
    btot_vals = (MULTIPLIERS * BTOT_SCALE) * base_cond['Btot']
    tasks.append((cell_idx, 'Btot', 'Btot', btot_vals, MULTIPLIERS,
                  base_cond, y0, Vleak, VK_base, tspan_arr, Idata_arr, stim_idx))

    if SWEEP_VK:
        VK_vals = VK_base + np.linspace(0.0, VK_DELTA_RANGE, len(MULTIPLIERS))
        tasks.append((cell_idx, 'VK', 'VK', VK_vals, MULTIPLIERS,
                      base_cond, y0, Vleak, VK_base, tspan_arr, Idata_arr, stim_idx))

# 4. Parallel execution
all_individual_runs = []
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(evaluate_single_sweep, task): task for task in tasks}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Sweeps"):
        try:
            df_result = future.result()
            all_individual_runs.append(df_result)
        except Exception as exc:
            print(f"\n[ERROR] Task failed: {exc}")
            traceback.print_exc()

if len(all_individual_runs) == 0:
    raise RuntimeError("All parallel workers failed.")

# 5. Aggregate + save raw data
master_df = pd.concat(all_individual_runs, ignore_index=True)
os.makedirs(SAVE_DIR, exist_ok=True)
master_df.to_csv(SAVE_DIR / OUT_RAW_CSV, index=False)
print(f"💾 Raw data saved to {SAVE_DIR / OUT_RAW_CSV}")

grouped = master_df.groupby(['Channel', 'Multiplier'])
mean_df = grouped[['Tau_f', 'Discharge_Time', 'Spike_Count', 'Adaptation_Index']].mean().reset_index()
sem_df  = grouped[['Tau_f', 'Discharge_Time', 'Spike_Count', 'Adaptation_Index']].agg(sem).reset_index()

# 6. Plot
preferred_order = ['gNa', 'gKv1', 'gKv3', 'gCa', 'gSK', 'gleak', 'Btot', 'VK']
channel_order = [ch for ch in preferred_order if ch in master_df['Channel'].unique()]

metrics = [
    ('Tau_f', r'$\tau_f$', True),
    ('Discharge_Time', r'$t_d$', False),
    ('Spike_Count', 'Spike Count', False),
    ('Adaptation_Index', 'Adaptation Index', False),
]

fig, axes = plt.subplots(len(metrics), len(channel_order),
                          figsize=(3.5 * len(channel_order), 3.2 * len(metrics)))
if len(channel_order) == 1:
    axes = axes.reshape(-1, 1)

for row, (col_name, ylabel, clip_to_1) in enumerate(metrics):
    for col, ch in enumerate(channel_order):
        ax = axes[row, col]
        m_sub = mean_df[mean_df['Channel'] == ch].sort_values('Multiplier')
        s_sub = sem_df[sem_df['Channel'] == ch].sort_values('Multiplier')

        x = m_sub['Multiplier']
        y_mean = m_sub[col_name]
        y_sem  = s_sub[col_name]

        if clip_to_1:
            y_mean = np.clip(y_mean, 0, 1)

        ax.plot(x, y_mean, marker='o', color='tab:blue', label='Mean')
        ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                        color='tab:blue', alpha=0.2, label='SEM')

        if row == 0:
            ax.set_title(ch)
        if row == len(metrics) - 1:
            ax.set_xlabel('$V_K$ absolute shift' if ch == 'VK' else 'Multiplier')
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(SAVE_DIR / OUT_FIG, dpi=300)
plt.show()
