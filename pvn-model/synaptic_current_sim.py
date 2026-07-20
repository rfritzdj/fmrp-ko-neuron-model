#%%
import numpy as np
import pyabf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import linregress
from numba import njit
from tqdm import tqdm

savePath='C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/'


# --- Baseline Parameters ---
VNa, VK, VCa = 58.0, -80.0, 68.0
Cm, pgamma, KD, F = 30.0, 0.01, 0.1, 0.0964853321    
mArea, d, Car = 3000.0, 0.1, 0.07

# Gating Parameters
Vm, Sm = -20.0, -7.0
Aah, Sah, Vah = 0.0025, 10.0, 18.4
Abh, Sbh, Vbh = 0.094, -5.5, -31.0
Aan1, Van1, San1 = 0.002, -36.0, -9.0
Abn1, Vbn1, Sbn1 = 0.017, -36.75, 6.785
Aan3, Van3, San3 = 3.2, 96.0, -12.6
Abn3, Vbn3, Sbn3 = 0.34, -36.0, 13.965
Va, Sa = 3.5, -11.4
nk, ksk = 5.0, 0.8

E_syn = 0.0  
tau_rise_AMPA = 0.2   
tau_rise_NMDA = 5.0   
Mg_ext = 1.0       


G_AMPA_DEFAULT = 10.0
G_NMDA_DEFAULT = 5.0
TAU_AMPA_DEFAULT = 2.0
TAU_NMDA_DEFAULT = 50.0
PF_AMPA_DEFAULT = 0.03
PF_NMDA_DEFAULT = 0.1


PLOT_SANITY_CHECK = False
SANITY_PLOT_STRIDE = 1   # set >1 to only plot every Nth (r,c) cell, e.g. 3 -> ~1/9 of cells

FREQ_DECAY_METHOD = 2   # 1 = original f_ss-anchored nonlinear fit, 2 = eFEL-style log-linregress decay-to-zero


SYN_ONSET_DELAY_MS = 0  

@njit
def calc_K(tau_rise, tau_decay):
    if tau_decay <= tau_rise: 
        return 1.0
    t_peak = (tau_rise * tau_decay) / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    K = 1.0 / (np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise))
    return K


@njit
def PVIN_HH_deriv(y, t, Iapp, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA, K_AMPA, K_NMDA,
                  Pf_AMPA, Pf_NMDA,
                  gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak):
    V, h, n1, n3, ca2i = y[0], y[1], y[2], y[3], y[4]
    
    mmax = 1.0 / (1.0 + np.exp((V - Vm) / Sm))
    ah = Aah / np.exp((V - Vah) / Sah)
    bh = Abh * (V - Vbh) / (1.0 - np.exp((V - Vbh) / Sbh))
    INa = gNa * (mmax ** 3) * h * (V - VNa)

    an1 = Aan1 * (V - Van1) / (1.0 - np.exp((V - Van1) / San1))
    bn1 = Abn1 / np.exp((V - Vbn1) / Sbn1)
    IKv1 = gKv1 * (n1 ** 4) * (V - VK)

    an3 = Aan3 * (V - Van3) / (1.0 - np.exp((V - Van3) / San3))
    bn3 = Abn3 / np.exp((V - Vbn3) / Sbn3)
    IKv3 = gKv3 * (n3 ** 2) * (V - VK)

    amax = 1.0 / (1.0 + np.exp((V - Va) / Sa))
    ICa = gCa * (amax ** 2) * (V - VCa)

    k_sk = (ca2i ** nk) / (ksk ** nk + ca2i ** nk)
    ISK = gSK * k_sk * (V - VK)

    Ileak = gleak * (V - Vleak)

    I_AMPA = 0.0
    I_NMDA = 0.0
    
    if t >= t_syn:
        dt_syn = t - t_syn
        exp_AMPA = np.exp(-dt_syn / t_decay_AMPA) - np.exp(-dt_syn / tau_rise_AMPA)
        I_AMPA = g_AMPA * K_AMPA * exp_AMPA * (V - E_syn)
        
        exp_NMDA = np.exp(-dt_syn / t_decay_NMDA) - np.exp(-dt_syn / tau_rise_NMDA)
        mg_block = 1.0 / (1.0 + np.exp(-0.062 * V) * (Mg_ext / 3.57))
        I_NMDA = g_NMDA * K_NMDA * mg_block * exp_NMDA * (V - E_syn)

    Isyn_total = I_AMPA + I_NMDA

    dVdt = (-Ileak - INa - IKv1 - IKv3 - ICa - ISK + Iapp - Isyn_total) / Cm
    dhdt = ah * (1.0 - h) - bh * h
    dn1dt = an1 * (1.0 - n1) - bn1 * n1
    dn3dt = an3 * (1.0 - n3) - bn3 * n3
    
    Ca_influx = -ICa - (Pf_AMPA * I_AMPA) - (Pf_NMDA * I_NMDA)
    dCadt = (Ca_influx / (2.0 * F * mArea * d) - pgamma * (ca2i - Car)) / (1.0 + Bt / KD)

    return np.array([dVdt, dhdt, dn1dt, dn3dt, dCadt])

# --- Dynamic RK4 Solver ---
# *** CHANGED: Pf_AMPA, Pf_NMDA threaded through as explicit arguments ***
@njit
def solve_pvin_rk4(y0, tspan, Idata, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA,
                   Pf_AMPA, Pf_NMDA,
                   gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak):
    n_steps = len(tspan)
    dt = tspan[1] - tspan[0]
    
    K_AMPA = calc_K(tau_rise_AMPA, t_decay_AMPA)
    K_NMDA = calc_K(tau_rise_NMDA, t_decay_NMDA)
    
    Y = np.zeros((n_steps, 5))
    Y[0] = y0
    y_current = np.copy(y0)
    
    for i in range(1, n_steps):
        t0 = tspan[i-1]
        t1 = tspan[i]
        t_half = t0 + 0.5 * dt
        I0, I1 = Idata[i-1], Idata[i]
        I_half = 0.5 * (I0 + I1)
        
        k1 = PVIN_HH_deriv(y_current, t0, I0, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA, K_AMPA, K_NMDA, Pf_AMPA, Pf_NMDA, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak)
        k2 = PVIN_HH_deriv(y_current + 0.5 * dt * k1, t_half, I_half, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA, K_AMPA, K_NMDA, Pf_AMPA, Pf_NMDA, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak)
        k3 = PVIN_HH_deriv(y_current + 0.5 * dt * k2, t_half, I_half, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA, K_AMPA, K_NMDA, Pf_AMPA, Pf_NMDA, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak)
        k4 = PVIN_HH_deriv(y_current + dt * k3, t1, I1, t_syn, g_AMPA, g_NMDA, t_decay_AMPA, t_decay_NMDA, K_AMPA, K_NMDA, Pf_AMPA, Pf_NMDA, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak)
        
        y_current = y_current + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        Y[i] = y_current
        
    return Y[:, 0]

def get_f_decay_rate(voltage, tspan_ms, threshold=-10.0,
                     tau_min=0.005, tau_max=1.0, min_spikes=4,
                     method=None):

    if method is None:
        method = FREQ_DECAY_METHOD

    dt_s = (tspan_ms[1] - tspan_ms[0]) / 1000.0
    fs = 1.0 / dt_s

    spike_idx, _ = find_peaks(voltage, height=threshold, distance=int(0.001 * fs))

    if len(spike_idx) < min_spikes:
        return np.nan, np.nan, len(spike_idx), spike_idx

    ISIs_ms = np.diff(tspan_ms[spike_idx])
    freq = 1.0 / (ISIs_ms / 1000.0)

    t_sec = (tspan_ms[spike_idx[1:]] - tspan_ms[spike_idx[1]]) / 1000.0

    if method == 1:
        # ---- unchanged ----
        f_0_fixed = freq[0]
        f_ss_fixed = freq[-1]

        def exp_model_tau_only(t, tau_f):
            return f_ss_fixed + (f_0_fixed - f_ss_fixed) * np.exp(-t / tau_f)

        threshold_freq = f_0_fixed - ((f_0_fixed - f_ss_fixed) * 0.632)
        tau_idx = np.where(freq <= threshold_freq)[0]
        tau_f_guess = t_sec[tau_idx[0]] if len(tau_idx) > 0 else t_sec[-1] / 3.0
        tau_f_guess = np.clip(tau_f_guess, tau_min, tau_max)

        try:
            popt, _ = curve_fit(exp_model_tau_only, t_sec, freq,
                                p0=[tau_f_guess],
                                bounds=([tau_min], [tau_max]),
                                maxfev=10000)
            tau_f_fit = popt[0]

            rail_margin = 0.05
            rail_width = (tau_max - tau_min) * rail_margin
            near_lower_rail = tau_f_fit < (tau_min + rail_width)
            near_upper_rail = tau_f_fit > (tau_max - rail_width)

            if tau_min < tau_f_fit < tau_max and not near_lower_rail and not near_upper_rail:
                return tau_f_fit, f_ss_fixed, len(spike_idx), spike_idx
            return np.nan, np.nan, len(spike_idx), spike_idx

        except (RuntimeError, ValueError):
            return np.nan, np.nan, len(spike_idx), spike_idx

    elif method == 2:
        # eFEL-style pure exponential decay to zero, f(t) = a*exp(-t/tau_f),
        # linearized as ln(f) = ln(a) - t/tau_f.
        #
        # CAPPING POLICY (changed):
        #   - Slope >= 0 (no decay)            -> tau_f = tau_max  (treat as "barely decays")
        #   - Fitted tau_f >  tau_max          -> clipped to tau_max
        #   - Fitted tau_f <  tau_min          -> clipped to tau_min
        #   - Genuinely undefined inputs       -> NaN
        #     (too few spikes, non-positive freq, < 2 timepoints, regression error)
        #
        # The returned second value is R^2 of the log-linear regression. For
        # capped fits (slope >= 0, or tau_f outside [tau_min, tau_max]) the R^2
        # of the raw regression is still passed through so the caller can
        # detect and downweight non-trustworthy fits if needed.

        if np.any(freq <= 0) or len(t_sec) < 2:
            return np.nan, np.nan, len(spike_idx), spike_idx

        log_freq = np.log(freq)
        try:
            reg = linregress(t_sec, log_freq)
        except ValueError:
            return np.nan, np.nan, len(spike_idx), spike_idx

        slope = reg.slope
        r_squared = reg.rvalue ** 2

        if slope >= 0:
            # No detectable decay -> represent as the slowest allowed tau.
            return tau_max, r_squared, len(spike_idx), spike_idx

        tau_f_fit = -1.0 / slope

        # Clip into [tau_min, tau_max] instead of rejecting.
        tau_f_capped = float(np.clip(tau_f_fit, tau_min, tau_max))
        return tau_f_capped, r_squared, len(spike_idx), spike_idx

    else:
        raise ValueError(f"FREQ_DECAY_METHOD/method must be 1 or 2, got {method!r}")

def extract_metrics(V, tspan, thresholdV=-10.0):
    peaks, _ = find_peaks(V, height=thresholdV)
    spike_count = len(peaks)

    if spike_count == 0:
        return 0.0, 0.0, 0.0, np.nan, 0.0, peaks
    elif spike_count == 1:
        return 1.0, 0.0, 0.0, np.nan, 1.0, peaks

    spike_times = tspan[peaks]
    discharge_time = (spike_times[-1] - spike_times[0]) / 1000.0

    mean_freq = 1.0 / np.mean(np.diff(spike_times) / 1000.0)

    tau_f, _, n_spikes_for_fit, _ = get_f_decay_rate(V, tspan, threshold=thresholdV)

    return float(spike_count), discharge_time, mean_freq, tau_f, float(n_spikes_for_fit), peaks

# --- Steady State Functions ---
@njit
def hmax(V0):
    ah = Aah / np.exp((V0 - Vah) / Sah)
    bh = Abh * (V0 - Vbh) / (1.0 - np.exp((V0 - Vbh) / Sbh))
    return ah / (ah + bh)
@njit
def n1max(V0):
    an1 = Aan1 * (V0 - Van1) / (1.0 - np.exp((V0 - Van1) / San1))
    bn1 = Abn1 / np.exp((V0 - Vbn1) / Sbn1)
    return an1 / (an1 + bn1)
@njit
def n3max(V0):
    an3 = Aan3 * (V0 - Van3) / (1.0 - np.exp((V0 - Van3) / San3))
    bn3 = Abn3 / np.exp((V0 - Vbn3) / Sbn3)
    return an3 / (an3 + bn3)
@njit
def ca2i0(V0, gCa):
    amax = 1.0 / (1.0 + np.exp((V0 - Va) / Sa))
    ICa = gCa * amax**2 * (V0 - VCa)
    return (ICa / (2.0 * F * mArea * d * pgamma)) + Car

# =============================================================================
# --- Main Simulation Execution ---
# =============================================================================
abfDataPath = 'C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/Abf Traces/'
date_samp, code_samp = '20230622', '0008'

abf = pyabf.ABF(f"{abfDataPath}/{date_samp}/{code_samp}.abf")
abf.setSweep(7)
Vdata, Idata = abf.sweepY, abf.sweepC
tspan = 1000.0 * abf.sweepX

Vdata2 = gaussian_filter1d(Vdata, 10)
V0 = Vleak = Vdata2[0]

gNa, gKv1, gKv3, gCa, gSK, gleak, Btot = 300.0, 15.0, 180.0, 8.0, 10.0, 3.0, 92.0
stim_idx = np.where(Idata > 0)[0]

t_step_onset = tspan[stim_idx[0]]
t_syn = t_step_onset + SYN_ONSET_DELAY_MS

y0 = np.array([V0, hmax(V0), n1max(V0), n3max(V0), ca2i0(V0, gCa)])

grid_res = 50
tau_AMPA_vals = np.linspace(TAU_AMPA_DEFAULT, 50.0,  grid_res)   # 2.0 -> 50.0
tau_NMDA_vals = np.linspace(TAU_NMDA_DEFAULT, 200.0, grid_res)   # 50.0 -> 120.0
g_AMPA_vals   = np.linspace(G_AMPA_DEFAULT,   30.0,  grid_res)   # 10.0 -> 30.0
g_NMDA_vals   = np.linspace(G_NMDA_DEFAULT,   30.0,  grid_res)   # 5.0  -> 30.0
Pf_NMDA_vals  = np.linspace(PF_NMDA_DEFAULT,  0.9,   grid_res)   # 0.1  -> 0.9
Pf_AMPA_vals  = np.linspace(PF_AMPA_DEFAULT,  0.6,   grid_res)   # 0.03 -> 0.9

# Matrices to hold feature maps (5 metrics per case: spike_count, discharge_time,
# mean_freq, tau_f, n_spikes_for_fit). Now 6 cases total.
results = {i: np.zeros((5, grid_res, grid_res)) for i in range(1, 7)}

# Pre-allocate 3D tensors for traces [Row, Col, TimeStep] for every case,
# so the sanity-check plotting below has a trace to draw for every case.
V_traces = {i: np.zeros((grid_res, grid_res, len(tspan)), dtype=np.float32) for i in range(1, 7)}
# Store detected spike indices per cell per case for the sanity-check overlay.
spike_indices = {i: [[None]*grid_res for _ in range(grid_res)] for i in range(1, 7)}

print(f"Synaptic onset delay relative to step current: {SYN_ONSET_DELAY_MS} ms")
print(f"Frequency decay rate fitting method: {FREQ_DECAY_METHOD} "
      f"({'f_ss-anchored nonlinear fit' if FREQ_DECAY_METHOD == 1 else 'eFEL-style log-linregress decay-to-zero'})")
print(f"Fixed defaults -> g_AMPA={G_AMPA_DEFAULT}, g_NMDA={G_NMDA_DEFAULT}, "
      f"tau_AMPA={TAU_AMPA_DEFAULT}, tau_NMDA={TAU_NMDA_DEFAULT}, "
      f"Pf_AMPA={PF_AMPA_DEFAULT}, Pf_NMDA={PF_NMDA_DEFAULT}")
print("Compiling functions and launching sweeps...")

for r in range(grid_res):
    for c in range(grid_res):
        # Case 1: Constant tau_NMDA (50ms) | Sweep tau_AMPA (Y) vs g_NMDA (X)
        V_1 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             G_AMPA_DEFAULT, g_NMDA_vals[c], tau_AMPA_vals[r], TAU_NMDA_DEFAULT,
                             PF_AMPA_DEFAULT, PF_NMDA_DEFAULT,
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m1 = extract_metrics(V_1, tspan)
        results[1][:, r, c] = m1[:5]
        V_traces[1][r, c, :] = V_1.astype(np.float32)
        spike_indices[1][r][c] = m1[5]

        # Case 2: Constant tau_NMDA (50ms) | Sweep tau_AMPA (Y) vs g_AMPA (X)
        V_2 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             g_AMPA_vals[c], G_NMDA_DEFAULT, tau_AMPA_vals[r], TAU_NMDA_DEFAULT,
                             PF_AMPA_DEFAULT, PF_NMDA_DEFAULT,
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m2 = extract_metrics(V_2, tspan)
        results[2][:, r, c] = m2[:5]
        V_traces[2][r, c, :] = V_2.astype(np.float32)
        spike_indices[2][r][c] = m2[5]

        # Case 3: Constant tau_AMPA (2ms)  | Sweep tau_NMDA (Y) vs g_NMDA (X)
        V_3 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             G_AMPA_DEFAULT, g_NMDA_vals[c], TAU_AMPA_DEFAULT, tau_NMDA_vals[r],
                             PF_AMPA_DEFAULT, PF_NMDA_DEFAULT,
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m3 = extract_metrics(V_3, tspan)
        results[3][:, r, c] = m3[:5]
        V_traces[3][r, c, :] = V_3.astype(np.float32)
        spike_indices[3][r][c] = m3[5]

        # Case 4: Constant tau_AMPA (2ms)  | Sweep tau_NMDA (Y) vs g_AMPA (X)
        V_4 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             g_AMPA_vals[c], G_NMDA_DEFAULT, TAU_AMPA_DEFAULT, tau_NMDA_vals[r],
                             PF_AMPA_DEFAULT, PF_NMDA_DEFAULT,
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m4 = extract_metrics(V_4, tspan)
        results[4][:, r, c] = m4[:5]
        V_traces[4][r, c, :] = V_4.astype(np.float32)
        spike_indices[4][r][c] = m4[5]

        # *** NEW Case 5: Constant g_AMPA/g_NMDA/tau_AMPA | Sweep tau_NMDA (Y) vs Pf_NMDA (X) ***
        V_5 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             G_AMPA_DEFAULT, G_NMDA_DEFAULT, TAU_AMPA_DEFAULT, tau_NMDA_vals[r],
                             PF_AMPA_DEFAULT, Pf_NMDA_vals[c],
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m5 = extract_metrics(V_5, tspan)
        results[5][:, r, c] = m5[:5]
        V_traces[5][r, c, :] = V_5.astype(np.float32)
        spike_indices[5][r][c] = m5[5]

        # *** NEW Case 6: Constant g_AMPA/g_NMDA/tau_NMDA | Sweep tau_AMPA (Y) vs Pf_AMPA (X) ***
        V_6 = solve_pvin_rk4(y0, tspan, Idata, t_syn,
                             G_AMPA_DEFAULT, G_NMDA_DEFAULT, tau_AMPA_vals[r], TAU_NMDA_DEFAULT,
                             Pf_AMPA_vals[c], PF_NMDA_DEFAULT,
                             gNa, gKv1, gKv3, gCa, gSK, gleak, Btot, Vleak)
        m6 = extract_metrics(V_6, tspan)
        results[6][:, r, c] = m6[:5]
        V_traces[6][r, c, :] = V_6.astype(np.float32)
        spike_indices[6][r][c] = m6[5]

# =============================================================================
# --- Plotting the Heatmaps (Synchronized Scales) ---
# =============================================================================
fig, axes = plt.subplots(6, 5, figsize=(24, 27))
metrics_labels = ['Spike Count', 'Discharge Time (s)', 'Mean Frequency (Hz)', 'Freq Decay Rate (s)', 'N Spikes Used in Tau Fit']
cmaps = ['viridis', 'plasma', 'magma', 'hot', 'cividis']

# Precompute global min/max limits for each feature column across all 6 cases
vlims = {}
for col in range(5):
    col_data = [results[case_idx][col, :, :] for case_idx in range(1, 7)]
    vlims[col] = (np.nanmin(col_data), np.nanmax(col_data))

case_meta = {
    1: {'title': 'Case 1: Constant $\\tau_{NMDA}$ (50ms)', 'ylabel': '$\\tau_{AMPA}$ (ms)', 'xlabel': '$g_{NMDA}$ (nS)', 'x_vec': g_NMDA_vals, 'y_vec': tau_AMPA_vals},
    2: {'title': 'Case 2: Constant $\\tau_{NMDA}$ (50ms)', 'ylabel': '$\\tau_{AMPA}$ (ms)', 'xlabel': '$g_{AMPA}$ (nS)', 'x_vec': g_AMPA_vals, 'y_vec': tau_AMPA_vals},
    3: {'title': 'Case 3: Constant $\\tau_{AMPA}$ (2ms)', 'ylabel': '$\\tau_{NMDA}$ (ms)', 'xlabel': '$g_{NMDA}$ (nS)', 'x_vec': g_NMDA_vals, 'y_vec': tau_NMDA_vals},
    4: {'title': 'Case 4: Constant $\\tau_{AMPA}$ (2ms)', 'ylabel': '$\\tau_{NMDA}$ (ms)', 'xlabel': '$g_{AMPA}$ (nS)', 'x_vec': g_AMPA_vals, 'y_vec': tau_NMDA_vals},
    5: {'title': 'Case 5: Constant $g_{AMPA}/g_{NMDA}/\\tau_{AMPA}$', 'ylabel': '$\\tau_{NMDA}$ (ms)', 'xlabel': '$Pf_{NMDA}$', 'x_vec': Pf_NMDA_vals, 'y_vec': tau_NMDA_vals},
    6: {'title': 'Case 6: Constant $g_{AMPA}/g_{NMDA}/\\tau_{NMDA}$', 'ylabel': '$\\tau_{AMPA}$ (ms)', 'xlabel': '$Pf_{AMPA}$', 'x_vec': Pf_AMPA_vals, 'y_vec': tau_AMPA_vals},
}

for row in tqdm(range(6)):
    case_idx = row + 1
    meta = case_meta[case_idx]
    
    for col in range(5):
        ax = axes[row, col]
        matrix_to_plot = results[case_idx][col, :, :]
        
        im = ax.pcolormesh(meta['x_vec'], meta['y_vec'], matrix_to_plot, 
                            cmap=cmaps[col], shading='auto',
                            vmin=vlims[col][0], vmax=vlims[col][1])
        
        if col == 1:
            ax.set_title(f"{meta['title']}\n\n{metrics_labels[col]}", fontsize=12, fontweight='bold')
        else:
            ax.set_title(metrics_labels[col], fontsize=11)
            
        ax.set_xlabel(meta['xlabel'], fontsize=10)
        ax.set_ylabel(meta['ylabel'], fontsize=10)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=9)

fig.suptitle(f"Synaptic onset delayed by {SYN_ONSET_DELAY_MS} ms relative to step current", 
             fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()


# Save results functionality (Includes full trace arrays for all 6 cases)
delay_tag = f"delay{int(SYN_ONSET_DELAY_MS)}ms"
np.savez_compressed(savePath+f'glutamate_spillage_results_150_V2_{delay_tag}_50.npz', 
                    case1=results[1], case2=results[2], 
                    case3=results[3], case4=results[4],
                    case5=results[5], case6=results[6],
                    V1_traces=V_traces[1], V2_traces=V_traces[2],
                    V3_traces=V_traces[3], V4_traces=V_traces[4],
                    V5_traces=V_traces[5], V6_traces=V_traces[6],
                    g_AMPA=g_AMPA_vals, g_NMDA=g_NMDA_vals,
                    tau_AMPA=tau_AMPA_vals, tau_NMDA=tau_NMDA_vals,
                    Pf_AMPA=Pf_AMPA_vals, Pf_NMDA=Pf_NMDA_vals,
                    syn_onset_delay_ms=SYN_ONSET_DELAY_MS)
print(f"Data successfully saved with delay tag '{delay_tag}'.")
print("Note: results arrays now have 5 rows per case:")
print("  [0]=spike_count, [1]=discharge_time, [2]=mean_freq, [3]=tau_f, [4]=n_spikes_used_in_tau_fit")
print(f"  tau_f ([3]) was fit using FREQ_DECAY_METHOD={FREQ_DECAY_METHOD} "
      f"({'f_ss-anchored nonlinear fit' if FREQ_DECAY_METHOD == 1 else 'eFEL-style log-linregress decay-to-zero'}).")
print("Cases 1-4: synaptic conductance/kinetics sweeps (unchanged design).")
print("Case 5: tau_NMDA vs Pf_NMDA (Ca-influx fraction), g_AMPA/g_NMDA/tau_AMPA fixed at defaults.")
print("Case 6: tau_AMPA vs Pf_AMPA (Ca-influx fraction), g_AMPA/g_NMDA/tau_NMDA fixed at defaults.")