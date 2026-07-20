"""
features.py
===========

Extract summary statistics from a single voltage trace. Two categories:

  Train-level: spike_count, mean_freq_Hz, adapt_idx, tau_f_s, latency_ms,
               isi_cv, discharge_time_s

  Single-spike shape: AP_peak_mV, AP_trough_mV, AP_halfwidth_ms
                     (averaged over the first 5 spikes via eFEL)

All features that can be computed will be; any that can't (e.g. fewer than
2 spikes → no ISI metrics) come back as NaN. Downstream code is NaN-aware.

One can also add the phase-space plot as a feature. It should be as straightforward as doing a difference 2D histogram.

"""

import numpy as np
import efel
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import linregress

# eFEL needs a "spike begin" voltage threshold. -10 mV is permissive enough
# for cells with junction-potential offsets while still catching real APs.
EFEL_THRESH = -10.0
V_THRESH    = -5.0      # for our own find_peaks call
LP_CUTOFF   = 2000.0    # low-pass before spike detection, to match
LP_ORDER    = 4         # the experimental filtering chain
N_SPIKES_FOR_SHAPE = 5  # average first N spikes for shape features

efel.api.set_setting('Threshold', EFEL_THRESH)

STAT_NAMES = [
    'spike_count', 'mean_freq_Hz', 'adapt_idx', 'tau_f_s',
    'latency_ms', 'isi_cv', 'discharge_time_s',
    'AP_peak_mV', 'AP_trough_mV', 'AP_halfwidth_ms',
]


def lowpass(V, fs):
    """4th-order Butterworth low-pass at LP_CUTOFF Hz."""
    b, a = butter(LP_ORDER, LP_CUTOFF, btype='low', fs=fs)
    return filtfilt(b, a, V)


def compute_spike_shape(V, tspan_ms, stim_on_ms, stim_off_ms, n_spikes=N_SPIKES_FOR_SHAPE):
    """Average first n_spikes APs' shape features using eFEL.

    Three features:
      AP_peak_mV     : peak voltage of the AP (absolute, not relative)
      AP_trough_mV   : minimum voltage between this and the next AP (AHP trough)
      AP_halfwidth_ms: AP width at half-amplitude
    """
    out = {'AP_peak_mV': np.nan, 'AP_trough_mV': np.nan, 'AP_halfwidth_ms': np.nan}
    if not np.all(np.isfinite(V)):
        return out
    trace = {'T': np.ascontiguousarray(tspan_ms, dtype=np.float64),
             'V': np.ascontiguousarray(V, dtype=np.float64),
             'stim_start': [float(stim_on_ms)],
             'stim_end':   [float(stim_off_ms)]}
    try:
        feats = efel.getFeatureValues(
            [trace],
            ['peak_voltage', 'min_AHP_values', 'AP_duration_half_width'],
            raise_warnings=False,
        )[0]
    except Exception:
        return out

    def avg_first_n(key):
        arr = feats.get(key)
        if arr is None or len(arr) == 0:
            return np.nan
        return float(np.nanmean(arr[:n_spikes]))

    out['AP_peak_mV']      = avg_first_n('peak_voltage')
    out['AP_trough_mV']    = avg_first_n('min_AHP_values')
    out['AP_halfwidth_ms'] = avg_first_n('AP_duration_half_width')
    return out


def compute_stats(V, tspan_ms, stim_on_ms, stim_off_ms, threshold=V_THRESH):
    """Compute all 10 features for one trace.

    Robust to missing values: if the trace doesn't spike, only spike_count is
    set (to 0); the rest stay NaN. If it spikes once, latency is set but ISI-
    derived features remain NaN. And so on.
    """
    nans = {k: np.nan for k in STAT_NAMES}
    if not np.all(np.isfinite(V)):
        return nans

    # Filter then detect peaks (matches what the experimental pipeline does)
    dt_s = (tspan_ms[1] - tspan_ms[0]) / 1000.0
    fs = 1.0 / dt_s
    V_f = lowpass(V, fs)
    spike_idx, _ = find_peaks(V_f, height=threshold, distance=int(0.001 * fs))
    n = len(spike_idx)

    out = dict(nans)
    out['spike_count'] = float(n)

    if n >= 1:
        out['latency_ms'] = float(tspan_ms[spike_idx[0]] - stim_on_ms)

    if n >= 2:
        ISIs_ms = np.diff(tspan_ms[spike_idx])
        out['mean_freq_Hz'] = float(1000.0 / np.mean(ISIs_ms))
        out['isi_cv'] = float(np.std(ISIs_ms) / np.mean(ISIs_ms)) \
                       if np.mean(ISIs_ms) > 0 else np.nan
        out['discharge_time_s'] = float(
            (tspan_ms[spike_idx[-1]] - tspan_ms[spike_idx[0]]) / 1000.0)

    if n >= 4:
        ISIs_ms = np.diff(tspan_ms[spike_idx])

        # adaptation index = mean of (ISI[i+1] - ISI[i]) / (ISI[i+1] + ISI[i])
        # Positive = slowing down (typical PV cells: ~0.01–0.02)
        diffs = np.diff(ISIs_ms)
        sums  = ISIs_ms[1:] + ISIs_ms[:-1]
        mask = sums > 0
        if mask.any():
            out['adapt_idx'] = float(np.mean(diffs[mask] / sums[mask]))

        # tau_f: time constant of frequency decay, fit log(freq) vs time
        # Only well-defined if frequency is genuinely decaying.
        freq = 1.0 / (ISIs_ms / 1000.0)
        t_sec = (tspan_ms[spike_idx[1:]] - tspan_ms[spike_idx[1]]) / 1000.0
        if np.all(freq > 0) and len(t_sec) >= 2:
            try:
                res = linregress(t_sec, np.log(freq))
                if res.slope < 0:
                    tau = -1.0 / res.slope
                    if 0.005 < tau < 1.0:
                        out['tau_f_s'] = float(tau)
            except ValueError:
                pass

    # Spike-shape features (use unfiltered V; eFEL applies its own threshold)
    if n >= 1:
        out.update(compute_spike_shape(V, tspan_ms, stim_on_ms, stim_off_ms))

    return out
