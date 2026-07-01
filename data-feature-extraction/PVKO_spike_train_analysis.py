#%%
"""
PVKO Spike Train Analysis Pipeline
==================================
Author: Rian Fritz D. Jalandoni


This module provides a standardized pipeline for extracting and analyzing 
electrophysiological intrinsic properties and spike-train features from Axon 
Binary Format (.abf) files. It supports feature extraction via eFEL, localized 
frequency decay fitting, non-parametric statistical testing, and publication-quality 
visualization exports.
"""

from itertools import combinations
from pathlib import Path
import warnings

import efel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import scikit_posthocs as sp
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import mannwhitneyu, kruskal, linregress
from tqdm import tqdm

# Suppress non-critical user and library warnings for stable console outputs
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. GLOBAL CONFIGURATIONS & PARAMETERS
# ==============================================================================

# Global directory paths for experimental metadata, raw ABF traces, and output figures
ROOT_DIR = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/')
ABF_DIR = ROOT_DIR / 'Abf Traces'
SAVE_DIR = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/plots/PVKO/')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Experimental experimental groups and corresponding publication-grade color palette
GROUPS = ['WT', 'KO']
GROUP_COLORS = {'WT': 'darkorange', 'KO': 'goldenrod'}

# Signal processing and feature extraction constants
V_THRESH = -5          # Action potential detection threshold (mV)
CUTOFF_FREQ = 1000     # Lowpass filter cutoff frequency (Hz)
FS_DEFAULT = 20000     # Default sampling frequency (Hz)
BUTTER_ORDER = 4       # Butterworth filter order

# Initialize eFEL global configurations
efel.api.set_setting('Threshold', V_THRESH)

# Target electrophysiological features extracted via eFEL
SPIKE_TRAIN_FEATURES = [
    'mean_frequency', 'ISI_CV', 'adaptation_index',
    'time_to_first_spike', 'spike_count', 'inv_ISI_values',
    'depolarized_base', 'ohmic_input_resistance', 'peak_time'
]

# ==============================================================================
# 2. STATISTICAL & UTILITY FUNCTIONS
# ==============================================================================

def safe_get(features_dict, key, default=np.nan):
    """Safely extracts a single feature scalar from the eFEL output dictionary.

    Args:
        features_dict (dict): Dictionary of features returned by eFEL.
        key (str): Target feature name key.
        default (float, optional): Value to return if key is missing/null. Defaults to np.nan.

    Returns:
        float: First element of the feature array, or the default value.
    """
    val = features_dict.get(key)
    if val is None:
        return default
    try:
        return val[0]
    except (TypeError, IndexError):
        return default


def get_significance_label(p_value):
    """Translates a p-value into standardized academic significance notation.

    Args:
        p_value (float): Computed p-value from a statistical test.

    Returns:
        str: Symbol representing the alpha significance bracket.
    """
    if p_value < 0.0005: 
        return "***"
    if p_value < 0.005:  
        return "**"
    if p_value < 0.05:   
        return "*"
    return "n.s."


def run_statistical_test(df, feature, groups):
    """Executes appropriate non-parametric statistical tests between experimental groups.

    Performs a Mann-Whitney U test for two-group designs, or a Kruskal-Wallis 
    test followed by post-hoc Dunn's test for multi-group comparison.

    Args:
        df (pd.DataFrame): Calculated features dataframe.
        feature (str): The column name containing the dependent variable.
        groups (list): List of group strings to evaluate.

    Returns:
        tuple: (dict of p-values mapped to group pairs, dict of group sample sizes)
    """
    df_filt = df[df['Mouse_info'].isin(groups)].dropna(subset=[feature])
    if df_filt.empty:
        return {}, {}

    group_data = {g: df_filt[df_filt['Mouse_info'] == g][feature].values for g in groups}
    n_sizes = {g: len(v) for g, v in group_data.items()}

    valid_groups = [g for g in groups if n_sizes[g] >= 2]
    if len(valid_groups) < 2:
        return {}, n_sizes

    p_values = {}
    if len(groups) == 2:
        _, p = mannwhitneyu(group_data[groups[0]], group_data[groups[1]], alternative='two-sided')
        p_values[(groups[0], groups[1])] = p
    else:
        valid_data = [group_data[g] for g in valid_groups]
        _, p_kw = kruskal(*valid_data)
        if p_kw < 0.05:
            posthoc = sp.posthoc_dunn(df_filt, val_col=feature, group_col='Mouse_info')
            for g1, g2 in combinations(valid_groups, 2):
                if g1 in posthoc.index and g2 in posthoc.columns:
                    p_values[(g1, g2)] = posthoc.loc[g1, g2]
                    
    return p_values, n_sizes


def add_significance_brackets(ax, order, p_values, y_max, y_range, offset=0.15):
    """Draws publication-ready significance bars and markers on a given axis.

    Args:
        ax (matplotlib.axes.Axes): Target axis to draw on.
        order (list): Specified display order of categories along the x-axis.
        p_values (dict): Mapping of group tuples to their respective p-values.
        y_max (float): Maximum data point height within the feature dataset.
        y_range (float): Variability range (standard deviation) used to scale brackets.
        offset (float, optional): Baseline vertical offset factor above y_max. Defaults to 0.15.
    """
    if not p_values:
        return
    y_base = y_max + offset * y_range
    pairs = [order] if len(order) == 2 else list(combinations(order, 2))
    
    for i, (g1, g2) in enumerate(pairs):
        key = (g1, g2) if (g1, g2) in p_values else (g2, g1)
        if key not in p_values:
            continue
            
        p = p_values[key]
        x1, x2 = order.index(g1), order.index(g2)
        y_coord = y_base + i * 0.1 * y_range
        
        # Plot bracket lines
        ax.plot([x1, x1, x2, x2], [y_coord, y_coord + 0.02 * y_range, y_coord + 0.02 * y_range, y_coord], c='k', lw=1)
        # Render significance text
        ax.text((x1 + x2) / 2, y_coord + 0.025 * y_range, get_significance_label(p), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# ==============================================================================
# 3. DIGITAL SIGNAL PROCESSING & KINETIC FITTING
# ==============================================================================

def lowpass_filter(data, cutoff_freq=CUTOFF_FREQ, fs=FS_DEFAULT, order=BUTTER_ORDER):
    """Applies a zero-phase forward-backward Butterworth lowpass filter.

    Args:
        data (np.ndarray): Raw voltage time-series signal.
        cutoff_freq (float, optional): Cutoff frequency in Hz. Defaults to CUTOFF_FREQ.
        fs (float, optional): Digitization sampling rate in Hz. Defaults to FS_DEFAULT.
        order (int, optional): Filter order parameter. Defaults to BUTTER_ORDER.

    Returns:
        np.ndarray: Denoised time-series data array.
    """
    b, a = butter(order, cutoff_freq, btype='low', fs=fs)
    return filtfilt(b, a, data)


def get_f_decay_rate(voltage, tspan_ms, method=2, threshold=0.0,
                     tau_min=0.005, tau_max=10.0, min_spikes=4):
    """Calculates the adaptation time constant (tau_f) of action potential frequency decay.

    Method 1 utilizes a 3-parameter non-linear least squares fit model:
        f(t) = f_ss + (f_0 - f_ss) * exp(-t / tau_f)
        
    Method 2 utilizes a log-linearized single exponential fit model towards zero:
        ln(f) = ln(a) - t / tau_f

    Args:
        voltage (np.ndarray): Membrane voltage recording trace.
        tspan_ms (np.ndarray): Timestamps corresponding to the voltage vector (ms).
        method (int, optional): Selection between method models (1 or 2). Defaults to 2.
        threshold (float, optional): Spike detection threshold level (mV). Defaults to 0.0.
        tau_min (float, optional): Lower boundary limit for tau parameter (s). Defaults to 0.005.
        tau_max (float, optional): Upper boundary limit for tau parameter (s). Defaults to 10.0.
        min_spikes (int, optional): Minimum required spike events to execute fit. Defaults to 4.

    Returns:
        tuple: (tau_f_fit, secondary_metric), where secondary_metric represents steady-state 
               frequency (Method 1) or coefficient of determination R^2 (Method 2).
    """
    dt_s = (tspan_ms[1] - tspan_ms[0]) / 1000.0
    fs = 1.0 / dt_s

    spike_idx, _ = find_peaks(voltage, height=threshold, distance=int(0.001 * fs))
    voltage = gaussian_filter1d(voltage, 1)
    
    if len(spike_idx) < min_spikes:
        return np.nan, np.nan

    is_intervals_ms = np.diff(tspan_ms[spike_idx])
    freq = 1.0 / (is_intervals_ms / 1000.0)
    t_sec = (tspan_ms[spike_idx[1:]] - tspan_ms[spike_idx[1]]) / 1000.0

    if method == 1:
        def exp_model(t, f_0, f_ss, tau_f):
            return f_ss + (f_0 - f_ss) * np.exp(-t / tau_f)

        f_0_guess = freq[0]
        f_ss_guess = freq[-1]

        threshold_freq = f_0_guess - ((f_0_guess - f_ss_guess) * 0.632)
        tau_idx = np.where(freq <= threshold_freq)[0]
        tau_f_guess = t_sec[tau_idx[0]] if len(tau_idx) > 0 else t_sec[-1] / 3.0
        tau_f_guess = np.clip(tau_f_guess, tau_min, tau_max)

        try:
            popt, _ = curve_fit(exp_model, t_sec, freq,
                                 p0=[f_0_guess, f_ss_guess, tau_f_guess],
                                 bounds=([0, 0, tau_min], [np.inf, np.inf, tau_max]),
                                 maxfev=10000)
            f_0_fit, f_ss_fit, tau_f_fit = popt

            if (f_0_fit - f_ss_fit) > 0 and tau_min < tau_f_fit < tau_max and f_ss_fit >= 0:
                return tau_f_fit, f_ss_fit
            return np.nan, np.nan
        except (RuntimeError, ValueError):
            return np.nan, np.nan

    elif method == 2:
        if np.any(freq <= 0) or len(t_sec) < 2:
            return np.nan, np.nan

        log_freq = np.log(freq)
        try:
            result = linregress(t_sec, log_freq)
        except ValueError:
            return np.nan, np.nan

        slope = result.slope
        r_squared = result.rvalue ** 2

        if slope >= 0:
            return np.nan, np.nan

        tau_f_fit = -1.0 / slope

        if tau_min < tau_f_fit < tau_max:
            return tau_f_fit, r_squared
        return np.nan, np.nan
    else:
        raise ValueError(f"Invalid execution method parameter specified: {method!r}")


def get_sag_ratio(sweep_data, stim_idx):
    """Extracts hyperpolarization-activated sag ratio metrics using eFEL.

    Args:
        sweep_data (pyabf.ABF): Loaded instance of an ABF single sweep object.
        stim_idx (np.ndarray): Indices representing the duration of the current injection.

    Returns:
        tuple: (sag_ratio1, sag_time_constant) extracted via standard eFEL criteria.
    """
    sweep_data.setSweep(0)
    tspan_ms = 1000.0 * sweep_data.sweepX
    trace = {
        'T': tspan_ms,
        'V': sweep_data.sweepY,
        'stim_start': [tspan_ms[stim_idx[0]]],
        'stim_end': [tspan_ms[stim_idx[-1]]]
    }
    features = efel.getFeatureValues([trace], ['sag_ratio1', 'sag_time_constant'])[0]
    return safe_get(features, 'sag_ratio1'), safe_get(features, 'sag_time_constant')

# ==============================================================================
# 4. DATA PROCESSING EXECUTION PIPELINE
# ==============================================================================

def process_ephys_data(df_meta, abf_dir):
    """Iterates through experimental records to parse metadata and extract spike features.

    Args:
        df_meta (pd.DataFrame): Curated metadata file linking cells to ABF directories.
        abf_dir (Path): Base path location pointing to root ABF storage.

    Returns:
        pd.DataFrame: Compiled database containing calculated electrophysiology features.
    """
    results = []
    for row in tqdm(df_meta.itertuples(), total=len(df_meta), desc="Processing cells"):
        if row.Input != 'step' or row.Self_eval == 'omit':
            continue
            
        abf_path = abf_dir / row.Date / f"{row.Code}.abf"
        if not abf_path.exists():
            continue
            
        try:
            data_samp = pyabf.ABF(str(abf_path))
            data_samp.setSweep(8)
            
            voltage = data_samp.sweepY
            current = data_samp.sweepC
            t_ms = 1000.0 * data_samp.sweepX
            
            stim_idx = np.where(current > 0)[0]
            if len(stim_idx) == 0:
                continue
                
            sag_ratio, sag_tau = get_sag_ratio(data_samp, stim_idx)
            
            trace = {
                'T': t_ms, 'V': voltage,
                'stim_start': [t_ms[stim_idx[0]]],
                'stim_end': [t_ms[stim_idx[-1]]]
            }
            features = efel.getFeatureValues([trace], SPIKE_TRAIN_FEATURES)[0]
            spike_times = features.get('peak_time')
            
            # Adaptation metric execution utilizing Method 2
            tau_fit, r2_fit = get_f_decay_rate(voltage, t_ms, method=2)
            
            results.append({
                'Date': row.Date,
                'Code': row.Code,
                'Mouse_info': row.Mouse_info,
                'Mean Frequency': safe_get(features, 'mean_frequency'),
                'ISI_CV': safe_get(features, 'ISI_CV'),
                'Adaptation Index': safe_get(features, 'adaptation_index'),
                'Latency': safe_get(features, 'time_to_first_spike'),
                'Discharge Time': (spike_times[-1] - spike_times[0]) / 1000.0 if spike_times is not None and len(spike_times) >= 2 else np.nan,
                'Sag Ratio': sag_ratio,
                'Sag Time Constant': sag_tau,
                'Spike Count': safe_get(features, 'spike_count'),
                'Initial Frequency': safe_get(features, 'inv_ISI_values'),
                'Depolarized Base': safe_get(features, 'depolarized_base'),
                'Ohmic Input Resistance': safe_get(features, 'ohmic_input_resistance'),
                'Frequency Decay Constant': tau_fit,
                'Decay Fit R_squared': r2_fit
            })
        except Exception as err:
            print(f"File process error caught on trace execution: {row.Code} ({row.Date}): {err}")
            continue
            
    df_res = pd.DataFrame(results)
    df_res.to_excel(SAVE_DIR / "Spike_Train_Features_aKO2.xlsx", index=False)
    return df_res

# ==============================================================================
# 5. GRAPHICS COMPOSITION & ANALYSIS EXPORTS
# ==============================================================================

def plot_feature_boxplot(df, feature, ax=None, order=None, palette=None, save_fig=False):
    """Generates an individual publication-grade boxplot with overlaying datapoints.

    Args:
        df (pd.DataFrame): Dataframe containing calculated electrophysiology features.
        feature (str): Column name representing target feature property.
        ax (matplotlib.axes.Axes, optional): Target axis layout box. Defaults to None.
        order (list, optional): Categorical display alignment order. Defaults to None.
        palette (list, optional): Color list structure maps. Defaults to None.
        save_fig (bool, optional): Toggles saving figure vector files. Defaults to False.

    Returns:
        matplotlib.axes.Axes: Configured plot axis instance object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))
        standalone = True
    else:
        standalone = False
        
    df_filt = df[df['Mouse_info'].isin(order)].dropna(subset=[feature])
    if df_filt.empty:
        if standalone: plt.close()
        return ax
        
    sns.boxplot(data=df_filt, x='Mouse_info', y=feature, order=order, 
                palette=palette, showfliers=False, width=0.4, ax=ax, zorder=500)
    sns.stripplot(data=df_filt, x='Mouse_info', y=feature, order=order, 
                  palette=palette, jitter=True, edgecolor='black', linewidth=0.4, ax=ax, zorder=100)
    
    p_vals, _ = run_statistical_test(df, feature, order)
    y_max, y_std = df_filt[feature].max(), df_filt[feature].std()
    add_significance_brackets(ax, order, p_vals, y_max, max(1e-9, y_std))
    
    ax.set_ylabel(feature, fontsize=14)
    ax.set_xlabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if standalone:
        plt.tight_layout()
        if save_fig:
            plt.savefig(SAVE_DIR / f"{feature}_SpikeTrain_PVKO.png", dpi=300, transparent=True)
        plt.show()
    return ax


def create_summary_grid(df, order, palette, ncols=6, nrows=2, save_fig=False):
    """Combines all processed feature graphs into a unified compilation matrix grid.

    Args:
        df (pd.DataFrame): Features dataframe reference object.
        order (list): Categorical sorting arrangement configuration.
        palette (list): Explicit color assignment keys mapping.
        ncols (int, optional): Explicit horizontal matrix size definition. Defaults to 6.
        nrows (int, optional): Explicit vertical matrix size definition. Defaults to 2.
        save_fig (bool, optional): Toggles saving vector file format. Defaults to False.
    """
    features = [c for c in df.columns if c not in ['Date', 'Code', 'Mouse_info'] and not df[c].isna().all()]
    if not features:
        print("Empty array vector set. Skipping execution.")
        return
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        if i >= len(axes):
            break
        plot_feature_boxplot(df, feat, ax=axes[i], order=order, palette=palette)
        axes[i].set_title(feat, fontsize=14, pad=10)
        
    for j in range(len(features), len(axes)):
        axes[j].axis('off')
        
    fig.suptitle("Spike-Train Feature Summary", fontsize=16, fontweight='bold', y=1.01)
    if save_fig:
        plt.savefig(SAVE_DIR / "Spike_Train_Feature_Summary_Grid_PVKO.png", dpi=300, transparent=True, bbox_inches='tight')
    plt.show()


def print_significance_summary(df, order):
    """Prints a clear text summary of the statistical results to the console.

    Args:
        df (pd.DataFrame): Features dataframe reference object.
        order (list): Categorical comparison definitions list.
    """
    features = [c for c in df.columns if c not in ['Date', 'Code', 'Mouse_info'] and not df[c].isna().all()]
    test_type = "Mann-Whitney U" if len(order) == 2 else "Kruskal-Wallis + Dunn's Post-Hoc"
    
    print(f"\n{'='*80}")
    print(f"SPIKE-TRAIN FEATURE SIGNIFICANCE SUMMARY ({test_type})")
    print(f"Significance thresholds: *** p<0.0005 | ** p<0.005 | * p<0.05 | n.s. >= 0.05")
    print(f"{'='*80}")
    
    for feat in features:
        p_vals, n_sizes = run_statistical_test(df, feat, order)
        if not p_vals:
            continue
            
        sizes = ", ".join(f"{g} (n={n_sizes[g]})" for g in order if n_sizes.get(g, 0) > 0)
        print(f"\n{feat}  |  Sample Sizes: {sizes}")
        
        for (g1, g2), p in p_vals.items():
            print(f"  {g1} vs {g2:<6}: p = {p:.6f}  ({get_significance_label(p)})")
            
    print(f"\n{'='*80}")

# ==============================================================================
# 6. PIPELINE RUNTIME EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Load metadata file containing cellular recording metadata
    EPHYS_meta = pd.read_excel(
        ROOT_DIR / "EPHYS_data_astrocytes.xlsx",
        converters={'Date': str, 'Cell_Number': str, 'Code': str}
    )
    
    # Standardize column header strings to prevent structural spacing index runtime breaks
    EPHYS_meta.columns = EPHYS_meta.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')
    
    # Run processing pipeline execution
    df_spike_train = process_ephys_data(EPHYS_meta, ABF_DIR)
    
    if df_spike_train.empty:
        print("Data extraction failed. Returned cell array matrix contains 0 records.")
    else:
        print("Feature extraction successfully processed.")
        
        # Save individual property graphs
        for column_feature in df_spike_train.columns:
            if column_feature not in ['Date', 'Code', 'Mouse_info'] and not df_spike_train[column_feature].isna().all():
                plot_feature_boxplot(
                    df_spike_train, 
                    column_feature, 
                    order=GROUPS, 
                    palette=[GROUP_COLORS[g] for g in GROUPS], 
                    save_fig=True
                )
                
        # Save structural multi-panel summary canvas grid
        create_summary_grid(
            df_spike_train, 
            GROUPS, 
            [GROUP_COLORS[g] for g in GROUPS], 
            ncols=6, 
            nrows=2,
            save_fig=True
        )
        
        # Display statistical analysis report summary to terminal console
        print_significance_summary(df_spike_train, GROUPS)