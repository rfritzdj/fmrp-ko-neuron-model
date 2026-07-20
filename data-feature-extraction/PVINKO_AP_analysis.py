#%%
"""
PVINKO Action Potential (AP) Tracking & Kinetics Pipeline
Author: Rian Fritz D. Jalandoni

This pipeline processes raw multi-sweep patch-clamp recordings to track kinetic 
modifications across successive action potentials (1st, 3rd, and 5th spikes). 
It integrates automated threshold detection, non-parametric multi-group statistics, 
and exports vector-grade figures alongside detailed summary tables.
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
from matplotlib.patches import Patch
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kruskal, mannwhitneyu
from tqdm import tqdm

warnings.filterwarnings('ignore')



MAIN_PATH = Path('C:/Users/jalan/Documents/PhD/Side_Projects/PainProject/EPHYS/')
ABF_DATA_PATH = MAIN_PATH / 'Abf Traces'
SAVE_PATH = Path('C:/Users/jalan/OneDrive/Desktop/PV_FRMP/astrocyte_KO/results/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Analysis Cohorts & Aesthetic Color Profiles
GROUPS_TO_COMPARE = ['WT', 'KO']
GROUP_COLORS = {'WT': 'darkorange', 'KO': 'goldenrod'}
SPIKE_ORDER = ['1st spike', '3rd spike', '5th spike']
SAVE_FIGURES = True

# Signal Filtering & Extraction Thresholds
V_THRESH = -5.0
efel.api.set_setting('Threshold', V_THRESH)

FEATURES_TO_EXTRACT = [
    'peak_voltage', 'min_AHP_values', 'AP_begin_voltage', 'spike_half_width',
    'AP_rise_time', 'AP_fall_time', 'AP_rise_rate', 'AP_fall_rate',
    'AHP_time_from_peak', 'AHP_depth_from_peak'
]

AP_VERTICAL_LABELS = {
    'AP peak': 'AP Peak (mV)',
    'AP trough': 'AP Trough (mV)',
    'spike threshold': 'Spike Threshold (mV)',
    'AP half-width': 'AP Half-width (ms)',
    'AP rise time': 'AP Rise Time (ms)',
    'AP fall time': 'AP Fall Time (ms)',
    'AP rise rate': 'AP Rise Rate (mV/ms)',
    'AP fall rate': 'AP Fall Rate (mV/ms)',
    'AHP time': 'AHP Time (ms)',
    'AHP depth from peak': 'AHP Depth (mV)'
}


def lowpass_filter(data, cutoff_freq=1000, fs=20000, order=4):
    """Applies a zero-phase forward-backward Butterworth lowpass filter."""
    b, a = butter(order, cutoff_freq, btype='low', fs=fs)
    return filtfilt(b, a, data)


def safe_access(features_dict, key, index):
    """Safely extracts a specific array index element from an eFEL metrics dictionary."""
    values = features_dict.get(key, [])
    if values is None or len(values) <= index:
        return np.nan
    return values[index]


def filter_valid_ap_features(df, feature_columns):
    """Filters data columns to discard empty or non-numeric vectors."""
    valid_features, valid_labels = [], []
    for f in feature_columns:
        if not df[f].isna().all():
            valid_features.append(f)
            valid_labels.append(AP_VERTICAL_LABELS.get(f, f))
    return valid_features, valid_labels

# ==============================================================================
# 3. EXTRACTION PIPELINE EXECUTION
# ==============================================================================

def extract_ap_waveforms(metadata_df, abf_dir):
    """Parses raw ABF files to extract geometric features of targeted spikes."""
    results = []
    
    for idx in tqdm(range(len(metadata_df)), desc="Extracting AP Kinetics"):
        row = metadata_df.iloc[idx]
        if row['Input'] != 'step' or row['Self_eval'] == 'omit':
            continue

        abf_file = abf_dir / str(row['Date']) / f"{row['Code']}.abf"
        if not abf_file.exists():
            continue

        try:
            abf = pyabf.ABF(str(abf_file))
            abf.setSweep(8)

            v_data = abf.sweepY
            i_data = abf.sweepC
            t_span = 1000 * abf.sweepX  # Convert to milliseconds
            fs_rate = 1 / ((t_span[1] - t_span[0]) / 1000)
            v_data = lowpass_filter(v_data, fs=fs_rate)

            spike_idx, _ = find_peaks(v_data, height=V_THRESH)
            stim_idx = np.where(i_data > 0)[0]

            if len(spike_idx) < 5 or len(stim_idx) == 0:
                continue

            trace_profile = {
                'T': t_span, 'V': v_data,
                'stim_start': [t_span[stim_idx[0]]], 'stim_end': [t_span[stim_idx[-1]]],
            }

            features = efel.get_feature_values([trace_profile], FEATURES_TO_EXTRACT)
            if not features or features[0] is None:
                continue

            n_spikes_efel = len(features[0].get('peak_voltage', []))
            target_spikes = {0: '1st spike', 2: '3rd spike', 4: '5th spike'}

            for spike_idx_num, spike_label in target_spikes.items():
                if n_spikes_efel >= (spike_idx_num + 1):
                    results.append({
                        'Date': row['Date'], 'Code': row['Code'], 'Mouse_info': row['Mouse_info'],
                        'Spike': spike_label,
                        'AP peak': safe_access(features[0], 'peak_voltage', spike_idx_num),
                        'AP trough': safe_access(features[0], 'min_AHP_values', spike_idx_num),
                        'spike threshold': safe_access(features[0], 'AP_begin_voltage', spike_idx_num),
                        'AP half-width': safe_access(features[0], 'spike_half_width', spike_idx_num),
                        'AP rise time': safe_access(features[0], 'AP_rise_time', spike_idx_num),
                        'AP fall time': safe_access(features[0], 'AP_fall_time', spike_idx_num),
                        'AP rise rate': safe_access(features[0], 'AP_rise_rate', spike_idx_num),
                        'AP fall rate': safe_access(features[0], 'AP_fall_rate', spike_idx_num),
                        'AHP time': safe_access(features[0], 'AHP_time_from_peak', spike_idx_num),
                        'AHP depth from peak': safe_access(features[0], 'AHP_depth_from_peak', spike_idx_num),
                    })
        except Exception as e:
            print(f"Skipping corrupt or unreadable trace file: {row['Code']} -> {e}")
            continue

    output_df = pd.DataFrame(results)
    output_df.to_excel(SAVE_PATH / 'AP_spike_features_with_cKO.xlsx', index=False)
    return output_df



def plot_feature_across_spikes(df, feature, vertical_label, groups, colors_dict, ax=None):
    """Plots box-whisker distributions across sequential spikes with post-hoc brackets."""
    df_filt = df[df['Mouse_info'].isin(groups)].dropna(subset=[feature])
    if df_filt.empty:
        if ax is not None: ax.axis('off')
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    positions, data_matrix, color_list = [], [], []
    for i, spike in enumerate(SPIKE_ORDER):
        for j, group in enumerate(groups):
            vals = df_filt[(df_filt['Spike'] == spike) & (df_filt['Mouse_info'] == group)][feature].values
            if len(vals) == 0: continue
            positions.append(i * (len(groups) + 1) + j)
            data_matrix.append(vals)
            color_list.append(colors_dict[group])

    if not data_matrix: return

    bp = ax.boxplot(data_matrix, positions=positions, patch_artist=True, showfliers=False,
                    boxprops=dict(alpha=0.3), zorder=10)
    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    np.random.seed(42)
    for pos, vals, color in zip(positions, data_matrix, color_list):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(pos + jitter, vals, s=35, c=color, edgecolors='black', linewidths=0.6, zorder=20)

    centers = [np.mean(positions[i:i + len(groups)]) for i in range(0, len(positions), len(groups))]
    ax.set_xticks(centers)
    ax.set_xticklabels(['1st Spike', '3rd Spike', '5th Spike'], fontsize=11)
    ax.set_ylabel(vertical_label, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Contextual Step Significance Calculations
    y_std = df_filt[feature].std() if df_filt[feature].std() > 0 else 1
    for i, spike in enumerate(SPIKE_ORDER):
        spike_df = df_filt[df_filt['Spike'] == spike]
        g_vals = [spike_df[spike_df['Mouse_info'] == g][feature].values for g in groups if len(spike_df[spike_df['Mouse_info'] == g]) > 0]
        
        if len(g_vals) < 2: continue
        
        _, p_kr = kruskal(*g_vals)
        if p_kr < 0.05:
            posthoc = sp.posthoc_dunn(spike_df, val_col=feature, group_col='Mouse_info')
            spike_positions = [i * (len(groups) + 1) + j for j in range(len(groups))]
            y_base = spike_df[feature].max() + 0.12 * y_std

            for k, (g1, g2) in enumerate(combinations(groups, 2)):
                if g1 not in posthoc.index or g2 not in posthoc.columns: continue
                p_val = posthoc.loc[g1, g2]
                sig = "***" if p_val < 0.0005 else "**" if p_val < 0.005 else "*" if p_val < 0.05 else "n.s."
                
                if sig != "n.s.":
                    x1, x2 = spike_positions[groups.index(g1)], spike_positions[groups.index(g2)]
                    y_coord = y_base + k * 0.15 * y_std
                    ax.plot([x1, x1, x2, x2], [y_coord, y_coord + 0.02 * y_std, y_coord + 0.02 * y_std, y_coord], c='k', lw=1)
                    ax.text((x1 + x2) / 2, y_coord + 0.025 * y_std, sig, ha='center', va='bottom', fontsize=9, fontweight='bold')

    if standalone:
        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(SAVE_PATH / f"AP_{feature.replace(' ', '_')}_PVKO.png", dpi=300, transparent=True)
        plt.show()


def create_ap_feature_summary_grid(df, groups, colors_dict):
    """Assembles all valid AP feature subplots into a high-density unified figures panel."""
    feature_cols = [c for c in df.columns if c not in ['Date', 'Code', 'Mouse_info', 'Spike']]
    valid_features, valid_labels = filter_valid_ap_features(df, feature_cols)

    ncols = 4
    nrows = int(np.ceil(len(valid_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (feature, label) in enumerate(zip(valid_features, valid_labels)):
        plot_feature_across_spikes(df, feature, label, groups, colors_dict, ax=axes[i])
        axes[i].set_title(feature, fontsize=13, pad=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    legend_elements = [Patch(facecolor=colors_dict[g], label=g) for g in groups]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "AP_Feature_Summary_Grid_PVKO.png", dpi=300, transparent=True)
    plt.show()


def generate_feature_summary_table(df, groups):
    """Compiles descriptive statistic metrics (Mean ± SEM) and Mann-Whitney P-values."""
    feature_cols = [c for c in df.columns if c not in ['Date', 'Code', 'Mouse_info', 'Spike']]
    summary_rows = []

    for feature in feature_cols:
        for spike in SPIKE_ORDER:
            spike_df = df[df['Spike'] == spike]
            row_entry = {'Feature': feature, 'Spike': spike}
            group_data = {}

            for g in groups:
                vals = spike_df[spike_df['Mouse_info'] == g][feature].dropna().values
                group_data[g] = vals
                
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    sem_val = np.std(vals, ddof=1) / np.sqrt(len(vals))
                    row_entry[f'{g}_Mean±SEM'] = f'{mean_val:.3f} ± {sem_val:.3f}'
                else:
                    row_entry[f'{g}_Mean±SEM'] = 'NaN'
                row_entry[f'{g}_N'] = len(vals)

            if len(groups) == 2 and len(group_data[groups[0]]) > 0 and len(group_data[groups[1]]) > 0:
                _, p = mannwhitneyu(group_data[groups[0]], group_data[groups[1]], alternative='two-sided')
                row_entry['p-value'] = f"{p:.6f}" if p >= 0.0001 else "<0.0001"
            else:
                row_entry['p-value'] = 'NaN'

            summary_rows.append(row_entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(SAVE_PATH / "AP_Feature_Summary_Table.xlsx", index=False)
    return summary_df



print("Executing AP Kinetic Extraction...")
EPHYS_meta = pd.read_excel(
    MAIN_PATH / "EPHYS_data_astrocytes.xlsx",
    converters={'Date': str, 'Cell_Number': str, 'Code': str}
)
# Standardize spaces and hyphens in column strings
EPHYS_meta.columns = EPHYS_meta.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')

# 1. Processing and File Mapping
df_results = extract_ap_waveforms(EPHYS_meta, ABF_DATA_PATH)

# 2. Rendering Graphics Panels
if not df_results.empty:
    print("Rendering statistical distributions and figures...")
    feature_columns = [c for c in df_results.columns if c not in ['Date', 'Code', 'Mouse_info', 'Spike']]
    valid_features_list, valid_labels_list = filter_valid_ap_features(df_results, feature_columns)

    # Draw individual category boxplots
    for feat, label_str in zip(valid_features_list, valid_labels_list):
        plot_feature_across_spikes(df_results, feat, label_str, GROUPS_TO_COMPARE, GROUP_COLORS)

    # Generate structural summary canvas layout grid
    create_ap_feature_summary_grid(df_results, GROUPS_TO_COMPARE, GROUP_COLORS)

    # 3. Export Summary Matrices
    print("Writing statistical calculation table records...")
    stats_table = generate_feature_summary_table(df_results, GROUPS_TO_COMPARE)
    print(stats_table.head(10))
    print("\nAnalysis execution successfully completed.")
else:
    print("Process aborted: Extracted feature array matrix contains no valid samples.")
