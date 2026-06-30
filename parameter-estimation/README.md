# PVIN Population Fitting Pipeline

A workflow for fitting a Hodgkin-Huxley model of parvalbumin (PV) interneurons
to a population of patch-clamp recordings, then using channel perturbation
analysis to identify which channels could underlie a phenotypic difference
between genotypes (e.g. WT vs KO).

## Scientific question

Given:
- A 7-conductance PVIN model: `gNa`, `gKv1`, `gKv3`, `gCa`, `gSK`, `gleak`, `Btot`
- Whole-cell patch-clamp recordings from N WT cells (and N KO cells)
- A canonical depolarizing current step (200 pA × 1 s on sweep 8)

We want to:

1. **Find parameter sets that reproduce the WT cohort's firing features** (spike count, adaptation, AP shape, etc.)
2. **Identify which channel changes could shift WT firing toward the KO phenotype** (channel perturbation analysis)


## Pipeline

```
1. feasibility_check.py
   → simulates 20k random parameter sets, saves features to CSV
   → extracts the same features from real cells

2. direct_ranking.py
   → loads the cached sims and ranks them by distance to cohort mean
   → saves top-N representative parameter sets

3. channel_perturbation.py
   → loads the representative sets
   → sweeps each channel over multipliers (0× to 3×)
   → plots mean ± SEM bands across the population
```

Optional: `snpe_inference.py` for a comparison using a normalizing flow.

## Dependencies

```bash
pip install numpy pandas matplotlib scipy numba efel pyabf tqdm
# optional, only for SNPE script:
pip install sbi
```

## Data layout

Expects:
- `EPHYS_data_astrocytes.xlsx` — metadata with columns `Date`, `Code`, `Input`, `Self_eval`, `Mouse_info`
- `Abf Traces/<date>/<code>.abf` — one ABF file per cell, sweep 8 = 200 pA step

## Output files

After running scripts 1-3:
- `feasibility_sims.csv` — 20k rows, each = one (params, features) pair
- `feasibility_real_wt.csv` — N rows, each = one real cell's features
- `WT_representative_params_direct.csv` — top-30 params closest to cohort
- `perturbation_raw_<ch>.csv` — per-model trace data for each channel sweep
- `perturbation_summary_<ch>.csv` — population mean ± SEM per multiplier
- `population_perturbation_bands.png` — the main perturbation figure

## Running KO

The simulation cloud is genotype-agnostic — the same 20k sims can be used for both WT and KO. To do KO:

1. Re-run script 1 with `Mouse_info == 'KO'` filter to extract KO cohort features into `feasibility_real_ko.csv`
2. Re-run script 2 with that CSV → `KO_representative_params_direct.csv`
3. Compare WT vs KO parameter distributions

## Channel perturbation interpretation

For each channel, look at how features (spike count, discharge time, etc.) change with multiplier. If scaling channel X by 1.5× moves a feature from WT value toward KO value, that channel could underlie the WT→KO change. Convergent evidence from multiple channels acting on the same biology (e.g., gCa↑ AND Btot↓ both raise free Ca) strengthens the mechanistic claim.

## Citations

If you use this pipeline, please cite .
