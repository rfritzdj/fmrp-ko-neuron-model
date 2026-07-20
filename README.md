# FMRP-KO PVN Modeling

Code accompanying the paper:
**"Astrocytic FMRP regulates the function of spinal parvalbumin-expressing neurons in Fragile X Syndrome"**

Paper link: https://www.biorxiv.org/content/10.64898/2026.07.17.737291v1

This repository contains the computational models and analysis pipelines used to identify candidate ionic mechanisms underlying the altered firing phenotype of spinal parvalbumin interneurons (PVNs) in FMRP knockout mice.

## Repository structure

### `pvn-model/`
Single-compartment Hodgkin-Huxley model of a spinal parvalbumin interneuron. Includes seven voltage-gated conductances (Na, Kv1, Kv3, Ca, SK, leak) plus calcium dynamics with a lumped intracellular buffer (Btot), and another model with synaptic input (AMPA/NMDA) for the astrocytes KO model. Used for channel-perturbation analysis and mechanism testing.

### `data-feature-extraction/`
Extracts firing features from whole-cell patch-clamp recordings (ABF files) for statistical comparison between genotypes. Features include spike count, discharge time, adaptation index, frequency decay time constant, and single-spike shape metrics (AP peak, trough, and half-width).

### `parameter-estimation/`
Population-level parameter inference pipeline. Given a cohort of PVN recordings, identifies parameter sets in the Hodgkin-Huxley model that reproduce the observed firing features. Two inference methods are provided:

- **Direct ranking** (`snpe/02_direct_ranking.py`): ranks a broad prior sample of simulations by NaN-aware z-distance to cohort mean. Robust when the model cannot fully match all features.
- **Sequential Neural Posterior Estimation** (`snpe/02a_snpe_inference.py` + `02b_predictive_top_n.py`): learns a continuous joint posterior over parameters using a normalizing flow. Gives credible intervals and parameter correlations.

Both methods produce representative parameter sets that can be used as input for downstream channel perturbation and WT-vs-KO comparison analyses.

## Requirements
```bash
pip install numpy pandas matplotlib scipy numba efel pyabf tqdm
pip install sbi   # optional, only for the SNPE pipeline
```

## Citation
If you use this code, please cite Qiu, H., Jalandoni, R. F., Derham, A., Reinish, N., Chen, C., Vong, A., Cook, E. P., Krishnaswamy, A., Khadra, A., & Sharif-Naeini, R. (2026). Astrocytic FMRP regulates the function of spinal parvalbumin-expressing neurons in Fragile X Syndrome. Department of Physiology, McGill University, Montreal, QC, Canada..
