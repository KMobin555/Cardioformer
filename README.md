# Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet

## Authors:

[Md Kamrujjaman Mobin](https://scholar.google.com/citations?user=0pXfjCcAAAAJ&hl=en)\* (kamrujjamanmobin123@gmail.com), [Md Saiful Islam](https://scholar.google.com/citations?user=tQT0OSAAAAAJ&hl=en)* (sislam@athabascau.ca), [Sadik Al Barid]() (nebir2002@gmail.com), [Md Masum]() (masum-cse@sust.edu).


## Preprint:
[![arXiv](https://img.shields.io/badge/arXiv-2505.05538-b31b1b.svg)](https://arxiv.org/abs/2505.05538)


## 🔍 Overview

This repository provides the implementation of **Cardioformer**, along with detailed descriptions of three publicly available ECG datasets used in our experiments. Cardioformer is a **multi-granularity patching Transformer** designed specifically for Electrocardiogram (ECG) classification, as presented in our paper:

> **Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet**  
> [arXiv:2505.05538](https://arxiv.org/abs/2505.05538)

Our approach introduces four key innovations that leverage the unique characteristics of ECG signals:

- **Cross-channel patching** to exploit inter-lead correlations  
- **Multi-granularity embedding** for capturing features at different temporal resolutions  
- **Two-stage multi-granularity self-attention** (intra- and inter-granularity) for efficient representation learning  
- **Residual Network (ResNet) blocks** to enhance feature learning across granularities  

We evaluate Cardioformer on three benchmark ECG datasets using consistent experimental setups. The results demonstrate its superior performance over four strong baselines, achieving the highest average ranking across all six evaluation metrics. Additionally, Cardioformer exhibits strong cross-dataset generalization, highlighting the **universality and robustness** of our proposed model across diverse ECG classification tasks.

<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/KMobin555/Cardioformer/main/assets/cardioformer_pipeline.png" alt="Cardioformer Pipeline" width="80%">
</p>

sdf -->

---
