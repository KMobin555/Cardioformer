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


---

# Token Embedding Method

<p align="center" style="background-color:#e9e9e9; padding:10px;">
  <img src="assets/token_embed.png" alt="Cardioformer's Token embeddings" width="80%">
</p>

we propose Cardioformer considering inter-channel dependencies (multi-channel), temporal properties (multi-timestamp), and multifaceted scale of temporal patterns (multi-granularity).


---

# Model Architecture

<p align="center" style="background-color:#e9e9e9; padding:10px;">
  <img src="assets/model_arc.png" alt="Cardioformer's Token embeddings" width="80%">
</p>


# Datasets

## Data Preprocessing

[PTB](https://physionet.org/content/ptbdb/1.0.0/) is a public ECG time series recording from 290 subjects, with 15 channels and a total of 8 labels representing 7 heart diseases and 1 health control. The raw sampling rate is 1000Hz. For this paper, we utilize a subset of 198 subjects, including patients with Myocardial infarction and healthy control subjects. We first downsample the sampling frequency to 250Hz and normalize the ECG signals using standard scalers. Subsequently, we process the data into single heartbeats through several steps. We identify the R-Peak intervals across all channels and remove any outliers. Each heartbeat is then sampled from its R-Peak position, and we ensure all samples have the same length by applying zero padding to shorter samples, with the maximum duration across all channels serving as the reference. This process results in 64,356 samples. For the training, validation, and test set splits, we employ the subject-independent setup. Specifically, we allocate 60%, 20%, and 20% of the total subjects, along with their corresponding samples, into the training, validation, and test sets, respectively.

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) is a large public ECG time series dataset recorded from 18,869 subjects, with 12 channels and 5 labels representing 4 heart diseases and 1 healthy control category. Each subject may have one or more trials. To ensure consistency, we discard subjects with varying diagnosis results across different trials, resulting in 17,596 subjects remaining. The raw trials consist of 10-second time intervals, with sampling frequencies of 100Hz and 500Hz versions. For our paper, we utilize the 500Hz version, then we downsample to 250Hz and normalize using standard scalers. Subsequently, each trial is segmented into non-overlapping 1-second samples with 250 timestamps, discarding any samples shorter than 1 second. This process results in 191,400 samples. For the training, validation, and test set splits, we employ the subject-independent setup. Specifically, we allocate 60%, 20%, and 20% of the total subjects, along with their corresponding samples, into the training, validation, and test sets, respectively.

[MIMIC-IV](https://physionet.org/content/mimic-iv-ecg/1.0/) is a large public ECG time series dataset recorded from 800,000 subjects, with 12 channels and 4 labels representing 3 heart diseases and 1 healthy control category. Each subject may have one or more trials. To ensure consistency, we discard subjects with varying diagnosis results across different trials, resulting in 19,931 subjects remaining. The raw trials consist of 10-second time intervals, with sampling frequencies of 100Hz and 500Hz versions. For our paper, we utilize the 500Hz version, then we downsample to 250Hz and normalize using standard scalers. Subsequently, each trial is segmented into non-overlapping 1-second samples with 250 timestamps, discarding any samples shorter than 1 second. This process results in 199,310 samples. For the training, validation, and test set splits, we employ the subject-independent setup. Specifically, we allocate 60%, 20%, and 20% of the total subjects, along with their corresponding samples, into the training, validation, and test sets, respectively.

# Run Experiements

Before running any experiments, make sure all processed datasets are placed in the `dataset/` directory. Install the required dependencies by running `pip install -r requirements.txt`. You can run the experiments using the provided Jupyter notebook `experiments.ipynb`, which includes all training and evaluation commands organized cell by cell. GPU devices can be specified using the `--devices` command-line argument (e.g., `--devices 0,1,2,3`), and the visible devices should be set using export `CUDA_VISIBLE_DEVICES=0,1,2,3`. Note that the devices passed via `--devices` must be a subset of the visible devices.

After training, the saved models will be located in `checkpoints/classification/`, and the evaluation results will be available in `results/classification/`. You can customize experiment parameters through command-line options—each parameter and its explanation can be found in the `run.py` file. For a quick test and to get familiar with the framework, we recommend using the `PTB dataset`, as it is small and fast to run.


# Train on Custom Dataset

To train the model on a custom dataset, you need to write a customized dataloader to load your processed data in `data_provider/data_loader.py`, and add its name in `data_provider/data_factory.py`. Then you can run the training script with the dataset name you added in `data_factory.py`, where the `--root_path` is the path to the root directory of the processed dataset and the `--data` is the name of the dataloader you added in `data_factory.py`.


# Citation

If you find this repo useful, please star our project and cite our paper.

> ⚠️ Note: This citation is for the preprint version available on arXiv.

```bash
@misc{mobin2025cardioformeradvancingaiecg,
      title={Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet}, 
      author={Md Kamrujjaman Mobin and Md Saiful Islam and Sadik Al Barid and Md Masum},
      year={2025},
      eprint={2505.05538},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.05538}, 
}
```


# Acknowledgement

This project is constructed based on the code in repo [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Thanks a lot for their amazing work on implementing state-of-arts time series methods!