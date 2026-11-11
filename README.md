# Motor Imagery EEG BCI

Adaptive sparse-channel EEG preprocessing and lightweight machine learning for
subject-independent motor-imagery and motor-execution brain-computer interfaces.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PhysioNet%20EEG-orange.svg)](https://physionet.org/content/eegmmidb/1.0.0/)

## Overview

This project studies how few EEG channels, how little calibration, and how
little computation are needed for a practical motor BCI. It evaluates
Riemannian tangent-space LDA, FBCSP-LDA, and EEGNet on the PhysioNet EEG Motor
Movement/Imagery Dataset.

The central finding is that a fixed, anatomy-informed 16-channel sensorimotor
montage can match a 64-channel pipeline in this evaluation while reducing the
feature dimension from 2,080 to 136. Label-free covariance re-centering also
improves cross-subject performance and has an online streaming formulation.

## Headline results

All headline results use subject-wise 5-fold cross-validation over 109
subjects. Test subjects are never used for training, channel selection, model
selection, or adaptation labels.

| Task | Best offline accuracy |
|---|---:|
| Execution - bilateral | 0.795 |
| Execution - unilateral | 0.716 |
| Imagery - bilateral | 0.667 |
| Imagery - unilateral | 0.641 |

The four tasks are:

1. Both fists vs both feet (execution)
2. Left vs right fist (execution)
3. Both fists vs both feet (imagery)
4. Left vs right fist (imagery)

The causal streaming pipeline uses a 4-second buffer, an 8–30 Hz Butterworth
filter, common-average reference, online covariance re-centering, tangent-space
features, and linear LDA. The core runtime implementation is in
[`rt_pipeline.py`](./rt_pipeline.py).

## Repository contents

This intentionally small repository contains only the main demonstration
notebook, the streaming inference implementation, the final research summary,
and the files needed to install the Python dependencies:

| Path | Purpose |
|---|---|
| [`rt_pipeline.py`](./rt_pipeline.py) | Causal online preprocessing and inference pipeline |
| [`Motor_Imagery_EEG_Research_Project.ipynb`](./Motor_Imagery_EEG_Research_Project.ipynb) | Main research notebook |
| [`FINAL_REPORT.md`](./FINAL_REPORT.md) | Research summary and results |
| [`requirements.txt`](./requirements.txt) | Python dependencies |

Large raw recordings, caches, virtual environments, generated models, logs,
archives, and PDFs are intentionally excluded from version control. The
dataset can be downloaded separately from PhysioNet.

## Setup

```bash
git clone https://github.com/AllamKeerthana/motor-imagery-eeg-bci.git
cd motor-imagery-eeg-bci

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Download and extract the EEGMMIDB dataset into the project directory so that
the data path is:

```text
eeg-motor-movementimagery-dataset-1.0.0/files/
```

The dataset is not included in this repository because of its size and
distribution terms.

## Reproducibility

Example benchmark commands:

```bash
python run_complete_rigorous_benchmark.py
python run_adaptive_channels.py
python run_eegnet_groupkfold.py
```

Most experiments expect the PhysioNet data directory described above and can
take substantial time and memory. GPU-specific scripts require a compatible
PyTorch installation. Review each script's command-line options before running
long experiments.

## Methodology

- Epochs are taken from 1–4 seconds after the cue and filtered to 8–30 Hz.
- Subject-wise `GroupKFold` prevents subject leakage.
- Channel and adaptation choices are selected inside training folds.
- Adaptation is label-free for held-out subjects.
- Results are reported with subject-level comparisons and corrected statistical
  tests where applicable.

Known negative results, evaluation corrections, and the full protocol are
documented in [`PROJECT_METHODOLOGY.md`](./PROJECT_METHODOLOGY.md) and
[`PERSONALIZED_ADAPTIVE_SPARSE_EEG_DOCUMENTATION.md`](./PERSONALIZED_ADAPTIVE_SPARSE_EEG_DOCUMENTATION.md).

## Citation and dataset

This work uses the PhysioNet EEG Motor Movement/Imagery Dataset:

> Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet,” *Circulation*,
> 2000. Dataset: https://physionet.org/content/eegmmidb/1.0.0/

Please follow PhysioNet's terms of use and cite the dataset and relevant
methods papers when reusing this work.

## License

No license has been selected yet. Until a license is added, all rights are
reserved by the copyright holder.
