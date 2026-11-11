# Adaptive Sparse EEG Motor Imagery BCI Classification: Comprehensive Project Documentation & Diagnostic Analysis

---

## 1. Project Context & Objectives

The goal of this project is to develop and evaluate adaptive sparse-channel EEG classification models for motor execution and motor imagery tasks using the **PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**.

The core research question investigates:
> **"Can adaptive sparse channel selection (using as few as 16 EEG channels) match or surpass full 64-channel performance, and does per-subject personalized channel selection outperform global fixed channel selection under limited calibration data?"**

---

## 2. Dataset & Task Architecture

### Dataset Specification
- **Database**: PhysioNet EEGMMIDB (109 subjects, 64 channels, 160 Hz sampling rate).
- **Preprocessing Pipeline**:
  1. Bandpass filter: 8.0 Hz – 30.0 Hz (retaining Motor-Imagery $\mu$ and $\beta$ rhythms).
  2. Common Average Reference (CAR).
  3. Standard 10-05 montage mapping.
  4. Epoch interval: -1.0s to 4.0s relative to cue onset; cropped to **1.0s – 4.0s post-cue classification window**.

### Task Families
The 12 motor task runs are structured into four distinct binary classification problems:

| Task Family | PhysioNet Runs | Class 0 (T1) | Class 1 (T2) |
|---|---:|---|---|
| **`execution_unilateral`** | 3, 7, 11 | Left fist execution | Right fist execution |
| **`imagery_unilateral`** | 4, 8, 12 | Imagined left fist | Imagined right fist |
| **`execution_bilateral`** | 5, 9, 13 | Both fists execution | Both feet execution |
| **`imagery_bilateral`** | 6, 10, 14 | Imagined both fists | Imagined both feet |

---

## 3. Methodology & Leakage Prevention Guardrails

The personalized channel experiment (`personalized_channel_experiment.py`) enforces strict zero-leakage research protocols:

```
                      Raw Subject EEG Epochs (64 Channels)
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Stratified Calibration / Test Split   │
                   │ (Calibration: N_calib per class)       │
                   └───────────────────┬───────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│ Calibration Set (X_calib) │                         │  Held-Out Test (X_test)   │
└───────────┬───────────────┘                         └───────────┬───────────────┘
            │                                                     │
            ├───────────────────────────────┐                     │
            ▼                               ▼                     │
┌───────────────────────┐       ┌───────────────────────┐         │
│ Global Fixed 16 Set   │       │ Personalized 16 Set   │         │
│ (From Training Set)   │       │ (Scored on X_calib)   │         │
└───────────┬───────────┘       └───────────┬───────────┘         │
            │                               │                     │
            ▼                               ▼                     │
┌───────────────────────┐       ┌───────────────────────┐         │
│ Fit FBCSP + LDA       │       │ Fit FBCSP + LDA       │         │
│ (on X_calib only)     │       │ (on X_calib only)     │         │
└───────────┬───────────┘       └───────────┬───────────┘         │
            │                               │                     │
            └───────────────────────┬───────┴─────────────────────┘
                                    ▼
                         ┌───────────────────────┐
                         │ Predict & Evaluate    │
                         │ (on X_test ONLY)      │
                         └───────────────────────┘
```

1. **Split-First Policy**: Calibration vs. test partitioning is executed *before* channel ranking, feature extraction, or spatial filter fitting.
2. **Calibration-Only Scoring**: Channel separability ($t$-statistic on 8-30 Hz band-power) is calculated strictly on calibration trials ($X_{\text{calib}}, y_{\text{calib}}$).
3. **Paired Test Evaluation**: Both Fixed and Personalized conditions evaluate on the exact same held-out test split per subject.
4. **Non-Parametric Statistical Testing**: Paired Wilcoxon signed-rank tests assess statistical significance.

---

## 4. Complete Empirical Results

### 4.1 Benchmark across All 4 Task Families ($N_{\text{calib}}=10$ trials/class, 20 total calibration trials)

| Task Group | Subjects | Mean Fixed Acc | Mean Personalized Acc | Mean Delta ($\Delta$) | Improved / Tied / Degraded | Wilcoxon $p$-value | Significance |
|---|---:|---:|---:|---:|---:|---:|---|
| **Real Unilateral** | 109 | **61.48%** | 59.68% | -1.80% | 43 / 13 / 53 (39.4% improved) | 0.18129 | Not Significant |
| **Imagined Unilateral** | 109 | 56.94% | **58.07%** | **+1.13%** | 54 / 11 / 44 (49.5% improved) | 0.43394 | Not Significant |
| **Real Bilateral** | 109 | 78.25% | **78.54%** | **+0.29%** | 43 / 28 / 38 (39.4% improved) | 0.88933 | Not Significant |
| **Imagined Bilateral** | 109 | 64.45% | **65.43%** | **+0.97%** | 54 / 15 / 40 (49.5% improved) | 0.27059 | Not Significant |

### 4.2 Calibration Sample Size Sweep (Imagined Unilateral Task)

| Calibration / Class | Total Calibration Trials | Mean Fixed Acc | Mean Personalized Acc | Mean Delta ($\Delta$) | Subjects Improved | Wilcoxon $p$-value |
|---:|---:|---:|---:|---:|---:|---:|
| **5 / class** | 10 trials | 52.66% | 51.88% | -0.78% | 45 / 109 (41.3%) | 0.49482 |
| **10 / class** | 20 trials | 56.94% | 58.07% | +1.13% | 54 / 109 (49.5%) | 0.43394 |
| **15 / class** | 30 trials | 58.71% | 58.11% | -0.60% | 43 / 109 (39.4%) | 0.77955 |
| **20 / class** | 40 trials | 58.60% | **62.12%** | **+3.53%** | 38 / 100 (38.0%) | 0.24841 |

---

## 5. Diagnostic Analysis: Why Are Accuracies Currently Around 56%–78%?

A key question in BCI research is why cross-subject or low-trial motor imagery classification typically peaks around 56–78% rather than 85–95%. The diagnostic breakdown reveals four primary root causes:

### 1. High Inter-Subject Variability & BCI Illiteracy
- **Physiological Differences**: Skull thickness, cortical folding geometry, electrode-skin impedance, and sensorimotor rhythm ($\mu/\beta$) peak frequencies differ dramatically across participants.
- **BCI Illiteracy**: Literature shows that 15% to 30% of human participants fail to generate detectable Event-Related Desynchronization (ERD/ERS) during motor imagery. When a participant's brain produces no motor imagery signal, no algorithm can classify it accurately without subject adaptation.

### 2. High Variance in Small Calibration Samples
- Estimating single-channel $t$-statistics or spatial covariance matrices from only 10 to 20 calibration trials suffers from high sampling noise.
- Spurious channel selection occurs (e.g. selecting noisy parietal/frontal electrodes instead of motor channels C3, Cz, C4).
- **Global channels act as a spatial prior**: Global channel selection derived across 80+ subjects forces the model to focus on canonical motor regions, compensating for small sample sizes.

### 3. Artifact Contamination & Low Signal-to-Noise Ratio (SNR)
- Non-invasive scalp EEG has a very low signal-to-noise ratio ($< 0 \text{ dB}$).
- Ocular artifacts (blinks, eye movements), muscle activity (jaw clenching, neck tension), and electrode drift distort raw signals. Currently, no automated Independent Component Analysis (ICA) or artifact rejection is applied.

### 4. Non-Stationarity Across Recording Sessions
- EEG signals drift over time due to mental fatigue, concentration shifts, electrode gel drying, and slight head movements. Spatial filters learned on early calibration trials often degrade on later test trials.

---

## 6. Actionable Blueprint & Strategic Roadmap for Higher Accuracy (80%+ Target)

To push classification performance toward **80%+ accuracy**, the following step-by-step advancements are recommended:

```
                                  IMPROVEMENT ROADMAP
                                  ───────────────────

   ┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
   │  1. Artifact Rejection    │ ───► │ 2. Riemannian Geometry    │ ───► │ 3. Calibration Adaptation │
   │  (ICA & EOG/EMG Pruning)  │      │    (Tangential Covariance)│      │    (Few-Shot Transfer)    │
   └───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

### Step 1: Automated Artifact Cleaning (ICA & Automated Thresholding)
- **Action**: Implement automated Independent Component Analysis (`mne.preprocessing.ICA` with `corrmap` or `mne-icalabel`) to automatically strip ocular ($Fp1, Fp2$) and muscular artifacts prior to epoching.
- **Expected Impact**: Reduces non-physiological variance by $+3\%$ to $+5\%$.

### Step 2: Riemannian Geometry & Covariance Matrix Classifiers (TS+LDA / FBRCSP)
- **Action**: Replace standard Euclidean CSP features with **Riemannian Minimum Distance to Mean (MDM)** or **Tangent Space LDA** (`pyriemann.estimation.Covariances`, `pyriemann.tangentspace.TangentSpace`).
- **Why it works**: EEG spatial covariance matrices lie on a Riemannian manifold of symmetric positive definite (SPD) matrices. Mapping to tangent space preserves non-Euclidean geometry and yields higher cross-subject robustness.
- **Expected Impact**: Increases classification accuracy by $+5\%$ to $+8\%$ on motor imagery tasks.

### Step 3: Domain Adaptation & Calibration Fine-Tuning
- **Action**: Implement **Maximum Mean Discrepancy (MMD)** alignment or **CORAL (Correlation Alignment)** to align the feature distributions of unseen test subjects with training subjects before prediction.
- **Expected Impact**: Reduces cross-subject domain shift, boosting subject-independent transfer performance.

### Step 4: Subject-Adaptive Multi-Band Filter Bank (FB-CSP-LDA with Lasso)
- **Action**: Apply $L_1$-regularized feature selection (Lasso LDA or ElasticNet) across 8 sub-bands (4–8 Hz, 8–10 Hz, 10–12 Hz, 12–14 Hz, 14–18 Hz, 18–22 Hz, 22–26 Hz, 26–30 Hz) during calibration.
- **Expected Impact**: Captures subject-specific $\mu/\beta$ frequency shifts.

---

## 7. Deliverables & File Registry

1. `personalized_channel_experiment.py`: Main python code executing Fixed vs Personalized comparisons.
2. `PERSONALIZED_CHANNEL_RESULTS.md`: Detailed empirical research report.
3. `PERSONALIZED_ADAPTIVE_SPARSE_EEG_DOCUMENTATION.md`: Complete documentation (this file).
4. `outputs_personalized/personalized_channels_all_subjects.csv`: Per-subject CSV benchmark output.
5. `outputs_personalized/personalized_experiment_summary.json`: JSON output of statistical test metrics.
