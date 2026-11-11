# Subject-Independent Motor-Imagery EEG Classification

## Research Project Report

### Abstract

This project investigates binary classification of left-fist and right-fist motor imagery from the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB). The study uses runs 4, 8, and 12, applies one consistent preprocessing pipeline to every subject, establishes a within-subject CSP+LDA baseline, and evaluates EEGNet under a subject-independent protocol. The central methodological requirement is that final test subjects are never used for training, early stopping, learning-rate scheduling, or model selection.

The project is designed as a reproducible baseline rather than a clinical claim. Its outputs include subject-level split information, classification metrics, a confusion matrix, and a concise record of methodological limitations and follow-up experiments.

## 1. Objectives

1. Build a clean binary motor-imagery EEG classification pipeline for T1 versus T2.
2. Quantify a classical within-subject CSP+LDA baseline.
3. Estimate cross-subject generalization using EEGNet and grouped subject splits.
4. Prevent test-set leakage during training and model selection.
5. Save the exact split and outputs so later sparse-channel experiments are comparable.
6. Report limitations and uncertainty honestly rather than treating one accuracy value as proof of usefulness.

## 2. Dataset and task

The source is the PhysioNet EEGMMIDB dataset stored locally in the project workspace. The dataset contains EDF recordings for subjects S001-S109. Runs R04, R08, and R12 are selected because they contain the two-class motor-imagery task.

- T1: left-fist motor imagery
- T2: right-fist motor imagery
- Sampling rate: standardized to 160 Hz
- Epoch interval: -1 to 4 seconds around each event
- Model interval: 1 to 4 seconds after the cue

## 3. Methodology

### 3.1 Preprocessing

The same preprocessing is applied to every subject:

1. Load available R04, R08, and R12 EDF files.
2. Concatenate the selected runs for each subject.
3. Clean trailing punctuation from channel names and attach a standard montage when possible.
4. Band-pass filter from 8 to 30 Hz to retain motor-imagery mu and beta activity.
5. Apply common-average EEG reference.
6. Create epochs with baseline correction from -1 to 0 seconds.
7. Crop every epoch to the 1-4 second post-cue window.
8. Reject epochs marked invalid by annotations.

The baseline deliberately avoids subject-dependent rejection thresholds. Artifact handling and ICA are reserved for controlled ablation experiments so that improvements can be attributed to a defined change.

### 3.2 Within-subject baseline

CSP extracts spatial filters that maximize class-related variance differences. LDA classifies the resulting features. A five-fold stratified cross-validation procedure is run independently within each of the first 20 usable subjects. CSP is fitted inside each training fold, preventing validation data from influencing spatial feature learning.

This experiment answers: how well can the task be classified when training data from the same person is available? It is not evidence of subject-independent performance.

### 3.3 Subject-independent EEGNet experiment

EEGNet uses temporal convolution, depthwise spatial convolution across electrodes, separable convolution, pooling, dropout, and a two-class linear output layer. Each epoch and channel is standardized across its own time samples. This normalization does not use statistics from another subject.

The data split is grouped by subject:

- 64% approximately: training subjects
- 16% approximately: validation subjects
- 20%: final test subjects

The exact proportions depend on the available subject count. Two `GroupShuffleSplit` operations create the split, and assertions verify that all subject sets are disjoint.

### 3.4 Model selection

The EEGNet is trained on training subjects only. Validation accuracy controls learning-rate scheduling, early stopping, and best-model selection. The test loader is not used in this loop. After training, the best validation-selected state is restored and evaluated on test subjects once.

## 4. Evaluation protocol

The project reports:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Confusion matrix
- Number and identity of test subjects
- Exact random seed and saved split

Accuracy is interpreted together with class balance and subject-level variability. Since epochs from one person are correlated, the number of independent subjects is more relevant than the raw number of epochs when discussing generalization.

## 5. Results currently available

The existing baseline record in `PROJECT_METHODOLOGY.md` reports the following completed run:

- Original dropout 0.50 configuration: 60.40% validation accuracy and 58.27% unseen-subject test accuracy, with 58.18% macro-F1.
- Fixed-seed dropout 0.25 configuration: 61.78% validation accuracy and 58.38% unseen-subject test accuracy, with 58.38% macro-F1.
- Reported confusion matrix counts: 288/209 for T1 and 201/287 for T2.

These are modest results near the binary chance level. The improvement from dropout 0.50 to 0.25 is small and does not justify a claim of large generalization improvement. The new notebook is configured to reproduce the protocol and save a new run under `outputs_research`.

## 6. Interpretation

The result is consistent with the known difficulty of cross-subject EEG transfer. EEG amplitude, electrode contact, anatomy, timing, and individual motor-imagery strategy differ between participants. A model can therefore learn patterns that are useful for some people but unstable for unseen people.

The low or moderate subject-independent result does not mean motor-imagery signal is absent. A stronger within-subject baseline can coexist with weak cross-subject transfer. The correct conclusion is that subject-specific information exists, while transfer without calibration remains challenging.

No clinical, diagnostic, or real-time deployment claim is supported by this baseline alone.

## 7. Reproducibility and deliverables

The new notebook is `Motor_Imagery_EEG_Research_Project.ipynb`. It contains:

1. Environment and seed setup.
2. Dataset loading and preprocessing.
3. Sanity checks.
4. CSP+LDA baseline.
5. Subject-wise train/validation/test split.
6. EEGNet training with validation-only selection.
7. Final unseen-subject evaluation.
8. Saved metrics, split data, and confusion matrix.

The notebook writes outputs to `outputs_research`. Run it in VS Code with a Python environment that supports Jupyter and the listed packages.

## 8. Limitations

- A single subject split gives a noisy estimate of generalization.
- The first 20 subjects are used for the classical baseline, while the main experiment uses all usable subjects.
- Per-epoch normalization is a modeling choice and may remove useful amplitude information.
- No ICA or subject-adaptive artifact correction is included in the baseline.
- The parameter sweep should not be used to select a final model from test performance.
- No external dataset or session-held-out evaluation is included.

## 9. Recommended next experiments

1. Repeat the grouped experiment over several fixed subject splits and report mean, standard deviation, and confidence intervals.
2. Compare per-epoch normalization with normalization statistics learned only from training subjects.
3. Evaluate fixed central motor-cortex channel subsets using the same saved subject split.
4. Add consistent artifact handling as an ablation, including an explicitly documented ICA protocol.
5. Compare 64, 32, 16, and 8 channels while keeping preprocessing, subjects, and training procedure fixed.
6. Evaluate lightweight calibration using a small amount of data from a new participant.

## Conclusion

This project implements a defensible research baseline for left-versus-right motor-imagery EEG classification. Its main contribution is methodological discipline: consistent preprocessing, a clearly labeled within-subject baseline, grouped subject-independent evaluation, and strict separation of validation from final testing. The current evidence supports a cautious conclusion: EEGNet achieves limited but measurable cross-subject performance, and stronger claims require repeated subject splits, controlled ablations, and calibration or external validation.
