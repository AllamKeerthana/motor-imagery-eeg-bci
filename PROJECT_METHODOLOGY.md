# PhysioNet Motor Imagery EEG Classification

## 1. Project Objective

Build and evaluate a motor-imagery EEG classifier using the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB). The primary task is binary classification of:

- `T1`: left-fist motor imagery
- `T2`: right-fist motor imagery

The main result is a **subject-independent** estimate: subjects in the final test set are never used during training, early stopping, learning-rate scheduling, or model selection.

## 2. Dataset

Local dataset root:

```text
C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0
```

The EEG files are under its `files` directory, with subject folders `S001` through `S109` and EDF recordings such as `S001R04.edf`.

The notebook uses runs `R04`, `R08`, and `R12`, which contain the left- versus right-fist motor-imagery task. Event annotations are read from the EDF annotations and only `T1` and `T2` are retained.

## 3. Methodology

### 3.1 Loading

1. Iterate over the available subject folders.
2. Load runs 4, 8, and 12 for each subject.
3. Concatenate the selected runs for each subject.
4. Clean trailing punctuation from channel names.
5. Attach the standard 10-05 montage where channel names match.
6. Skip missing files or subjects without usable recordings.

### 3.2 Preprocessing

The same preprocessing is applied to every subject:

1. Band-pass filter from 8 to 30 Hz to retain the motor-imagery mu and beta bands.
2. Apply a common-average EEG reference.
3. Create epochs from -1 to 4 seconds around each event.
4. Apply baseline correction using -1 to 0 seconds.
5. Keep the post-cue classification window from 1 to 4 seconds.
6. Reject epochs marked as invalid by annotations.

This baseline deliberately avoids automatic subject-dependent artifact thresholds. ICA can be evaluated later as a separate controlled experiment, but it should be fitted and applied consistently.

### 3.3 Classical Baseline: CSP + LDA

The CSP+LDA experiment is a within-subject baseline. For each of the first 20 usable subjects:

1. Extract the subject's epochs and labels.
2. Perform stratified 5-fold cross-validation within that subject.
3. Fit CSP only on each training fold.
4. Classify CSP features using LDA.
5. Report mean accuracy and macro-F1 across folds.

This result answers how well the task can be classified when data from the same person is available. It must not be presented as subject-independent performance.

### 3.4 Main Model: EEGNet

EEGNet is used for the subject-independent experiment. Its stages are:

1. Temporal convolution.
2. Depthwise spatial convolution across EEG channels.
3. Separable temporal convolution.
4. Pooling and dropout.
5. A final linear layer for two-class prediction.

Inputs are normalized independently for each epoch and channel across time, then reshaped for PyTorch as `[batch, 1, channels, time]`.

### 3.5 Subject-Wise Split

The split is performed using subject IDs rather than random epochs:

1. Hold out 20% of subjects as the final test set.
2. From the remaining subjects, hold out 20% as validation subjects.
3. Use the remaining subjects for training.
4. Assert that all three subject sets are disjoint.

This prevents epochs from the same person appearing in multiple splits.

### 3.6 Training and Model Selection

1. Train EEGNet using training subjects only.
2. Monitor validation accuracy only.
3. Use validation accuracy for learning-rate scheduling.
4. Save the model with the best validation accuracy.
5. Stop early when validation accuracy has not improved for the configured patience.
6. Restore the best validation-selected model.

The test set is not evaluated during training.

### 3.7 Final Evaluation

After model selection, evaluate the restored model once on the unseen test subjects. Report:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Confusion matrix

The subject split and plots are saved under `outputs/` for reproducibility.

## 4. Execution Steps in VS Code

1. Open `Untitled3-3_updated_corrected.ipynb` in VS Code.
2. Select a Python environment with Jupyter support.
3. Install the packages listed below in that environment:

   ```powershell
   python -m pip install mne numpy matplotlib scikit-learn torch jupyter
   ```

4. Run the imports and configuration cell.
5. Run the data sanity-check cell for subjects 1, 2, and 3.
6. Run the CSP+LDA baseline.
7. Run the all-subject loading cell and inspect the printed shape, subject count, and class counts.
8. Confirm the train, validation, and test subject lists are disjoint.
9. Train EEGNet using validation subjects for model selection.
10. Run the final test evaluation and save the plots and split file.

## 5. Correctness Checklist

- [x] Runs 4, 8, and 12 are selected for the binary task.
- [x] T1 and T2 are the only retained event classes.
- [x] Filtering and referencing are applied consistently.
- [x] CSP is fitted inside each cross-validation training fold.
- [x] The EEGNet split is grouped by subject.
- [x] Validation, not test, controls early stopping and scheduling.
- [x] The final test evaluation occurs after model selection.
- [ ] Run the local notebook end to end and record the actual metrics.
- [ ] Repeat the EEGNet experiment with fixed seeds or repeated subject splits if a stronger statistical claim is needed.
- [ ] Compare sparse-channel configurations using the same subject split.

## 6. Important Interpretation Notes

The CSP+LDA and EEGNet results answer different questions and should be reported separately. Accuracy above 50% is not automatically sufficient evidence of a useful BCI system; class balance, subject-level variability, confidence intervals, repeated splits, and an external or session-held-out evaluation should also be considered.

The current EEGNet normalization is per-epoch and per-channel. It avoids cross-subject statistics, but it is a modeling choice that should be reported and compared with alternatives in later experiments.

## 7. Why Accuracy and Validation Are Low

The observed subject-independent result is close to the 50% binary chance level, but the implementation checks completed successfully. The main reasons are:

1. **Cross-subject generalization is difficult.** EEG amplitude, electrode impedance, anatomy, timing, and motor-imagery strategy vary strongly between people. A model trained on one group does not automatically transfer to unseen subjects.
2. **There are few independent examples per subject.** The current setup produces about 45 usable trials per subject. The total epoch count is large, but the effective independent sample size is the number of subjects.
3. **The validation estimate is noisy.** Validation contains only 18 subjects, so a few difficult subjects can move the accuracy noticeably. Epoch-level accuracy should not be interpreted as 4,898 independent human observations.
4. **The baseline is deliberately conservative.** It uses one fixed filter/reference rule, no ICA, no artifact-specific rejection threshold, and no subject adaptation. These choices improve reproducibility but may leave subject-specific noise in the input.
5. **Per-epoch normalization may remove useful amplitude information.** It reduces subject scale differences, but it can also remove class-related amplitude differences. This should be compared with training-only channel normalization in a later experiment.
6. **The first parameter sweep was short.** Its 15 training epochs were not directly comparable with the 150-epoch early-stopped baseline. A configuration that looks good early may not remain best after convergence.

These results should therefore be described as a challenging baseline, not as evidence that the task is impossible. Within-subject CSP+LDA is higher for some participants, confirming that subject-specific signal exists but does not transfer uniformly.

## 8. Improvements Applied

The main experiment now uses fixed NumPy and PyTorch seeds and the lower-dropout configuration suggested by the controlled comparison (`dropout=0.25`). The subject split, preprocessing, validation-only model selection, and final test set remain unchanged, so the new result can be compared fairly with the original baseline.

The parameter comparison is retained as exploratory evidence. Stronger claims require repeating the experiment across several subject splits or reporting confidence intervals, and should not be based on selecting a configuration from the final test set.

### Measured Results

The original reproducible run with dropout `0.50` reached `60.40%` validation accuracy and `58.27%` unseen-subject test accuracy with `58.18%` macro-F1. After applying fixed seeds and dropout `0.25`, validation accuracy increased to `61.78%`; the final test result was `58.38%` accuracy and `58.38%` macro-F1. The confusion matrix was 288/209 for T1 and 201/287 for T2.

The improvement is modest, which is expected for a single subject split. It improves class balance in the final predictions, but it does not justify claiming a large generalization gain. The result remains a challenging but valid cross-subject baseline.

Recommended next experiments are:

- compare per-epoch normalization with training-only channel normalization;
- evaluate a fixed central motor-cortex channel subset;
- add consistent artifact handling or ICA as a separate ablation;
- use repeated subject-wise splits;
- test lightweight subject calibration using training data from a new participant.
