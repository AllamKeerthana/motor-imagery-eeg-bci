# Adaptive Sparse-Channel EEG Preprocessing and Lightweight AI Classification for Real-Time Motor Imagery BCI

## Abstract

We study how few EEG channels a subject-independent motor-imagery/execution BCI needs, and whether a lightweight pipeline with label-free adaptive preprocessing can run in real time. On the PhysioNet EEG Motor Movement/Imagery dataset (109 subjects, 64 channels) we evaluate four binary tasks (left vs right fist and both fists vs both feet, each in execution and imagery) with 5-fold subject-wise cross-validation, so that test subjects are never used for training, channel selection, normalisation, early stopping or model selection. A fixed 16-channel sensorimotor montage is statistically indistinguishable from all 64 channels, whereas 16 channels chosen by a data-driven t-test lose 4–8 points. Label-free per-subject covariance re-centering, the adaptive preprocessing step, adds 4–7 points over no adaptation, and an online (running-mean) version keeps essentially all of that benefit. The deployable pipeline (causal 8–30 Hz filter, online re-centering, tangent-space LDA, or an 858-parameter EEGNet) reaches 0.63–0.76 subject-independent accuracy and decides in about 0.5 ms per 3 s window on a laptop CPU. Per-subject channel selection from 20 calibration trials did not help and is reported as a negative result. During the work we found and removed several evaluation leaks in our own earlier pipeline.

## 1. Objectives

1. Subject-independent classification of four task groups with EEGNet, FBCSP-LDA and Riemannian tangent-space LDA.
2. Reduce 64 channels to 16 without losing accuracy.
3. Adaptive, label-free per-subject preprocessing and few-shot calibration.
4. Lightweight models and a causal streaming pipeline suitable for a Raspberry Pi.

Out of scope in this report: Raspberry Pi timing (pending) and own-EEG collection (future work).

## 2. Data and tasks

PhysioNet EEG Motor Movement/Imagery, 109 subjects, 64 channels, 160 Hz. Epochs 1–4 s after the cue (481 samples). Task groups: execution unilateral (runs 3, 7, 11; left vs right fist), imagery unilateral (4, 8, 12), execution bilateral (5, 9, 13; both fists vs both feet), imagery bilateral (6, 10, 14). About 4,900 trials per task.

## 3. Methods

**Evaluation protocol.** 5-fold `GroupKFold` over subjects, identical folds for every model. Anything chosen (channels, variants, online-adaptation settings) is chosen inside the training fold only, using inner subject-wise cross-validation where selection among alternatives is needed. Each outer test fold is scored once. All variants are also reported separately as reference. Paired tests over folds (or over subjects for per-subject experiments) with Holm correction inside each declared family.

**Channel sets.** (a) 16 channels by a Welch t-test on log band-power, computed on non-CAR training-subject data; (b) a fixed sensorimotor set chosen from anatomy before any results: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CPz, CP4, P3, Pz, P4. The common-average reference is applied over the channels actually used.

**Models.** Riemannian tangent-space LDA (Ledoit–Wolf covariance), FBCSP-LDA, EEGNet (1,842 parameters at 16 channels, 2,610 at 64) and a compact EEGNet (858 parameters).

**Adaptive preprocessing.** Per-subject label-free re-centering: each subject's trial covariances are whitened by that subject's mean covariance, so no labels from the test subject are used. Streaming version: running mean M_t = (w0·P + Σ C_i)/(w0 + t) with prior P from training subjects, optionally rescaled to the user's own amplitude; configuration (w0, rescaling) chosen by nested CV.

**Few-shot calibration.** 0, 2, 5, 10 labelled trials per class from the target subject added to the training set (same subjects and same test trials at every size).

**Streaming pipeline** (`rt_pipeline.py`, numpy + scipy only): 4 s buffer (first 1 s filter warm-up), causal 4th-order Butterworth 8–30 Hz, CAR over 16 channels, covariance with 5% shrinkage, online re-centering, tangent space (136 features), linear LDA.

**Leaks found and removed from our earlier pipeline.** CAR over 64 channels before selecting 16; a hard-coded fixed-16 set evaluated on subjects that helped choose it; a calibration sweep whose subject set silently changed; selection on a single train/test split. All headline numbers below come from the corrected pipeline; earlier outputs are listed in `SUPERSEDED.md`.

## 4. Results (subject-independent, mean over 5 folds)

### 4.1 Channel reduction

Riemannian + re-centering, accuracy vs number of channels (t-test channels):

| Task | 4 | 8 | 12 | 16 | 24 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| execution bilateral | 0.559 | 0.611 | 0.655 | 0.697 | 0.728 | 0.746 | 0.761 |
| execution unilateral | 0.554 | 0.595 | 0.613 | 0.617 | 0.655 | 0.666 | 0.700 |
| imagery bilateral | 0.563 | 0.603 | 0.613 | 0.614 | 0.648 | 0.650 | 0.653 |
| imagery unilateral | 0.540 | 0.573 | 0.596 | 0.594 | 0.602 | 0.622 | 0.629 |

16 vs 64 (t-test channels), paired over folds, Holm: significantly worse in three of four tasks (p ≤ 0.023); EEGNet 16 vs 64: worse in all four (p ≤ 0.047).

**Fixed sensorimotor 16 channels** (Riemannian + re-centering, variant chosen by nested CV):

| Task | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| execution bilateral | 0.777 | 0.780 | 0.775 | 0.777 | 0.861 |
| execution unilateral | 0.700 | 0.706 | 0.684 | 0.695 | 0.775 |
| imagery bilateral | 0.668 | 0.669 | 0.672 | 0.669 | 0.740 |
| imagery unilateral | 0.637 | 0.634 | 0.631 | 0.633 | 0.697 |

Versus the t-test 16-channel baseline the gain is +4.4 to +8.3 points (Holm-significant in two tasks; execution bilateral p = 0.053; imagery unilateral not significant). Versus 64 channels: no significant difference in any task (all Holm p ≥ 0.37). Almost all of the gain comes from the channel set; filter-bank features add about 3 points on execution bilateral only, and pooling execution and imagery data adds nothing reliable.

### 4.2 Models

| Model (16 ch) | exec-bil | exec-uni | img-bil | img-uni | Parameters |
|---|---|---|---|---|---|
| FBCSP-LDA (t-test ch) | 0.565 | 0.544 | 0.552 | 0.555 | 16 features |
| Riemannian TS-LDA (t-test ch) | 0.625 | 0.574 | 0.560 | 0.553 | 136 features |
| Riemannian + re-centering (t-test ch) | 0.697 | 0.617 | 0.614 | 0.594 | 136 features |
| EEGNet, sensorimotor ch | 0.722 | 0.639 | 0.643 | 0.607 | 1,842 |
| EEGNet compact, sensorimotor ch | 0.718 | 0.637 | 0.656 | 0.613 | 858 |
| EEGNet 64 ch | 0.705 | 0.622 | 0.640 | 0.600 | 2,610 |

EEGNet on the sensorimotor set beats EEGNet on t-test channels in all four tasks (p ≤ 0.017) and is statistically equal to 64 channels (p ≥ 0.16). Euclidean alignment did not help EEGNet. The compact network is within about 1.5 points of the standard one at 47% of the parameters.

### 4.3 Adaptation

Streaming ablation, 16 sensorimotor channels, accuracy:

| Condition | exec-bil | exec-uni | img-bil | img-uni |
|---|---|---|---|---|
| zero-phase filter + batch re-centering | 0.744 | 0.695 | 0.660 | 0.631 |
| causal filter + batch re-centering | 0.755 | 0.695 | 0.666 | 0.635 |
| causal filter, no adaptation | 0.698 | 0.647 | 0.600 | 0.597 |
| causal + online re-centering, w0 = 5 (pre-registered) | 0.728 | 0.632 | 0.624 | 0.604 |
| **causal + online re-centering, nested-selected** | **0.756** | **0.687** | **0.659** | **0.627** |

Causal filtering costs nothing. Online and batch re-centering differ by under 1 point (not significant). Adaptation beats no adaptation by 3–6 points (Holm-significant for imagery bilateral; the others have raw p 0.02–0.07 with only five folds). Our pre-registered prior weight was too strong; the rescaled prior chosen by inner CV fixed this. Cold start is real: with the pre-registered setting, execution unilateral scores 0.55 on the first five windows and 0.67 after twenty.

Few-shot calibration (10 labelled trials per class): execution bilateral 0.693 → 0.743, imagery bilateral 0.611 → 0.632 (Holm p < 0.001 and 0.022); execution and imagery unilateral unchanged (p = 1.0).

## 5. Negative results

- **Per-subject channel selection** (pre-registered; scores blended from the sensorimotor prior and the subject's own 10 + 10 calibration trials): λ = 0.5 is not better in any task and significantly worse in execution bilateral and imagery unilateral (Holm p = 0.014 and 0.033); pure personalisation (λ = 1) is worse in all four tasks; random 16 channels is worse still. Earlier within-subject fixed-vs-personalised comparison (20 calibration trials, no training subjects) was also null.
- Euclidean alignment for EEGNet: no gain.
- int8 quantisation of a 5 KB model: slower and larger than fp32.

## 6. Deployment and real-time capability

Held-out replay of the streaming pipeline (21–22 unseen users per task): accuracy 0.806 / 0.709 / 0.635 / 0.665 (exec-bil / exec-uni / img-bil / img-uni), decision latency about 0.5 ms mean and under 0.9 ms at the 95th percentile per 3 s window, on a laptop CPU, single process. The EEGNet-compact ONNX model is 5 KB and runs in about 0.5 ms. Raspberry Pi timing: see `PI_RUN_INSTRUCTIONS.md` (not yet measured).

## 7. Limitations

- Replay of cue-locked epochs, not a live continuous self-paced stream; no rest/idle detection.
- Online re-centering needs a short unlabeled warm-up; offline tables use batch re-centering, which is transductive.
- Only five folds, so paired tests have limited power; we report effect sizes and fold counts as well.
- The sensorimotor channel set was fixed from anatomy before results, but it was not compared against other anatomical sets.
- Imagery tasks remain modest (0.63–0.67); imagery-unilateral recall is 0.555.
- Latency was measured on a laptop, not on a Raspberry Pi; Emotiv hardware was not used (a 14-channel Emotiv montage simulated from PhysioNet reaches only 0.55–0.57 because it lacks C3/Cz/C4).

## 8. Future work

Raspberry Pi timing and power; a continuous-stream demonstration with rest-state detection; own EEG collection with a consumer headset under ethics approval (protocol: cue-locked left/right imagery, ≥ 40 trials per class, same preprocessing); comparison of several anatomical montages; more folds or repeated cross-validation.

## 9. Reproducibility (file map)

| Purpose | File |
|---|---|
| Rigorous 5-fold benchmark (classical) | `run_complete_rigorous_benchmark.py`, `outputs_rigorous_benchmark/table_a_*.csv` |
| Stats for Tables A/B | `stats_rigorous.py` |
| EEGNet 5-fold (GPU) | `run_eegnet_gpu.py`, `run_eegnet_sm.py`, `outputs_eegnet_groupkfold/` |
| Channel sweep, few-shot | `run_channel_sweep.py`, `run_fewshot.py` |
| Nested sensorimotor experiment | `run_nested_improve.py`, `stats_nested.py` |
| Adaptive channels | `run_adaptive_channels.py`, `stats_adaptive.py` |
| Streaming pipeline and evaluation | `rt_pipeline.py`, `run_realtime_eval.py`, `run_realtime_nested.py`, `stats_streaming.py` |
| Caches | `build_epoch_cache.py`, `build_raw_cache.py` |
| ONNX export | `export_deploy_sm.py`, `pi_infer.py` |
| All final tables | `outputs_final/RESULTS.md` |
| Defence notes | `DEFENSE_NOTES.md` |
