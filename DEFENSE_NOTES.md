# Defence notes — what we claim, the evidence, and likely questions

Title: *Adaptive Sparse-Channel EEG Preprocessing and Lightweight AI Classification for Real-Time Motor Imagery BCI*

All numbers: PhysioNet EEGMMI, 109 subjects, subject-independent 5-fold GroupKFold (test subjects never seen in training). Evidence files are under `outputs_rigorous_benchmark/`, `outputs_eegnet_groupkfold/`, `outputs_final/`.

## Title phrase -> claim -> evidence

| Phrase | Claim we make | Evidence | Limit to admit |
|---|---|---|---|
| Sparse-channel | A FIXED 16-channel sensorimotor set (FC3,FCz,FC4,C5-C6,CP3,CPz,CP4,P3,Pz,P4) is statistically equal to 64 channels; data-driven (t-test) 16 channels lose 4-8 points. | `table_g_nested_selected.csv`, `table_e_channel_sweep.csv`, `outputs_final/stats_nested_vs_baselines.csv`, `table_c_eegnet_sm.csv` | The set was fixed from anatomy before results; 5 folds = low statistical power. |
| Adaptive preprocessing | The adaptation is label-free per-subject re-centering of covariances (+4 to +7 points over no adaptation) with an ONLINE running-mean version that keeps almost all of the benefit; a few labelled trials help bilateral tasks only. | `table_j_streaming.csv`, `table_k_streaming_nested.csv`, `outputs_final/stats_streaming.csv`, `table_f_fewshot.csv` | Needs a short unlabeled warm-up (cold start is worse). Per-subject CHANNEL selection was tested and does NOT help (`table_i_adaptive_channels_*.csv`). |
| Lightweight AI | EEGNet-compact 858 parameters (5 KB ONNX) and a 136-feature Riemannian+LDA model. | `table_c_eegnet_sm.csv`, `deploy/*_compact_sm`, `rt_pipeline.py` | int8 quantisation was slower and larger for a 5 KB model; fp32 recommended. |
| Real-time | Streaming pipeline (causal filter, online re-centering, LDA) decides in about 0.5 ms (p95 < 0.9 ms) per 3 s window on a laptop CPU; replay matches the offline evaluation. | `python rt_pipeline.py deploy_rt/<task>` | Replay of cue-locked epochs, NOT a live continuous self-paced stream; no idle/rest detection; Raspberry Pi timing still to be measured. Say "real-time capable (measured latency 0.5 ms vs 3 s window)". |
| Motor imagery | Imagery accuracy 0.63-0.66 (execution 0.69-0.76). | `table_k_streaming_nested.csv` | Imagery is modest; recall is low for imagery_unilateral (0.555). |

## Headline numbers (mean over 5 folds, subject-independent)

Offline, 16 sensorimotor ch, Riemannian + re-centering, nested variant selection: exec-bilateral 0.777, exec-unilateral 0.700, imagery-bilateral 0.668, imagery-unilateral 0.637 (`table_g_nested_selected.csv`).
Streaming (causal filter + online re-centering, nested config): 0.756 / 0.687 / 0.659 / 0.627 (`table_k_streaming_nested.csv`).

## Streaming ablation (what each step costs)

| Condition | exec-bil | exec-uni | img-bil | img-uni |
|---|---|---|---|---|
| A zero-phase filter + batch re-centering | 0.744 | 0.695 | 0.660 | 0.631 |
| B causal filter + batch re-centering | 0.755 | 0.695 | 0.666 | 0.635 |
| N causal, no adaptation | 0.698 | 0.647 | 0.600 | 0.597 |
| C pre-registered online (w0=5) | 0.728 | 0.632 | 0.624 | 0.604 |
| K online, config chosen by nested CV | 0.756 | 0.687 | 0.659 | 0.627 |

Causal filtering costs nothing (A vs B). Online vs batch re-centering differs by < 1 point (not significant). The pre-registered w0=5 was worse; the amplitude-rescaled prior chosen by nested CV fixes that.

## Integrity points (say these proactively)

- Test subjects never used for training, channel selection, normalisation, early stopping or model/variant selection. Variant and online-config selection use inner subject-wise CV inside the training fold.
- Leaks found and fixed in our earlier pipeline: CAR over 64 channels before selecting 16; a hard-coded "fixed-16" evaluated on the subjects that chose it; a calibration sweep that silently changed the subject set. Legacy outputs (`outputs_personalized*`, `EEG-DS/outputs_*`) are superseded.
- Multiple comparisons: Holm correction within each declared family. With 5 folds, tests are low-powered; we report effect sizes and folds-better counts too.
- Re-centering is transductive in the offline tables (uses the test user's unlabeled trials); the streaming tables remove that by making it causal/online.
- Negative results reported: per-subject channel selection (worse than fixed; random-16 control worse still), Euclidean alignment did not help EEGNet, personalization from 20 trials did not help.

## Likely questions

1. **Why is your accuracy only ~0.65-0.78?** Cross-subject (zero calibration) motor imagery on PhysioNet is hard; published cross-subject numbers on this dataset are in this range or lower for 2-class tasks, and we use 3 s windows with no per-user training. Execution tasks are higher than imagery, as expected.
2. **Why is it "adaptive" if channel selection did not help?** The adaptation is in the preprocessing: label-free per-user re-centering that runs online. We tested adaptive channel selection as a hypothesis, pre-registered, and report it as a negative result.
3. **Is 16 fixed channels really as good as 64?** Statistically indistinguishable in our 5 folds (p >= 0.16 for EEGNet, >= 0.37 for Riemannian after correction). We do not claim superiority; with more folds the test would be more sensitive.
4. **Did you tune on test data?** No. Selections use inner subject-wise CV inside training folds; the pre-registered w0=5 result is reported next to the nested one.
5. **Is it really real-time?** The pipeline is causal and decides in ~0.5 ms on a laptop vs a 3 s window; replay equals the offline evaluation. It is a cue-locked replay, not a live stream, and the Pi measurement is pending.
6. **Why not a bigger deep model?** 858-parameter EEGNet is within ~1.5 points of the 1,842-parameter one; accuracy is limited by cross-subject variability, not model capacity.
7. **What about the Emotiv headset?** A 14-channel Emotiv montage simulated from PhysioNet scored 0.55-0.57 (no C3/Cz/C4); own-data collection is future work, no results claimed.
