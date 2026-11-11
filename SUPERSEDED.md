# Superseded results and documents

These files were produced by the earlier pipeline. They are kept for history but must NOT be cited. Reasons are given per item.

| Item | Why superseded | Replaced by |
|---|---|---|
| `outputs_personalized/`, `PERSONALIZED_CHANNEL_RESULTS.md`, `personalized_channel_experiment.py` | Fixed-16 channel lists were hard-coded from one split and then evaluated on all 109 subjects (includes the selection subjects); CAR applied over 64 channels before selecting 16; calibration sweep changed the subject set (n = 109 / 109 / 109 / 100). | `table_b_withinsubject_109subjs.csv`, `table_i_adaptive_channels_*.csv`, `table_f_fewshot.csv` |
| `outputs_personalized_riemann/`, `personalized_channel_riemann_experiment.py` | Same issues as above. | as above |
| `EEG-DS/outputs_all_tasks_independent/`, `outputs_all_tasks_fbcsp_independent/`, `outputs_all_tasks_subject_wise/`, `outputs_all_tasks_fbcsp_subject_wise/`, `outputs_all_tasks_riemann_independent/` | Single train/test split (22 test subjects), CAR over 64 channels before 16-channel selection; EEGNet subject-wise covered only 20 subjects (about 9 test trials each; chance level). | `outputs_eegnet_groupkfold/`, `outputs_rigorous_benchmark/` |
| `EEG-DS/outputs/`, `outputs_research/`, `outputs_subject_independent/`, notebooks `EEGNet_Subject_Independent.ipynb`, `Motor_Imagery_EEG_Research_Project.ipynb`, `Untitled3-3_updated_corrected.ipynb` | One task only, single split; `final_metrics.json` is the same run copied three times (0.5838). | `outputs_eegnet_groupkfold/table_c_eegnet_gpu.csv` |
| `Motor_Imagery_EEG_Project_Report.*`, `PROJECT_*.md`, `ALL_TASKS_ADAPTIVE_SPARSE_METHOD.md`, `PERSONALIZED_ADAPTIVE_SPARSE_EEG_DOCUMENTATION.md` | Contain claims not supported by the corrected results (for example "16 channels is enough", and that personalised channels improve accuracy). | `FINAL_REPORT.md`, `outputs_final/RESULTS.md` |
| `groupkfold_unified_benchmark.py` | Intermediate version of the rigorous benchmark; no stats or Emotiv arm. | `run_complete_rigorous_benchmark.py` |
| `outputs_rigorous_benchmark/table_a_*.csv` columns `acc_16_*` | 16-channel scores were computed on channels ranked on CAR-64 data (minor leak, fixed in later runs). Use them only as the "t-test 16" baseline with that caveat; the corrected equivalents are in `table_d_classical_metrics.csv` and `table_e_channel_sweep.csv`. | `table_d_*`, `table_e_*` |
