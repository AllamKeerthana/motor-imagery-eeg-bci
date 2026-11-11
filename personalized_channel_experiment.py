"""
Per-subject adaptive channel selection experiment.

Compares, per subject, under an identical calibration/test split:
  - FIXED channels  : the global 16-channel set already selected in your
                       subject-independent FBCSP-LDA run
  - PERSONALIZED     : 16 channels selected using ONLY that subject's own
                       calibration trials

Leakage guardrails:
  1. The calibration/test split happens BEFORE any scoring or selection.
  2. Channel scoring uses ONLY calibration labels/data — never test data.
  3. The FIXED channel set is the one already produced by your existing
     subject-independent pipeline.
  4. Both conditions (fixed vs personalized) use the exact same
     calibration/test split per subject, so the comparison is paired.
  5. No test-set trial is ever used for training, scoring, or model selection.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
import mne
from mne.decoding import CSP

RANDOM_SEED = 42
N_CALIB_PER_CLASS = 10   # default 10 trials per class (20 calibration trials total)
K_CHANNELS = 16           # matches fixed-16 setup
FREQ_BANDS = [(8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0)]

DATA_DIR = Path(r'C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0\files')
OUTPUT_DIR = Path(r'C:\SEM_7\PhaseProject\outputs_personalized')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_FAMILIES = {
    'execution_unilateral': [3, 7, 11],
    'imagery_unilateral': [4, 8, 12],
    'execution_bilateral': [5, 9, 13],
    'imagery_bilateral': [6, 10, 14],
}

# Global fixed 16 channels from verified subject-independent baseline runs
FIXED_CHANNELS = {
    'execution_unilateral': ["C3", "Cz", "Cp5", "Cp1", "Cp4", "Af3", "F3", "F6", "T7", "T8", "Pz", "P2", "P4", "P6", "Po7", "Po3"],
    'imagery_unilateral': ["Fc3", "Fc2", "Fc6", "C3", "C1", "Cz", "Cp6", "Af3", "F1", "T7", "T10", "Tp8", "P3", "P6", "O1", "Iz"],
    'execution_bilateral': ["Fc3", "C3", "Cz", "Cp3", "Cp2", "Cp6", "F3", "Ft7", "T9", "P7", "P5", "P4", "Po7", "Poz", "Po8", "O1"],
    'imagery_bilateral': ["Fc3", "Fcz", "Fc2", "C3", "Cz", "Fpz", "F3", "F2", "Ft8", "T7", "T9", "P3", "P1", "P6", "Po7", "Poz"],
}


def seed_everything(seed=RANDOM_SEED):
    np.random.seed(seed)
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')


def load_subject_epochs(subject_id, task_group):
    """
    Loads EDF recordings for a subject and task family.
    Returns X (trials, channels, time), y (labels), ch_names, sfreq (160.0).
    """
    runs = TASK_FAMILIES[task_group]
    subject_dir = DATA_DIR / f'S{subject_id:03d}'
    raws = []
    for run in runs:
        path = subject_dir / f'S{subject_id:03d}R{run:02d}.edf'
        if path.exists():
            raws.append(mne.io.read_raw_edf(path, preload=True, verbose=False))
    if not raws:
        return None
    raw = mne.concatenate_raws(raws)
    if raw.info['sfreq'] != 160.0:
        raw.resample(160.0, npad='auto', verbose=False)
    raw.rename_channels(lambda name: name.strip('.'))
    raw.set_montage('standard_1005', on_missing='ignore')
    raw.filter(8.0, 30.0, fir_design='firwin', skip_by_annotation='edge', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = {name: code for name, code in event_id.items() if name in ('T1', 'T2')}
    if len(event_id) != 2:
        return None
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=-1.0, tmax=4.0,
        baseline=(-1.0, 0.0), picks='eeg', preload=True,
        reject_by_annotation=True, verbose=False,
    ).crop(tmin=1.0, tmax=4.0)
    if len(epochs) == 0:
        return None
    labels = np.sort(np.unique(epochs.events[:, -1]))
    y = (epochs.events[:, -1] == labels[1]).astype(np.int64)
    return epochs.get_data(copy=False).astype(np.float32), y, epochs.ch_names, 160.0


def calibration_test_split(X, y, n_calib_per_class, seed=RANDOM_SEED):
    """
    Picks a stratified calibration set and leaves the rest as held-out test.
    """
    rng = np.random.RandomState(seed)
    calib_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = min(n_calib_per_class, len(cls_idx) - 1)
        calib_idx.extend(cls_idx[:n])
    calib_idx = np.array(sorted(calib_idx))
    test_idx = np.array([i for i in range(len(y)) if i not in set(calib_idx)])
    return calib_idx, test_idx


def bandpass_fast(X, low, high, sfreq=160.0):
    """Fast bandpass filter across time axis using SciPy Butterworth SOS."""
    sos = butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def score_channels(X_calib, y_calib, sfreq=160.0, freq_band=(8.0, 30.0)):
    """
    Per-channel band-power separability using absolute two-sample t-statistic.
    Calculated purely on calibration data.
    """
    n_trials, n_channels, n_times = X_calib.shape
    filtered = bandpass_fast(X_calib, freq_band[0], freq_band[1], sfreq=sfreq)
    band_power = np.mean(filtered ** 2, axis=2)  # trial x channel

    classes = np.unique(y_calib)
    assert len(classes) == 2, "Binary T1/T2 labels required."
    a = band_power[y_calib == classes[0]]
    b = band_power[y_calib == classes[1]]

    t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    scores = np.abs(t_stat)
    return np.nan_to_num(scores, nan=0.0)


def select_top_k(scores, ch_names, k):
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    return sorted(top_idx.tolist()), [ch_names[i] for i in top_idx]


def train_eval_condition(X_calib, y_calib, X_test, y_test, sfreq, ch_idx):
    """
    Trains CSP (per band) + shrinkage LDA on calibration data ONLY,
    evaluates on held-out test data.
    """
    Xc = X_calib[:, ch_idx, :]
    Xt = X_test[:, ch_idx, :]

    train_feats, test_feats = [], []
    for lo, hi in FREQ_BANDS:
        Xcb = bandpass_fast(Xc, lo, hi, sfreq=sfreq)
        Xtb = bandpass_fast(Xt, lo, hi, sfreq=sfreq)

        csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
        train_feats.append(csp.fit_transform(Xcb, y_calib))
        test_feats.append(csp.transform(Xtb))

    train_X = np.concatenate(train_feats, axis=1)
    test_X = np.concatenate(test_feats, axis=1)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(train_X, y_calib)
    
    preds = clf.predict(test_X)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))
    return acc, f1


def run_experiment_for_family(subject_ids, task_group, k=K_CHANNELS, n_calib_per_class=N_CALIB_PER_CLASS):
    fixed_channel_names = FIXED_CHANNELS[task_group]
    results = []

    print(f"\n=========================================================================", flush=True)
    print(f"RUNNING PERSONALIZED CHANNEL EXPERIMENT: {task_group.upper()}", flush=True)
    print(f"Configuration: N_calib_per_class={n_calib_per_class} (Total={2*n_calib_per_class}), K={k}", flush=True)
    print(f"=========================================================================", flush=True)

    for subj in subject_ids:
        res = load_subject_epochs(subj, task_group)
        if res is None:
            continue
        X, y, ch_names, sfreq = res

        calib_idx, test_idx = calibration_test_split(X, y, n_calib_per_class)
        if len(test_idx) < 4:
            continue

        X_calib, y_calib = X[calib_idx], y[calib_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 1. Personalized channels (scored from calibration data only)
        scores = score_channels(X_calib, y_calib, sfreq)
        pers_idx, pers_names = select_top_k(scores, ch_names, k)

        # 2. Fixed channels (reuse global selection)
        fixed_idx = [ch_names.index(c) for c in fixed_channel_names if c in ch_names]

        # Evaluate both conditions on exact same split
        acc_fixed, f1_fixed = train_eval_condition(X_calib, y_calib, X_test, y_test, sfreq, fixed_idx)
        acc_pers, f1_pers = train_eval_condition(X_calib, y_calib, X_test, y_test, sfreq, pers_idx)

        # Compute Jaccard Overlap between personalized and fixed channels
        overlap = len(set(pers_names).intersection(set(fixed_channel_names))) / float(len(set(pers_names).union(set(fixed_channel_names))))

        rec = {
            "subject": subj,
            "task_group": task_group,
            "n_calib_per_class": n_calib_per_class,
            "total_calib_trials": len(calib_idx),
            "n_test_trials": len(test_idx),
            "acc_fixed_channels": acc_fixed,
            "acc_personalized_channels": acc_pers,
            "f1_fixed_channels": f1_fixed,
            "f1_personalized_channels": f1_pers,
            "delta_accuracy": acc_pers - acc_fixed,
            "delta_f1": f1_pers - f1_fixed,
            "channel_jaccard_overlap": overlap,
            "personalized_channels": pers_names,
            "fixed_channels": fixed_channel_names,
        }
        results.append(rec)
        print(f"[{task_group}] S{subj:03d} | Fixed: {acc_fixed*100:.1f}% | Personalized: {acc_pers*100:.1f}% | Delta: {rec['delta_accuracy']*100:+.1f}% | Overlap: {overlap:.2f}", flush=True)

    return results


def summarize_results(results):
    if not results:
        return {}
    df = pd.DataFrame(results)
    deltas = df["delta_accuracy"].values
    fixed = df["acc_fixed_channels"].values
    pers = df["acc_personalized_channels"].values

    n_improved = int(np.sum(deltas > 0))
    n_tied = int(np.sum(deltas == 0))
    n_degraded = int(np.sum(deltas < 0))
    n_total = len(deltas)

    stat, p_val = None, None
    if n_total >= 6 and not np.all(deltas == 0):
        try:
            stat, p_val = stats.wilcoxon(pers, fixed)
        except Exception:
            pass

    print("\n-------------------------------------------------------------------------", flush=True)
    print(f"EXPERIMENT SUMMARY: {results[0]['task_group'].upper()}", flush=True)
    print("-------------------------------------------------------------------------", flush=True)
    print(f"Subjects evaluated                 : {n_total}", flush=True)
    print(f"Mean Accuracy, Fixed Channels      : {fixed.mean()*100:.2f}% (std: {fixed.std()*100:.2f}%)", flush=True)
    print(f"Mean Accuracy, Personalized Chs    : {pers.mean()*100:.2f}% (std: {pers.std()*100:.2f}%)", flush=True)
    print(f"Mean Accuracy Delta (Pers - Fixed) : {deltas.mean()*100:+.2f}%", flush=True)
    print(f"Subjects Improved / Tied / Degraded: {n_improved} / {n_tied} / {n_degraded} ({n_improved/n_total*100:.1f}% improved)", flush=True)
    if p_val is not None:
        print(f"Wilcoxon Signed-Rank Test p-value  : {p_val:.5f} ({'Statistically Significant (p < 0.05)' if p_val < 0.05 else 'Not Statistically Significant'})", flush=True)
    print("-------------------------------------------------------------------------\n", flush=True)

    summary_dict = {
        "task_group": results[0]["task_group"],
        "n_subjects": n_total,
        "mean_acc_fixed": float(fixed.mean()),
        "mean_acc_personalized": float(pers.mean()),
        "mean_delta_acc": float(deltas.mean()),
        "n_improved": n_improved,
        "n_tied": n_tied,
        "n_degraded": n_degraded,
        "improvement_ratio": float(n_improved / n_total),
        "wilcoxon_p_value": float(p_val) if p_val is not None else None,
    }
    return summary_dict


def run_full_personalized_study():
    seed_everything(RANDOM_SEED)
    subject_ids = list(range(1, 110))
    
    all_experiment_records = []
    all_summaries = []

    # 1. Main comparison across all 4 task groups (N_calib_per_class=10)
    for family in TASK_FAMILIES.keys():
        records = run_experiment_for_family(subject_ids, family, k=16, n_calib_per_class=10)
        summary = summarize_results(records)
        all_experiment_records.extend(records)
        all_summaries.append(summary)

    # 2. Calibration size sweep for imagined_unilateral (N_calib_per_class in [5, 10, 15, 20])
    print("\n=========================================================================", flush=True)
    print("STARTING CALIBRATION SAMPLE SIZE SWEEP (Imagined Unilateral Task)", flush=True)
    print("=========================================================================", flush=True)
    sweep_records = []
    for n_calib in [5, 10, 15, 20]:
        recs = run_experiment_for_family(subject_ids, "imagery_unilateral", k=16, n_calib_per_class=n_calib)
        summ = summarize_results(recs)
        summ["sweep_n_calib_per_class"] = n_calib
        sweep_records.append(summ)

    # Save outputs
    df_all = pd.DataFrame(all_experiment_records)
    df_all.to_csv(OUTPUT_DIR / "personalized_channels_all_subjects.csv", index=False)
    
    (OUTPUT_DIR / "personalized_channels_all_subjects.json").write_text(
        json.dumps(all_experiment_records, indent=2), encoding="utf-8"
    )
    
    (OUTPUT_DIR / "personalized_experiment_summary.json").write_text(
        json.dumps({"task_summaries": all_summaries, "calibration_size_sweep": sweep_records}, indent=2),
        encoding="utf-8"
    )
    
    print("\n[COMPLETE] All personalized channel experiments finished successfully!")
    print(f"Results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_full_personalized_study()
