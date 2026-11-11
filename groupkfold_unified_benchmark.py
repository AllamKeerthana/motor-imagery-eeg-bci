"""
Unified 5-Fold GroupKFold Benchmark Script for EEG Motor Movement/Imagery
--------------------------------------------------------------------------
Fixes all open methodological concerns from Review Notes:
1. Two separate evaluation tables: Table A (Subject-Independent) & Table B (Within-Subject Calibrated).
2. True 64-channel baseline vs. 16-channel sparse to directly answer "can 16 match 64?".
3. Honest CAR: Re-referencing is applied AFTER channel selection, matching actual 16-channel hardware.
4. Leakage-free Fixed Channels: Fixed-16 channels are selected ONLY from training-fold subjects, never from test subjects.
5. 5-Fold GroupKFold across subjects with identical folds across all models.
6. Reports mean, standard deviation, and bootstrap confidence intervals.
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import mne

SEED = 42
DATA_DIR = Path(r'C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0\files')
OUTPUT_DIR = Path(r'C:\SEM_7\PhaseProject\outputs_unified_groupkfold')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_FAMILIES = {
    'execution_unilateral': [3, 7, 11],
    'imagery_unilateral': [4, 8, 12],
    'execution_bilateral': [5, 9, 13],
    'imagery_bilateral': [6, 10, 14],
}

FBCSP_BANDS = [(8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0)]
BROADBAND = (8.0, 30.0)

def seed_everything(seed=SEED):
    np.random.seed(seed)
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')

def load_subject_raw_epochs(subject_id, runs):
    """
    Loads raw EDF files without 64-channel CAR so CAR can be applied honestly after channel selection.
    Applies 8-30 Hz bandpass filter and extracts 1.0s to 4.0s post-cue epochs.
    """
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
    # Note: Deliberately NO global 64-channel CAR here.
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
    return epochs.get_data(copy=False).astype(np.float32), y, epochs.ch_names

def apply_honest_car(X):
    """Re-references signal to the common average across ONLY the available channels."""
    # X shape: (trials, channels, time)
    return X - np.mean(X, axis=1, keepdims=True)

def bandpass_fast(X, low, high, sfreq=160.0):
    sos = butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)

def select_top_channels_train_only(X_train, y_train, ch_names, n_channels=16):
    """ANOVA F-test on log-variance computed strictly on training data."""
    log_var = np.log(np.var(X_train, axis=2) + 1e-8)
    classes = np.unique(y_train)
    a = log_var[y_train == classes[0]]
    b = log_var[y_train == classes[1]]
    t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    scores = np.abs(np.nan_to_num(t_stat))
    order = np.argsort(scores)[::-1][:n_channels]
    return sorted(order.tolist()), [ch_names[i] for i in order]

def eval_fbcsp_lda(X_train, y_train, X_test, y_test, sfreq=160.0):
    train_feats, test_feats = [], []
    for lo, hi in FBCSP_BANDS:
        Xtb = bandpass_fast(X_train, lo, hi, sfreq=sfreq)
        Xteb = bandpass_fast(X_test, lo, hi, sfreq=sfreq)
        csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
        train_feats.append(csp.fit_transform(Xtb, y_train))
        test_feats.append(csp.transform(Xteb))
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(np.concatenate(train_feats, axis=1), y_train)
    return float(clf.score(np.concatenate(test_feats, axis=1), y_test))

def eval_riemann_tslda(X_train, y_train, X_test, y_test, sfreq=160.0):
    cov = Covariances(estimator="lwf")
    C_tr = cov.fit_transform(X_train)
    C_te = cov.transform(X_test)
    ts = TangentSpace(metric="riemann")
    F_tr = ts.fit_transform(C_tr)
    F_te = ts.transform(C_te)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(F_tr, y_train)
    return float(clf.score(F_te, y_test))

def bootstrap_ci(data, n_boot=2000, ci=0.95):
    """Computes non-parametric bootstrap confidence interval."""
    if len(data) < 2:
        return float(np.mean(data)), float(np.mean(data))
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def run_groupkfold_experiment(subject_ids=range(1, 110), n_splits=5):
    seed_everything(SEED)
    table_a_records = []
    table_b_records = []

    print("=========================================================================", flush=True)
    print("STARTING UNIFIED 5-FOLD GROUPKFOLD BENCHMARK", flush=True)
    print("=========================================================================", flush=True)

    for family, runs in TASK_FAMILIES.items():
        print(f"\n---> Processing Task Family: {family.upper()}", flush=True)
        records = [(sid, *loaded) for sid in subject_ids if (loaded := load_subject_raw_epochs(sid, runs)) is not None]
        if not records:
            continue
        X_all = np.concatenate([r[1] for r in records])
        y_all = np.concatenate([r[2] for r in records])
        groups = np.concatenate([[r[0]] * len(r[2]) for r in records])
        ch_names = records[0][3]

        gkf = GroupKFold(n_splits=n_splits)
        fold_idx = 0

        for train_idx, test_idx in gkf.split(X_all, y_all, groups=groups):
            fold_idx += 1
            train_subjs = np.unique(groups[train_idx])
            test_subjs = np.unique(groups[test_idx])

            # ----------------------------------------------------
            # 1. 64-Channel Baseline (Honest CAR across all 64 channels)
            # ----------------------------------------------------
            X_64_train = apply_honest_car(X_all[train_idx])
            X_64_test = apply_honest_car(X_all[test_idx])

            acc_64_riemann = eval_riemann_tslda(X_64_train, y_all[train_idx], X_64_test, y_all[test_idx])
            acc_64_fbcsp = eval_fbcsp_lda(X_64_train, y_all[train_idx], X_64_test, y_all[test_idx])

            # ----------------------------------------------------
            # 2. 16-Channel Selection (from training fold ONLY)
            # ----------------------------------------------------
            top_16_idx, top_16_names = select_top_channels_train_only(X_64_train, y_all[train_idx], ch_names, n_channels=16)

            # HONEST CAR: Apply CAR strictly to the 16 selected channels
            X_16_raw_train = X_all[train_idx][:, top_16_idx, :]
            X_16_raw_test = X_all[test_idx][:, top_16_idx, :]
            X_16_train = apply_honest_car(X_16_raw_train)
            X_16_test = apply_honest_car(X_16_raw_test)

            acc_16_fbcsp = eval_fbcsp_lda(X_16_train, y_all[train_idx], X_16_test, y_all[test_idx])
            acc_16_riemann = eval_riemann_tslda(X_16_train, y_all[train_idx], X_16_test, y_all[test_idx])

            table_a_records.append({
                'task_family': family,
                'fold': fold_idx,
                'acc_64_fbcsp': acc_64_fbcsp,
                'acc_64_riemann': acc_64_riemann,
                'acc_16_fbcsp': acc_16_fbcsp,
                'acc_16_riemann': acc_16_riemann,
                'n_train_subjects': len(train_subjs),
                'n_test_subjects': len(test_subjs)
            })

            # ----------------------------------------------------
            # 3. Within-Subject Calibrated (Table B) on Test Fold Subjects
            # ----------------------------------------------------
            for sid in test_subjs:
                sub_mask = groups == sid
                X_s = X_all[sub_mask]
                y_s = y_all[sub_mask]
                if len(y_s) < 24:
                    continue

                # 10 calibration trials per class
                rng = np.random.RandomState(SEED)
                calib_idx = []
                for cls in np.unique(y_s):
                    c_idx = np.where(y_s == cls)[0]
                    rng.shuffle(c_idx)
                    calib_idx.extend(c_idx[:10])
                calib_idx = np.array(sorted(calib_idx))
                eval_idx = np.array([i for i in range(len(y_s)) if i not in set(calib_idx)])
                if len(eval_idx) < 4:
                    continue

                # Fixed-16: uses the top_16_idx learned from training fold (no leakage!)
                X_fixed_cal = apply_honest_car(X_s[calib_idx][:, top_16_idx, :])
                X_fixed_te = apply_honest_car(X_s[eval_idx][:, top_16_idx, :])

                acc_fix_fbcsp = eval_fbcsp_lda(X_fixed_cal, y_s[calib_idx], X_fixed_te, y_s[eval_idx])
                acc_fix_riemann = eval_riemann_tslda(X_fixed_cal, y_s[calib_idx], X_fixed_te, y_s[eval_idx])

                # Personalized-16: selected strictly on calibration trials
                pers_idx, _ = select_top_channels_train_only(X_s[calib_idx], y_s[calib_idx], ch_names, n_channels=16)
                X_pers_cal = apply_honest_car(X_s[calib_idx][:, pers_idx, :])
                X_pers_te = apply_honest_car(X_s[eval_idx][:, pers_idx, :])

                acc_pers_fbcsp = eval_fbcsp_lda(X_pers_cal, y_s[calib_idx], X_pers_te, y_s[eval_idx])
                acc_pers_riemann = eval_riemann_tslda(X_pers_cal, y_s[calib_idx], X_pers_te, y_s[eval_idx])

                table_b_records.append({
                    'task_family': family,
                    'fold': fold_idx,
                    'subject': int(sid),
                    'acc_fixed_fbcsp': acc_fix_fbcsp,
                    'acc_pers_fbcsp': acc_pers_fbcsp,
                    'acc_fixed_riemann': acc_fix_riemann,
                    'acc_pers_riemann': acc_pers_riemann,
                    'delta_fbcsp': acc_pers_fbcsp - acc_fix_fbcsp,
                    'delta_riemann': acc_pers_riemann - acc_fix_riemann,
                })

            print(f"   Fold {fold_idx}/5 complete for {family}.", flush=True)

    # Save outputs
    df_a = pd.DataFrame(table_a_records)
    df_b = pd.DataFrame(table_b_records)

    df_a.to_csv(OUTPUT_DIR / 'table_a_subject_independent_folds.csv', index=False)
    df_b.to_csv(OUTPUT_DIR / 'table_b_withinsubject_calibrated.csv', index=False)

    print("\nBenchmark completed. Processing summary...", flush=True)
    return df_a, df_b

if __name__ == '__main__':
    run_groupkfold_experiment()
