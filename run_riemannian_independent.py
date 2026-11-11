"""
Run Riemannian TS-LDA under the exact same subject-independent protocol as FBCSP and EEGNet.
Saves results to outputs_all_tasks_riemann_independent.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import mne

from adaptive_sparse_fbcsp import (
    load_subject_family,
    select_channels,
    SEED,
    TASK_FAMILIES,
    DATA_DIR
)

def bandpass_fast(X, low=8.0, high=30.0, sfreq=160.0):
    sos = butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)

def run_riemann_independent(subject_ids=range(1, 110), n_channels=16):
    output_dir = Path(r'c:\SEM_7\PhaseProject\outputs_all_tasks_riemann_independent')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')

    results = []
    for family, runs in TASK_FAMILIES.items():
        records = [(sid, *loaded) for sid in subject_ids if (loaded := load_subject_family(sid, runs)) is not None]
        assert records, f'No usable recordings for {family}'
        X = np.concatenate([r[1] for r in records])
        y = np.concatenate([r[2] for r in records])
        groups = np.concatenate([[r[0]] * len(r[2]) for r in records])
        names = records[0][3]

        train_valid, test = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED).split(X, y, groups))
        train_rel, valid_rel = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED + 1).split(X[train_valid], y[train_valid], groups[train_valid]))
        train, valid = train_valid[train_rel], train_valid[valid_rel]

        channels, selected_names = select_channels(X[train], y[train], names, n_channels)
        
        # Filter 8-30 Hz on selected channels
        X_sel = X[:, channels]
        X_filt = bandpass_fast(X_sel, 8.0, 30.0)

        # Fit covariance on train
        cov = Covariances(estimator="lwf")
        cov_train = cov.fit_transform(X_filt[train])
        cov_valid = cov.transform(X_filt[valid])
        cov_test = cov.transform(X_filt[test])

        # Tangent space reference point on train only
        ts = TangentSpace(metric="riemann")
        ts_train = ts.fit_transform(cov_train)
        ts_valid = ts.transform(cov_valid)
        ts_test = ts.transform(cov_test)

        # Shrinkage LDA
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(ts_train, y[train])

        valid_acc = float(accuracy_score(y[valid], clf.predict(ts_valid)))
        preds = clf.predict(ts_test)
        test_acc = float(accuracy_score(y[test], preds))
        bal_acc = float(balanced_accuracy_score(y[test], preds))
        f1 = float(f1_score(y[test], preds, average='macro'))

        res = {
            'protocol': 'subject-independent',
            'model': 'Riemannian TS-LDA',
            'task_family': family,
            'n_channels': n_channels,
            'channels': selected_names,
            'validation_accuracy': valid_acc,
            'accuracy': test_acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': f1,
            'n_test_trials': int(len(test)),
            'n_subjects': int(len(np.unique(groups))),
            'test_subjects': [int(v) for v in sorted(np.unique(groups[test]))]
        }
        results.append(res)
        print(f"[{family}] Subject-Independent Riemannian TS-LDA -> Acc: {test_acc*100:.2f}%, BalAcc: {bal_acc*100:.2f}%, F1: {f1*100:.2f}%", flush=True)

    out_file = output_dir / 'all_tasks_riemann_independent.json'
    out_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.DataFrame(results).to_csv(output_dir / 'all_tasks_riemann_independent.csv', index=False)
    print("Saved to", out_file)
    return results

if __name__ == '__main__':
    run_riemann_independent()
