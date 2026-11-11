from pathlib import Path
import json
import warnings

import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42
DATA_DIR = Path(r'C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0\files')
TASK_FAMILIES = {
    'execution_unilateral': [3, 7, 11],
    'imagery_unilateral': [4, 8, 12],
    'execution_bilateral': [5, 9, 13],
    'imagery_bilateral': [6, 10, 14],
}
BANDS = [(8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0)]


def seed_everything():
    np.random.seed(SEED)
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore')


def load_subject_family(subject_id, runs):
    subject_dir = DATA_DIR / f'S{subject_id:03d}'
    all_X, all_y, channel_names = [], [], None
    for run in runs:
        path = subject_dir / f'S{subject_id:03d}R{run:02d}.edf'
        if not path.exists():
            continue
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if raw.info['sfreq'] != 160.0:
            raw.resample(160.0, npad='auto', verbose=False)
        raw.rename_channels(lambda name: name.strip('.'))
        raw.set_montage('standard_1005', on_missing='ignore')
        raw.filter(8.0, 30.0, fir_design='firwin', skip_by_annotation='edge', verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_id = {name: code for name, code in event_id.items() if name in ('T1', 'T2')}
        if len(event_id) != 2:
            continue
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1.0, tmax=4.0,
                            baseline=(-1.0, 0.0), picks='eeg', preload=True,
                            reject_by_annotation=True, verbose=False).crop(tmin=1.0, tmax=4.0)
        labels = np.sort(np.unique(epochs.events[:, -1]))
        all_X.append(epochs.get_data(copy=False).astype(np.float32))
        all_y.append((epochs.events[:, -1] == labels[1]).astype(np.int64))
        channel_names = epochs.ch_names
    if not all_X:
        return None
    return np.concatenate(all_X), np.concatenate(all_y), channel_names


def select_channels(X_train, y_train, names, n_channels):
    log_variance = np.log(np.var(X_train, axis=2) + 1e-8)
    scores, _ = f_classif(log_variance, y_train)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    indices = np.argsort(scores)[::-1][:n_channels]
    indices = np.sort(indices)
    return indices, [names[index] for index in indices]


def bandpass(X, low, high, sfreq=160.0):
    sos = butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def fit_fbcsp(X_train, y_train, n_components=4):
    filters = []
    for low, high in BANDS:
        train_band = bandpass(X_train, low, high)
        csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
        csp.fit(train_band, y_train)
        filters.append(csp)
    return filters


def transform_fbcsp(X, filters):
    features = []
    for (low, high), csp in zip(BANDS, filters):
        features.append(csp.transform(bandpass(X, low, high)))
    return np.concatenate(features, axis=1)


def score_split(X, y, train_idx, valid_idx, test_idx, channels, task_family, subject=None):
    selected = X[:, channels]
    filters = fit_fbcsp(selected[train_idx], y[train_idx])
    train_features = transform_fbcsp(selected[train_idx], filters)
    valid_features = transform_fbcsp(selected[valid_idx], filters)
    test_features = transform_fbcsp(selected[test_idx], filters)
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    model.fit(train_features, y[train_idx])
    valid_accuracy = accuracy_score(y[valid_idx], model.predict(valid_features))
    predictions = model.predict(test_features)
    return {
        'protocol': 'subject-wise' if subject is not None else 'subject-independent',
        'task_family': task_family,
        'subject': None if subject is None else int(subject),
        'validation_accuracy': float(valid_accuracy),
        'accuracy': float(accuracy_score(y[test_idx], predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(y[test_idx], predictions)),
        'macro_f1': float(f1_score(y[test_idx], predictions, average='macro')),
        'n_channels': int(len(channels)),
        'channels': list(channels),
        'n_test_trials': int(len(test_idx)),
    }


def run_independent(subject_ids, output_dir, n_channels=16):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for family, runs in TASK_FAMILIES.items():
        records = [(sid, *loaded) for sid in subject_ids if (loaded := load_subject_family(sid, runs)) is not None]
        assert records, f'No usable recordings for {family}'
        X = np.concatenate([r[1] for r in records]); y = np.concatenate([r[2] for r in records])
        assert set(np.unique(y)) == {0, 1}, f'{family} must contain both classes'
        groups = np.concatenate([[r[0]] * len(r[2]) for r in records])
        names = records[0][3]
        train_valid, test = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED).split(X, y, groups))
        train_rel, valid_rel = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED + 1).split(X[train_valid], y[train_valid], groups[train_valid]))
        train, valid = train_valid[train_rel], train_valid[valid_rel]
        assert set(train).isdisjoint(valid) and set(train).isdisjoint(test) and set(valid).isdisjoint(test)
        channels, selected_names = select_channels(X[train], y[train], names, n_channels)
        result = score_split(X, y, train, valid, test, channels, family)
        result.update({'channels': selected_names, 'n_subjects': int(len(np.unique(groups))), 'test_subjects': [int(v) for v in sorted(np.unique(groups[test]))]})
        results.append(result)
        (output_dir / 'all_tasks_fbcsp_independent.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(result, flush=True)
    (output_dir / 'all_tasks_fbcsp_independent.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return pd.DataFrame(results)


def run_subject_wise(subject_ids, output_dir, n_channels=16):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for family, runs in TASK_FAMILIES.items():
        for subject in subject_ids:
            loaded = load_subject_family(subject, runs)
            if loaded is None:
                continue
            X, y, names = loaded
            assert set(np.unique(y)) == {0, 1}, f'{family} subject {subject} must contain both classes'
            indices = np.arange(len(y))
            train_valid, test = train_test_split(indices, test_size=.20, stratify=y, random_state=SEED)
            train, valid = train_test_split(train_valid, test_size=.25, stratify=y[train_valid], random_state=SEED)
            assert set(train).isdisjoint(valid) and set(train).isdisjoint(test) and set(valid).isdisjoint(test)
            channels, selected_names = select_channels(X[train], y[train], names, n_channels)
            result = score_split(X, y, train, valid, test, channels, family, subject)
            result['channels'] = selected_names
            results.append(result)
            pd.DataFrame(results).to_csv(output_dir / 'all_tasks_fbcsp_subject_wise.csv', index=False)
            print(result, flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / 'all_tasks_fbcsp_subject_wise.csv', index=False)
    return frame
