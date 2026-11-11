from pathlib import Path
import copy
import json
import random
import warnings

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_selection import f_classif
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_DIR = Path(r'C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0\files')
TASK_FAMILIES = {
    'execution_unilateral': [3, 7, 11],
    'imagery_unilateral': [4, 8, 12],
    'execution_bilateral': [5, 9, 13],
    'imagery_bilateral': [6, 10, 14],
}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    mne.set_log_level('WARNING')
    warnings.filterwarnings('ignore', message="Montage name 'standard_1005' is deprecated")
    warnings.filterwarnings('ignore', message='Limited 1 annotation')
    warnings.filterwarnings('ignore', message='Concatenation of Annotations')


def load_subject_family(subject_id, runs):
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
    return epochs.get_data(copy=False).astype(np.float32), y, epochs.ch_names


def select_sparse_channels(X_train, y_train, channel_names, n_channels):
    if n_channels >= X_train.shape[1]:
        return np.arange(X_train.shape[1]), list(channel_names)
    log_variance = np.log(np.var(X_train, axis=2) + 1e-8)
    scores, _ = f_classif(log_variance, y_train)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    selected = np.argsort(scores)[::-1][:n_channels]
    selected = np.sort(selected)
    return selected, [channel_names[index] for index in selected]


class EEGNet(nn.Module):
    def __init__(self, channels, samples, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, channels, samples)).flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, 2)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


def make_loader(X, y, indices, shuffle, batch_size=64):
    selected = X[indices]
    normalized = (selected - selected.mean(axis=2, keepdims=True)) / (selected.std(axis=2, keepdims=True) + 1e-6)
    tensors = torch.tensor(normalized, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(y[indices], dtype=torch.long)
    return DataLoader(TensorDataset(tensors, targets), batch_size=batch_size, shuffle=shuffle)


def augment_training_batch(inputs):
    noise = torch.randn_like(inputs) * 0.01
    if inputs.shape[-1] > 20:
        shift = int(np.random.randint(-8, 9))
        inputs = torch.roll(inputs, shifts=shift, dims=-1)
    return inputs + noise


def fit_and_score(X, y, train_idx, valid_idx, test_idx, epochs=150, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = make_loader(X, y, train_idx, True, batch_size)
    valid_loader = make_loader(X, y, valid_idx, False, batch_size)
    test_loader = make_loader(X, y, test_idx, False, batch_size)
    model = EEGNet(X.shape[1], X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_valid = -np.inf
    wait = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = augment_training_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
        valid_accuracy = correct / len(valid_loader.dataset)
        scheduler.step(valid_accuracy)
        if valid_accuracy > best_valid:
            best_valid = valid_accuracy
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            predictions.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            truth.extend(yb.numpy())
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)
    return {
        'validation_accuracy': float(best_valid),
        'accuracy': float(accuracy_score(truth, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, predictions)),
        'macro_f1': float(f1_score(truth, predictions, average='macro')),
        'n_test_trials': int(len(truth)),
    }


def run_independent(subject_ids, output_dir, n_channels=16):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for family, runs in TASK_FAMILIES.items():
        records = []
        for subject_id in subject_ids:
            result = load_subject_family(subject_id, runs)
            if result is not None:
                X, y, names = result
                records.append((subject_id, X, y, names))
        if not records:
            continue
        X = np.concatenate([record[1] for record in records])
        y = np.concatenate([record[2] for record in records])
        groups = np.concatenate([[record[0]] * len(record[2]) for record in records])
        names = records[0][3]
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        train_valid_idx, test_idx = next(gss_test.split(X, y, groups=groups))
        gss_valid = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 1)
        train_rel, valid_rel = next(gss_valid.split(X[train_valid_idx], y[train_valid_idx], groups=groups[train_valid_idx]))
        train_idx = train_valid_idx[train_rel]
        valid_idx = train_valid_idx[valid_rel]
        selected, selected_names = select_sparse_channels(X[train_idx], y[train_idx], names, n_channels)
        metrics = fit_and_score(X[:, selected], y, train_idx, valid_idx, test_idx)
        metrics.update({'protocol': 'subject-independent', 'task_family': family, 'n_channels': int(len(selected)), 'channels': selected_names, 'n_subjects': int(len(np.unique(groups))), 'test_subjects': [int(value) for value in sorted(np.unique(groups[test_idx]))]})
        results.append(metrics)
        print(metrics)
    (output_dir / 'all_tasks_sparse_independent.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return pd.DataFrame(results)


def run_subject_wise(subject_ids, output_dir, n_channels=16):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for family, runs in TASK_FAMILIES.items():
        for subject_id in subject_ids:
            result = load_subject_family(subject_id, runs)
            if result is None:
                continue
            X, y, names = result
            indices = np.arange(len(y))
            train_valid_idx, test_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=SEED)
            train_idx, valid_idx = train_test_split(train_valid_idx, test_size=0.25, stratify=y[train_valid_idx], random_state=SEED)
            selected, selected_names = select_sparse_channels(X[train_idx], y[train_idx], names, n_channels)
            metrics = fit_and_score(X[:, selected], y, train_idx, valid_idx, test_idx, batch_size=16)
            metrics.update({'protocol': 'subject-wise', 'task_family': family, 'subject': int(subject_id), 'n_channels': int(len(selected)), 'channels': selected_names})
            results.append(metrics)
            print(metrics)
    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / 'all_tasks_sparse_subject_wise.csv', index=False)
    (output_dir / 'all_tasks_sparse_subject_wise.json').write_text(frame.to_json(orient='records', indent=2), encoding='utf-8')
    return frame
