"""
Per-subject adaptive channel selection experiment -- Riemannian version.

Extends personalized_channel_experiment.py by adding a Riemannian
tangent-space classifier (Covariance -> TangentSpace -> shrinkage LDA)
alongside your existing FBCSP-LDA, so you get a direct, same-run
comparison of:

    channel set (fixed / personalized)  x  classifier (FBCSP-LDA / Riemannian TS-LDA)
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
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

RANDOM_SEED = 42
N_CALIB_PER_CLASS = 10
K_CHANNELS = 16
FBCSP_BANDS = [(8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0)]
RIEMANN_BAND = (8.0, 30.0)

DATA_DIR = Path(r'C:\SEM_7\PhaseProject\eeg-motor-movementimagery-dataset-1.0.0\files')
OUTPUT_DIR = Path(r'C:\SEM_7\PhaseProject\outputs_personalized_riemann')
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
    """Loads EDF files for a given subject and task family."""
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
    sos = butter(4, [low, high], btype='bandpass', fs=sfreq, output='sos')
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def score_channels(X_calib, y_calib, sfreq=160.0, freq_band=(8.0, 30.0)):
    filtered = bandpass_fast(X_calib, freq_band[0], freq_band[1], sfreq=sfreq)
    band_power = np.mean(filtered ** 2, axis=2)
    classes = np.unique(y_calib)
    assert len(classes) == 2, "Binary T1/T2 labels required."
    a = band_power[y_calib == classes[0]]
    b = band_power[y_calib == classes[1]]
    t_stat, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.abs(np.nan_to_num(t_stat))


def select_top_k(scores, ch_names, k):
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    return sorted(top_idx.tolist()), [ch_names[i] for i in top_idx]


def eval_fbcsp_lda(X_calib, y_calib, X_test, y_test, sfreq, ch_idx):
    Xc = X_calib[:, ch_idx, :]
    Xt = X_test[:, ch_idx, :]
    train_feats, test_feats = [], []
    for lo, hi in FBCSP_BANDS:
        Xcb = bandpass_fast(Xc, lo, hi, sfreq=sfreq)
        Xtb = bandpass_fast(Xt, lo, hi, sfreq=sfreq)
        csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
        train_feats.append(csp.fit_transform(Xcb, y_calib))
        test_feats.append(csp.transform(Xtb))
    train_X = np.concatenate(train_feats, axis=1)
    test_X = np.concatenate(test_feats, axis=1)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(train_X, y_calib)
    return float(clf.score(test_X, y_test))


def eval_riemann_tslda(X_calib, y_calib, X_test, y_test, sfreq, ch_idx):
    """
    Riemannian Tangent-Space LDA pipeline with Ledoit-Wolf shrinkage covariance.
    """
    Xc = X_calib[:, ch_idx, :]
    Xt = X_test[:, ch_idx, :]
    lo, hi = RIEMANN_BAND
    Xcb = bandpass_fast(Xc, lo, hi, sfreq=sfreq)
    Xtb = bandpass_fast(Xt, lo, hi, sfreq=sfreq)

    # Ledoit-Wolf shrinkage covariance estimation
    cov = Covariances(estimator="lwf")
    Cc = cov.fit_transform(Xcb)
    Ct = cov.transform(Xtb)

    # Project to Tangent Space (Reference estimated from calibration data only)
    ts = TangentSpace(metric="riemann")
    Fc = ts.fit_transform(Cc)
    Ft = ts.transform(Ct)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(Fc, y_calib)
    return float(clf.score(Ft, y_test))


def run_experiment(subject_ids, task_group, fixed_channel_names, k=K_CHANNELS, n_calib_per_class=N_CALIB_PER_CLASS):
    results = []
    print(f"\n=========================================================================", flush=True)
    print(f"RUNNING RIEMANNIAN VS FBCSP EXPERIMENT: {task_group.upper()}", flush=True)
    print(f"Configuration: N_calib_per_class={n_calib_per_class}, K={k}", flush=True)
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

        scores = score_channels(X_calib, y_calib, sfreq)
        pers_idx, pers_names = select_top_k(scores, ch_names, k)
        fixed_idx = [ch_names.index(c) for c in fixed_channel_names if c in ch_names]

        row = {
            "subject": subj,
            "task_group": task_group,
            "n_calib_per_class": n_calib_per_class,
            "n_test_trials": len(test_idx),
            "personalized_channels": pers_names,
        }

        try:
            row["acc_fixed_fbcsp"] = eval_fbcsp_lda(X_calib, y_calib, X_test, y_test, sfreq, fixed_idx)
            row["acc_pers_fbcsp"] = eval_fbcsp_lda(X_calib, y_calib, X_test, y_test, sfreq, pers_idx)
            row["acc_fixed_riemann"] = eval_riemann_tslda(X_calib, y_calib, X_test, y_test, sfreq, fixed_idx)
            row["acc_pers_riemann"] = eval_riemann_tslda(X_calib, y_calib, X_test, y_test, sfreq, pers_idx)
        except Exception as e:
            print(f"[WARN] subj={subj} failed: {e}", flush=True)
            continue

        results.append(row)
        print(f"[{task_group}] S{subj:03d} | "
              f"FBCSP: Fixed={row['acc_fixed_fbcsp']*100:.1f}% Pers={row['acc_pers_fbcsp']*100:.1f}% | "
              f"Riemann: Fixed={row['acc_fixed_riemann']*100:.1f}% Pers={row['acc_pers_riemann']*100:.1f}%", flush=True)

    return results


def summarize(results):
    if not results:
        return {}
    def col(name):
        return np.array([r[name] for r in results])

    fbcsp_fixed, fbcsp_pers = col("acc_fixed_fbcsp"), col("acc_pers_fbcsp")
    riem_fixed, riem_pers = col("acc_fixed_riemann"), col("acc_pers_riemann")

    def report(name, fixed, pers):
        delta = pers - fixed
        n_total = len(delta)
        n_improved = int(np.sum(delta > 0))
        try:
            _, p = stats.wilcoxon(pers, fixed)
        except Exception:
            p = None
        print(f"\n--- {name} ---", flush=True)
        print(f"Mean fixed:        {fixed.mean()*100:.2f}%", flush=True)
        print(f"Mean personalized: {pers.mean()*100:.2f}%", flush=True)
        print(f"Mean delta:        {delta.mean()*100:+.2f}%", flush=True)
        print(f"Improved:          {n_improved}/{n_total} ({n_improved/n_total*100:.1f}%)", flush=True)
        print(f"Wilcoxon p-value:  {p:.5f}" if p is not None else "Wilcoxon: skipped", flush=True)
        return {
            "mean_fixed": float(fixed.mean()), "mean_personalized": float(pers.mean()),
            "mean_delta": float(delta.mean()), "n_improved": n_improved,
            "n_total": n_total, "wilcoxon_p": float(p) if p is not None else None,
        }

    print(f"\n=========================================================================", flush=True)
    print(f"SUMMARY FOR TASK GROUP: {results[0]['task_group'].upper()} (N={len(results)})", flush=True)
    print(f"=========================================================================", flush=True)
    out = {
        "task_group": results[0]["task_group"],
        "fbcsp": report("FBCSP-LDA", fbcsp_fixed, fbcsp_pers),
        "riemann": report("Riemannian TS-LDA", riem_fixed, riem_pers),
    }

    # Direct comparison: Riemannian vs FBCSP on personalized channels
    print("\n--- Classifier Comparison (Personalized Channels) ---", flush=True)
    print(f"FBCSP-LDA mean    : {fbcsp_pers.mean()*100:.2f}%", flush=True)
    print(f"Riemannian mean   : {riem_pers.mean()*100:.2f}%", flush=True)
    print(f"Riemannian Delta  : {(riem_pers.mean() - fbcsp_pers.mean())*100:+.2f}%", flush=True)
    try:
        _, p_clf = stats.wilcoxon(riem_pers, fbcsp_pers)
        print(f"Wilcoxon p-value  : {p_clf:.5f} ({'Statistically Significant' if p_clf < 0.05 else 'Not Significant'})", flush=True)
        out["clf_comparison_delta"] = float(riem_pers.mean() - fbcsp_pers.mean())
        out["clf_comparison_p"] = float(p_clf)
    except Exception:
        out["clf_comparison_p"] = None

    return out


def run_full_study():
    seed_everything(RANDOM_SEED)
    subject_ids = list(range(1, 110))
    
    all_experiment_records = []
    all_summaries = []

    for family in TASK_FAMILIES.keys():
        fixed_ch = FIXED_CHANNELS[family]
        records = run_experiment(subject_ids, family, fixed_ch, k=16, n_calib_per_class=10)
        summary = summarize(records)
        all_experiment_records.extend(records)
        all_summaries.append(summary)

    # Save outputs
    pd.DataFrame(all_experiment_records).to_csv(OUTPUT_DIR / "personalized_riemann_all_subjects.csv", index=False)
    (OUTPUT_DIR / "personalized_riemann_all_subjects.json").write_text(json.dumps(all_experiment_records, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "personalized_riemann_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("\n[COMPLETE] All Riemannian personalized channel experiments finished!", flush=True)
    print(f"Outputs saved under: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    run_full_study()
