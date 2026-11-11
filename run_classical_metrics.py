"""Classical models on the SAME GroupKFold folds, with precision/recall/F1/AUC, parameter counts and latency.
Uses cache_epochs/*.npz. Channel selection on non-CAR training-fold data; CAR over used channels.
Models: riemann16 (TS-LDA), riemann16_recentered (label-free per-subject re-centering), fbcsp16.
Output: outputs_rigorous_benchmark/table_d_classical_metrics.csv
"""
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm
import run_complete_rigorous_benchmark as R

ROOT = Path(r'C:\SEM_7\PhaseProject'); OUT = ROOT / 'outputs_rigorous_benchmark' / 'table_d_classical_metrics.csv'

def metrics(clf, Ftr, ytr, Fte, yte):
    clf.fit(Ftr, ytr); p = clf.decision_function(Fte); pred = (p > 0).astype(int)
    n_par = clf.coef_.size + 1
    t0 = time.perf_counter(); clf.predict(Fte[:50]); lat = (time.perf_counter() - t0) / 50 * 1000
    return dict(accuracy=accuracy_score(yte, pred), balanced_accuracy=balanced_accuracy_score(yte, pred),
                precision=precision_score(yte, pred, zero_division=0), recall=recall_score(yte, pred, zero_division=0),
                f1=f1_score(yte, pred, zero_division=0), macro_f1=f1_score(yte, pred, average='macro'),
                auc=roc_auc_score(yte, p), lda_params=n_par, lda_latency_ms=round(lat, 3))

def recenter(C, g):
    out = np.zeros_like(C)
    for s in np.unique(g):
        m = g == s; W = invsqrtm(mean_covariance(C[m], metric='riemann')); out[m] = np.array([W @ c @ W for c in C[m]])
    return out

rows = []
for fam in R.TASK_FAMILIES:
    z = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True); X, y, g, ch = z['X'], z['y'], z['g'], list(z['ch'])
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        sel, _ = R.select_top_channels_train_only(X[tr], y[tr], ch, 16)
        Xs = R.apply_honest_car(X[:, sel, :]); Xtr, Xte = Xs[tr], Xs[te]
        cov = Covariances('lwf'); Ctr = cov.fit_transform(Xtr); Cte = cov.transform(Xte)
        ts = TangentSpace(metric='riemann'); rows.append(dict(task_family=fam, fold=fold, model='riemann16',
            n_features=16 * 17 // 2, **metrics(LDA(solver='lsqr', shrinkage='auto'), ts.fit_transform(Ctr), y[tr], ts.transform(Cte), y[te])))
        Ctr2, Cte2 = recenter(Ctr, g[tr]), recenter(Cte, g[te]); ts = TangentSpace(metric='riemann')
        rows.append(dict(task_family=fam, fold=fold, model='riemann16_recentered', n_features=136,
            **metrics(LDA(solver='lsqr', shrinkage='auto'), ts.fit_transform(Ctr2), y[tr], ts.transform(Cte2), y[te])))
        Ftr, Fte = [], []
        for lo, hi in R.FBCSP_BANDS:
            c = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
            Ftr.append(c.fit_transform(R.bandpass_fast(Xtr, lo, hi), y[tr])); Fte.append(c.transform(R.bandpass_fast(Xte, lo, hi)))
        rows.append(dict(task_family=fam, fold=fold, model='fbcsp16', n_features=16,
            **metrics(LDA(solver='lsqr', shrinkage='auto'), np.hstack(Ftr), y[tr], np.hstack(Fte), y[te])))
        print(fam, fold, [round(r['accuracy'], 3) for r in rows[-3:]], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
