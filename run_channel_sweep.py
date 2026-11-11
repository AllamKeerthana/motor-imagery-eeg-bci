"""Minimum-channel sweep, all 4 tasks, same GroupKFold folds. Riemannian TS-LDA with label-free per-subject re-centering.
Channels ranked by Welch-t on log-variance of NON-CAR TRAINING-fold data; nested top-k; CAR over the k used channels.
Output: outputs_rigorous_benchmark/table_e_channel_sweep.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm
ROOT = Path(r'C:\SEM_7\PhaseProject'); OUT = ROOT / 'outputs_rigorous_benchmark' / 'table_e_channel_sweep.csv'
TASKS = ['execution_unilateral', 'imagery_unilateral', 'execution_bilateral', 'imagery_bilateral']
KS = [4, 8, 12, 16, 24, 32, 64]

def rank_channels(X, y):
    lv = np.log(np.var(X, axis=2) + 1e-8); t, _ = stats.ttest_ind(lv[y == 0], lv[y == 1], axis=0, equal_var=False)
    return np.argsort(np.abs(np.nan_to_num(t)))[::-1]

def recenter(C, g):
    out = np.zeros_like(C)
    for s in np.unique(g):
        m = g == s; W = invsqrtm(mean_covariance(C[m], metric='riemann')); out[m] = np.array([W @ c @ W for c in C[m]])
    return out

rows = []
for fam in TASKS:
    z = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True); X, y, g = z['X'], z['y'], z['g']
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        order = rank_channels(X[tr], y[tr])
        for k in KS:
            sel = np.sort(order[:k]); Xs = X[:, sel, :]; Xs = Xs - Xs.mean(1, keepdims=True)
            cov = Covariances('lwf'); Ctr = recenter(cov.fit_transform(Xs[tr]), g[tr]); Cte = recenter(cov.transform(Xs[te]), g[te])
            ts = TangentSpace(metric='riemann'); clf = LDA(solver='lsqr', shrinkage='auto')
            clf.fit(ts.fit_transform(Ctr), y[tr]); p = clf.decision_function(ts.transform(Cte)); pr = (p > 0).astype(int); yt = y[te]
            rows.append(dict(task_family=fold and fam, fold=fold, k=k, accuracy=accuracy_score(yt, pr), balanced_accuracy=balanced_accuracy_score(yt, pr),
                precision=precision_score(yt, pr, zero_division=0), recall=recall_score(yt, pr, zero_division=0), f1=f1_score(yt, pr, zero_division=0),
                auc=roc_auc_score(yt, p), n_features=k * (k + 1) // 2))
            print(fam, fold, k, round(rows[-1]['accuracy'], 3), flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
