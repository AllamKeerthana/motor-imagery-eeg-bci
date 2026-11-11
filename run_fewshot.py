"""Few-shot calibration curve (fixed subject set, fixed target test trials across all calibration sizes).
Source model: Riemannian TS-LDA, 16 channels chosen on training-fold subjects (non-CAR), re-centered per subject.
For each test-fold subject: calibration pool = 10 trials/class (seeded); test = all remaining trials (identical for every n).
n in {0,2,5,10}/class: LDA retrained on training-subject features + the first n/class target trials (target trials up-weighted x10 by replication).
Subjects with <12 trials in either class are dropped for ALL n. Output: table_f_fewshot.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm
ROOT = Path(r'C:\SEM_7\PhaseProject'); OUT = ROOT / 'outputs_rigorous_benchmark' / 'table_f_fewshot.csv'
TASKS = ['execution_unilateral', 'imagery_unilateral', 'execution_bilateral', 'imagery_bilateral']; NS = [0, 2, 5, 10]; REP = 10

def rank(X, y):
    lv = np.log(np.var(X, axis=2) + 1e-8); t, _ = stats.ttest_ind(lv[y == 0], lv[y == 1], axis=0, equal_var=False)
    return np.argsort(np.abs(np.nan_to_num(t)))[::-1]
def recenter_one(C):
    W = invsqrtm(mean_covariance(C, metric='riemann')); return np.array([W @ c @ W for c in C])

rows = []
for fam in TASKS:
    z = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True); X, y, g = z['X'], z['y'], z['g']
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        sel = np.sort(rank(X[tr], y[tr])[:16]); Xs = X[:, sel, :]; Xs = Xs - Xs.mean(1, keepdims=True)
        cov = Covariances('lwf'); C = cov.fit_transform(Xs)
        Ctr = np.zeros_like(C[tr])
        for s in np.unique(g[tr]): m = g[tr] == s; Ctr[m] = recenter_one(C[tr][m])
        ts = TangentSpace(metric='riemann'); Ftr = ts.fit_transform(Ctr)
        for s in np.unique(g[te]):
            idx = np.where(g == s)[0]; ys = y[idx]
            if min((ys == 0).sum(), (ys == 1).sum()) < 12: continue
            rng = np.random.RandomState(42 + int(s)); pool = []
            for c in (0, 1):
                ci = np.where(ys == c)[0]; rng.shuffle(ci); pool.append(ci[:10])
            test_i = np.setdiff1d(np.arange(len(idx)), np.concatenate(pool))
            Fs = ts.transform(recenter_one(C[idx]))   # label-free re-centering on all of the subject's unlabeled trials
            for n in NS:
                if n == 0: Fx, yx = Ftr, y[tr]
                else:
                    ci = np.concatenate([pool[0][:n], pool[1][:n]])
                    Fx = np.vstack([Ftr] + [Fs[ci]] * REP); yx = np.concatenate([y[tr]] + [ys[ci]] * REP)
                clf = LDA(solver='lsqr', shrinkage='auto').fit(Fx, yx)
                rows.append(dict(task_family=fam, fold=fold, subject=int(s), n_per_class=n, accuracy=clf.score(Fs[test_i], ys[test_i]), n_test=len(test_i)))
        print(fam, fold, flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)
