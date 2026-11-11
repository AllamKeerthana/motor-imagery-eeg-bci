"""Unified channel-configuration experiment with PER-SUBJECT outputs (for subject-level paired statistics and equivalence tests).
Identical pipeline for every configuration: non-CAR cache -> CAR over the k used channels -> Ledoit-Wolf covariance -> per-subject label-free re-centering
(offline BATCH, one reference per subject from that subject's own same-task unlabeled trials) -> tangent space (fit on TRAIN subjects) -> shrinkage LDA.
Same GroupKFold(5) folds, same subjects. Channel choice uses TRAIN subjects only (labels of test subjects never used).
Configs: sm16 (fixed anatomical, defined before results) | tt16 (Welch t on log temporal variance of the 8-30 Hz epoch, pooled over subjects)
         | ttz16 (same score but log-variance standardised WITHIN each training subject first -> stronger data-driven baseline)
         | rand16_1..5 (random sets) | tt8, tt24, tt32 (nested prefixes of the tt ranking) | all64.
Output per (task, fold, config, test subject): n, TP, FP, FN, TN  ->  outputs_rigorous_benchmark/table_u_channels_<task>.csv  (+ timing/feature dims)
usage: python run_channel_unified.py <task>
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm
ROOT = Path(os.environ.get('EEG_PROJECT_ROOT', r'C:\SEM_7\PhaseProject')); B = ROOT / 'outputs_rigorous_benchmark'
SM = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4']

def welch_rank(lv, y):
    t, _ = stats.ttest_ind(lv[y == 0], lv[y == 1], axis=0, equal_var=False); return np.argsort(np.abs(np.nan_to_num(t)))[::-1]
def run(fam):
    z = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True); X, y, g = z['X'], z['y'], z['g']; ch = [c.strip('.').upper() for c in z['ch']]
    sm_idx = np.array(sorted(ch.index(c) for c in SM)); LV = np.log(np.var(X, axis=2) + 1e-8)
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        r_tt = welch_rank(LV[tr], y[tr])
        LVz = LV[tr].copy()
        for s in np.unique(g[tr]): m = g[tr] == s; LVz[m] = (LVz[m] - LVz[m].mean(0)) / (LVz[m].std(0) + 1e-8)
        r_ttz = welch_rank(LVz, y[tr])
        cfg = {'sm16': sm_idx, 'tt16': np.sort(r_tt[:16]), 'ttz16': np.sort(r_ttz[:16]), 'tt8': np.sort(r_tt[:8]), 'tt24': np.sort(r_tt[:24]),
               'tt32': np.sort(r_tt[:32]), 'all64': np.arange(64)}
        for k in range(1, 6): cfg[f'rand16_{k}'] = np.sort(np.random.RandomState(1000 * fold + k).choice(64, 16, replace=False))
        for name, idx in cfg.items():
            t0 = time.time(); Xs = X[:, idx]; Xs = Xs - Xs.mean(1, keepdims=True)
            C = Covariances('lwf').fit_transform(Xs)
            for s in np.unique(g):                                  # one batch reference per subject (own same-task trials)
                m = g == s; W = invsqrtm(mean_covariance(C[m], metric='riemann')); C[m] = np.array([W @ c @ W for c in C[m]])
            ts = TangentSpace(metric='riemann').fit(C[tr]); Ftr = ts.transform(C[tr]); clf = LDA(solver='lsqr', shrinkage='auto').fit(Ftr, y[tr])
            t1 = time.perf_counter(); Fte = ts.transform(C[te]); pred = clf.predict(Fte); inf_ms = (time.perf_counter() - t1) / len(te) * 1000
            for s in np.unique(g[te]):
                m = g[te] == s; yt, pp = y[te][m], pred[m]
                rows.append(dict(task_family=fam, fold=fold, config=name, subject=int(s), n=int(m.sum()), tp=int(((pp == 1) & (yt == 1)).sum()), fp=int(((pp == 1) & (yt == 0)).sum()),
                                 fn=int(((pp == 0) & (yt == 1)).sum()), tn=int(((pp == 0) & (yt == 0)).sum()), k=len(idx), n_features=len(idx) * (len(idx) + 1) // 2,
                                 tangent_lda_ms_per_trial=inf_ms, channels=' '.join(ch[i] for i in idx)))
            print(fam, fold, name, f'{time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(B / f'table_u_channels_{fam}.csv', index=False)
if __name__ == '__main__': run(sys.argv[1])
