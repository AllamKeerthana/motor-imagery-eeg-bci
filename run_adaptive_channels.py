"""Adaptive (per-subject) channel selection, PRE-REGISTERED design (no selection among arms on test data).
For every outer GroupKFold(5) fold (same folds as all other tables) and every test subject s:
  calibration pool = 10 trials/class (RandomState(42+s), same as run_fewshot.py); TEST = all other trials of s (identical for every arm).
  Channel set per arm (16 channels):
    sm        : fixed anatomical sensorimotor set (lambda=0)
    lam0.25/0.5/1.0 : top-16 of (1-lam)*z(prior) + lam*z(subject evidence); prior = 1 on sm channels else 0;
                      evidence = |Welch t| of log-variance computed ONLY from s's 10+10 calibration trials (non-CAR)
    rand      : random 16 channels (control, seeded by subject)
  Classifier (same for all arms): trials -> CAR over the 16 channels -> sample covariance (+5% shrink to scaled identity) ->
    per-subject label-free Riemannian re-centering (training subjects each re-centered on own trials) -> tangent space (fit on TRAIN subjects only)
    -> shrinkage LDA trained on TRAIN-subject features + the 20 calibration trials replicated x10.
  Target subject's tangent features use re-centering on all of s's UNLABELED trials (label-free).
  Subjects with <12 trials in a class are excluded from ALL arms.
Primary comparison (declared in advance): lam0.5 vs sm, paired Wilcoxon over subjects, Holm over the 4 tasks.
Output: outputs_rigorous_benchmark/table_i_adaptive_channels.csv
"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm
ROOT = Path(r'C:\SEM_7\PhaseProject'); OUT = ROOT / 'outputs_rigorous_benchmark' / 'table_i_adaptive_channels.csv'
TASKS = ['execution_unilateral', 'imagery_unilateral', 'execution_bilateral', 'imagery_bilateral']
SM = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4']
LAMS = [0.25, 0.5, 1.0]; REP = 10; K = 16; ALPHA = 0.05

def z(v): return (v - v.mean()) / (v.std() + 1e-12)
def set_cov(Cfull, idx):
    S = Cfull[:, idx][:, :, idx]; P = np.eye(K) - 1.0 / K
    S = np.einsum('ij,njk,lk->nil', P, S, P)
    tr = np.trace(S, axis1=1, axis2=2)[:, None, None] / K
    return (1 - ALPHA) * S + ALPHA * tr * np.eye(K)
def recenter(C, g):
    out = np.empty_like(C)
    for s in np.unique(g):
        m = g == s; W = invsqrtm(mean_covariance(C[m], metric='riemann')); out[m] = W @ C[m] @ W
    return out
def evidence(Xc, yc):
    lv = np.log(np.var(Xc, axis=2) + 1e-8); t, _ = stats.ttest_ind(lv[yc == 0], lv[yc == 1], axis=0, equal_var=False)
    return np.abs(np.nan_to_num(t))

def main(tasks, max_folds=5):
    global OUT
    OUT = ROOT / 'outputs_rigorous_benchmark' / f'table_i_adaptive_channels_{tasks[0]}.csv'
    done = set()
    if OUT.exists(): d = pd.read_csv(OUT); done = set(zip(d.task_family, d.fold))
    rows = pd.read_csv(OUT).to_dict('records') if OUT.exists() else []
    for fam in tasks:
        zf = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True); X, y, g, ch = zf['X'], zf['y'], zf['g'], [c.strip('.').upper() for c in zf['ch']]
        sm_idx = np.sort([ch.index(c) for c in SM]); prior = np.zeros(64); prior[sm_idx] = 1
        Cfull = np.einsum('nct,ndt->ncd', X.astype(np.float64), X.astype(np.float64)) / X.shape[2]
        for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
            if fold > max_folds or (fam, fold) in done: continue
            t0 = time.time(); cache = {}
            def train_side(idx):
                key = tuple(idx)
                if key not in cache:
                    C = recenter(set_cov(Cfull[tr], np.array(idx)), g[tr]); ts = TangentSpace(metric='riemann').fit(C)
                    cache[key] = (ts, ts.transform(C))
                return cache[key]
            for s in np.unique(g[te]):
                ii = np.where(g == s)[0]; ys = y[ii]
                if min((ys == 0).sum(), (ys == 1).sum()) < 12: continue
                rng = np.random.RandomState(42 + int(s)); pool = []
                for c in (0, 1):
                    ci = np.where(ys == c)[0]; rng.shuffle(ci); pool.append(ci[:10])
                cal = np.concatenate(pool); test_i = np.setdiff1d(np.arange(len(ii)), cal)
                ev = z(evidence(X[ii][cal][:, :, :], ys[cal]))
                arms = {'sm': sm_idx}
                for lam in LAMS: arms[f'lam{lam}'] = np.sort(np.argsort(-((1 - lam) * z(prior) + lam * ev), kind='stable')[:K])
                arms['rand'] = np.sort(np.random.RandomState(1000 + int(s)).choice(64, K, replace=False))
                for name, idx in arms.items():
                    ts, Ftr = train_side(list(idx))
                    Fs = ts.transform(recenter(set_cov(Cfull[ii], idx), np.zeros(len(ii), int)))
                    Fx = np.vstack([Ftr] + [Fs[cal]] * REP); yx = np.concatenate([y[tr]] + [ys[cal]] * REP)
                    acc = LDA(solver='lsqr', shrinkage='auto').fit(Fx, yx).score(Fs[test_i], ys[test_i])
                    rows.append(dict(task_family=fam, fold=fold, subject=int(s), arm=name, accuracy=acc, n_test=len(test_i),
                                     overlap_with_sm=len(set(idx) & set(sm_idx))))
            print(fam, fold, f'{time.time() - t0:.0f}s', flush=True); pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == '__main__':
    main(sys.argv[1:] or TASKS)
