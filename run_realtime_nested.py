"""Online-adaptation configuration chosen by NESTED subject-wise CV (no test-fold tuning), causal pipeline, 16 sensorimotor channels.
Per outer fold (same GroupKFold(5) folds): inner GroupKFold(3) over the outer-TRAIN subjects scores every (w0, scale_prior) config of the
ONLINE re-centering (LDA trained on inner-train subjects, batch-whitened; inner-validation subjects processed online in arrival order);
best inner mean accuracy is chosen -> LDA refit on all outer-train subjects -> outer test scored ONCE with the chosen config.
Grid (fixed in advance): w0 in {0.5, 1, 2, 5, 10} x scale_prior in {False, True}.
The pre-registered w0=5 / no-scaling result stays in table_j_streaming.csv (condition C_causal_online_w0=5).
Exports the fold-1 chosen-config model + held-out replay windows to deploy_rt/<task>/ for the Pi.
Output: outputs_rigorous_benchmark/table_k_streaming_nested.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import rt_pipeline as RT

ROOT = Path(r'C:\SEM_7\PhaseProject')
B = ROOT / 'outputs_rigorous_benchmark'
TASKS = ['execution_unilateral', 'imagery_unilateral', 'execution_bilateral', 'imagery_bilateral']
SM = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4']
GRID = [(w0, sp) for w0 in (0.5, 1, 2, 5, 10) for sp in (False, True)]
NF = RT.K * (RT.K + 1) // 2


def batch_feats(C, g):
    out = np.empty((len(C), NF))
    for s in np.unique(g):
        m = g == s
        out[m] = RT.tangent(RT.whiten(C[m], C[m].mean(0)))
    return out


def online_feats(C, g, P, w0, sp):
    out = np.empty((len(C), NF))
    for s in np.unique(g):
        m = np.where(g == s)[0]
        out[m] = RT.tangent(RT.whiten(C[m], RT.online_means(C[m], P, w0, sp)))
    return out


rows = []
for fam in TASKS:
    raw = np.load(ROOT / 'cache_raw16' / f'{fam}.npz', allow_pickle=True)
    Xr, y, g = raw['X'].astype(np.float64), raw['y'], raw['g']
    Cc = RT.cov(RT.window(Xr))
    for fold, (tr, te) in enumerate(GroupKFold(5).split(Xr, y, groups=g), 1):
        subj = np.unique(g[tr])
        inner = {c: [] for c in GRID}
        for itr, iva in GroupKFold(3).split(subj, groups=subj):
            a = tr[np.isin(g[tr], subj[itr])]
            b = tr[np.isin(g[tr], subj[iva])]
            lda = LDA(solver='lsqr', shrinkage='auto').fit(batch_feats(Cc[a], g[a]), y[a])
            P = Cc[a].mean(0)
            for c in GRID:
                sc = lda.decision_function(online_feats(Cc[b], g[b], P, *c))
                inner[c].append(((sc > 0).astype(int) == y[b]).mean())
        best = max(GRID, key=lambda c: np.mean(inner[c]))
        lda = LDA(solver='lsqr', shrinkage='auto').fit(batch_feats(Cc[tr], g[tr]), y[tr])
        P = Cc[tr].mean(0)
        sc = lda.decision_function(online_feats(Cc[te], g[te], P, *best))
        p = (sc > 0).astype(int)
        yt = y[te]
        rows.append(dict(task_family=fam, fold=fold, w0=best[0], scale_prior=best[1], inner_acc=float(np.mean(inner[best])), accuracy=(p == yt).mean(),
                         precision=precision_score(yt, p, zero_division=0), recall=recall_score(yt, p, zero_division=0), f1=f1_score(yt, p), auc=roc_auc_score(yt, sc)))
        print(fam, fold, 'chosen', best, round(rows[-1]['accuracy'], 3), flush=True)
        if fold == 1:
            out = ROOT / 'deploy_rt' / fam
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / 'model.npz', coef=lda.coef_[0], intercept=lda.intercept_[0], P=P, w0=float(best[0]), scale_prior=int(best[1]), channels=np.array(SM))
            np.savez(out / 'replay.npz', X=Xr[te].astype(np.float32), y=y[te], g=g[te])
        pd.DataFrame(rows).to_csv(B / 'table_k_streaming_nested.csv', index=False)
