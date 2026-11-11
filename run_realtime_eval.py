"""Three-condition streaming evaluation (subject-independent, same GroupKFold(5) folds), 16 sensorimotor channels, LDA on tangent features.
 A  zero-phase 8-30 Hz (offline cache) + BATCH re-centering (subject's whole session)      -> offline reference
 B  CAUSAL 8-30 Hz                    + BATCH re-centering                                 -> cost of causal filtering
 C  CAUSAL 8-30 Hz                    + ONLINE re-centering (running mean, prior weight w0) -> the deployable pipeline; same trained LDA as B
Training subjects are always whitened with their own batch mean (label-free). w0=5 is fixed in advance; w0 in {1,20} are sensitivity checks.
Also: cold-start accuracy vs number of unlabeled windows seen, a no-adaptation control, and a streaming-class vs vectorised equivalence check.
Exports the fold-1 condition-C model + held-out replay windows to deploy_rt/<task>/ (for the Pi).
Outputs: outputs_rigorous_benchmark/table_j_streaming.csv, table_j_coldstart.csv
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
NF = RT.K * (RT.K + 1) // 2


def batch_feats(C, g):
    out = np.empty((len(C), NF))
    for s in np.unique(g):
        m = g == s
        out[m] = RT.tangent(RT.whiten(C[m], C[m].mean(0)))
    return out


def online_feats(C, g, P, w0):
    out = np.empty((len(C), NF))
    for s in np.unique(g):
        m = np.where(g == s)[0]
        out[m] = RT.tangent(RT.whiten(C[m], RT.online_means(C[m], P, w0)))
    return out


def metr(y, s):
    p = (s > 0).astype(int)
    return dict(accuracy=(p == y).mean(), precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p, zero_division=0),
                f1=f1_score(y, p), auc=roc_auc_score(y, s))


def norm_tangent(C):
    return RT.tangent(C / np.trace(C, axis1=1, axis2=2)[:, None, None] * RT.K)


rows, cold = [], []
for fam in TASKS:
    raw = np.load(ROOT / 'cache_raw16' / f'{fam}.npz', allow_pickle=True)
    Xr, y, g = raw['X'].astype(np.float64), raw['y'], raw['g']
    ref = np.load(ROOT / 'cache_epochs' / f'{fam}.npz', allow_pickle=True)
    chn = [c.strip('.').upper() for c in ref['ch']]
    Xz = ref['X'][:, [chn.index(c) for c in SM]].astype(np.float64)
    Xz = Xz - Xz.mean(1, keepdims=True)
    Cz = RT.cov(Xz)
    Cc = RT.cov(RT.window(Xr))
    for fold, (tr, te) in enumerate(GroupKFold(5).split(Xr, y, groups=g), 1):
        lda = LDA(solver='lsqr', shrinkage='auto').fit(batch_feats(Cz[tr], g[tr]), y[tr])
        rows.append(dict(task_family=fam, fold=fold, condition='A_zerophase_batch', **metr(y[te], lda.decision_function(batch_feats(Cz[te], g[te])))))
        lda = LDA(solver='lsqr', shrinkage='auto').fit(batch_feats(Cc[tr], g[tr]), y[tr])
        rows.append(dict(task_family=fam, fold=fold, condition='B_causal_batch', **metr(y[te], lda.decision_function(batch_feats(Cc[te], g[te])))))
        P = Cc[tr].mean(0)
        gs = g[te]
        for w0 in (1, 5, 20):
            sc = lda.decision_function(online_feats(Cc[te], gs, P, w0))
            rows.append(dict(task_family=fam, fold=fold, condition=f'C_causal_online_w0={w0}', **metr(y[te], sc)))
            if w0 != 5:
                continue
            pos = np.concatenate([np.arange(1, (gs == s).sum() + 1) for s in pd.unique(gs)])
            order = np.concatenate([np.where(gs == s)[0] for s in pd.unique(gs)])
            ok = ((sc > 0).astype(int) == y[te])[order]
            for lo, hi, nm in ((1, 5, '1-5'), (6, 10, '6-10'), (11, 20, '11-20'), (21, 999, '21+')):
                mk = (pos >= lo) & (pos <= hi)
                cold.append(dict(task_family=fam, fold=fold, windows_seen=nm, n=int(mk.sum()), acc=ok[mk].mean()))
            if fold == 1:
                model = dict(coef=lda.coef_[0], intercept=lda.intercept_[0], P=P, w0=5.0)
                pipe = RT.Online(model)
                first = pd.unique(gs)[0]
                idx = np.where(gs == first)[0][:25]
                pipe.reset()
                s_stream = np.array([pipe.step(Xr[te][i].astype(np.float32)) for i in idx])
                print(fam, 'streaming-class vs vectorised, max abs diff:', float(np.abs(s_stream - sc[idx]).max()), flush=True)
                out = ROOT / 'deploy_rt' / fam
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out / 'model.npz', coef=lda.coef_[0], intercept=lda.intercept_[0], P=P, w0=5.0, channels=np.array(SM))
                np.savez(out / 'replay.npz', X=Xr[te].astype(np.float32), y=y[te], g=g[te])
        l0 = LDA(solver='lsqr', shrinkage='auto').fit(norm_tangent(Cc[tr]), y[tr])
        rows.append(dict(task_family=fam, fold=fold, condition='N_causal_no_adaptation', **metr(y[te], l0.decision_function(norm_tangent(Cc[te])))))
        print(fam, fold, [round(r['accuracy'], 3) for r in rows[-6:]], flush=True)
        pd.DataFrame(rows).to_csv(B / 'table_j_streaming.csv', index=False)
        pd.DataFrame(cold).to_csv(B / 'table_j_coldstart.csv', index=False)
