"""EEGNet reproducibility + equal-information comparison, 16 sensorimotor channels, same GroupKFold(5) folds and the same inner-validation split as run_eegnet_sm.py.
Configs: seeds {1,2} without adaptation (seed 42 already in table_c_eegnet_sm.csv) x {compact, eegnet};
         seed 42 with per-subject Euclidean alignment (label-free, batch; same information access as batch re-centering) x {compact, eegnet}.
Saves per-test-subject confusion counts -> outputs_eegnet_groupkfold/table_c_eegnet_seeds_subjects.csv  and fold-level metrics -> table_c_eegnet_seeds.csv
"""
import time, copy, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch, torch.nn as nn
import run_eegnet_gpu as G
SM = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4']
CSV, CSVS = G.OUT / 'table_c_eegnet_seeds.csv', G.OUT / 'table_c_eegnet_seeds_subjects.csv'

def train_probs(X, y, tr, va, te, kind, seed, max_epochs=200, patience=30):      # identical recipe to G.train_eval, but returns test probabilities
    sd = X[tr].std() + 1e-6; T = lambda idx: torch.tensor(X[idx] / sd, dtype=torch.float32).unsqueeze(1).to(G.DEV)
    Xtr, Xva, Xte = T(tr), T(va), T(te); ytr = torch.tensor(y[tr]).to(G.DEV)
    torch.manual_seed(seed); np.random.seed(seed)
    model = G.make_model(kind, X.shape[1], X.shape[2]).to(G.DEV); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=8, factor=0.5); lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best, bs, wait = -1, None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(Xtr), device=G.DEV)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(G.augment(Xtr[idx])), ytr[idx]).backward(); opt.step()
        v = ((G.predict(model, Xva) > .5).astype(int) == y[va]).mean(); sch.step(v)
        if v > best: best, bs, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(bs); return G.predict(model, Xte), best

done = set(zip(*[pd.read_csv(CSV)[c] for c in ('task_family', 'fold', 'model', 'seed', 'ea')])) if CSV.exists() else set()
for fam in G.TASKS:
    z = np.load(G.CACHE / f'{fam}.npz', allow_pickle=True); X, y, g, ch = z['X'], z['y'], z['g'], [c.strip('.').upper() for c in z['ch']]
    Xs0 = G.car(X[:, np.array([ch.index(c) for c in SM]), :]); Xs_ea = G.euclid_align(Xs0, g)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=42 + fold).split(X[tr], y[tr], groups=g[tr])); tr_i, va_i = tr[a], tr[b]
        for kind in ('compact', 'eegnet'):
            for seed, ea in ((1, False), (2, False), (42, True)):
                if (fam, fold, kind, seed, ea) in done: continue
                t0 = time.time(); p, vbest = train_probs(Xs_ea if ea else Xs0, y, tr_i, va_i, te, kind, seed); pred = (p > .5).astype(int); yt = y[te]
                row = dict(task_family=fam, fold=fold, model=kind, seed=seed, ea=ea, accuracy=accuracy_score(yt, pred), balanced_accuracy=balanced_accuracy_score(yt, pred),
                           macro_f1=f1_score(yt, pred, average='macro'), precision=precision_score(yt, pred, zero_division=0), recall=recall_score(yt, pred, zero_division=0),
                           f1=f1_score(yt, pred), auc=roc_auc_score(yt, p), val_acc=vbest, seconds=round(time.time() - t0))
                pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)
                sub = []
                for s in np.unique(g[te]):
                    m = g[te] == s; yy, pp = yt[m], pred[m]
                    sub.append(dict(task_family=fam, fold=fold, model=kind, seed=seed, ea=ea, subject=int(s), n=int(m.sum()), tp=int(((pp == 1) & (yy == 1)).sum()),
                                    fp=int(((pp == 1) & (yy == 0)).sum()), fn=int(((pp == 0) & (yy == 1)).sum()), tn=int(((pp == 0) & (yy == 0)).sum())))
                pd.DataFrame(sub).to_csv(CSVS, mode='a', header=not CSVS.exists(), index=False)
                print(fam, fold, kind, seed, 'EA' if ea else '--', round(row['accuracy'], 3), row['seconds'], 's', flush=True)
