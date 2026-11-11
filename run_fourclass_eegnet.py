"""4-class (left fist / right fist / both fists / both feet) subject-independent 5-fold GroupKFold EEGNet.
Pools the unilateral + bilateral caches of one condition (execution or imagery); y = y_binary + 2*is_bilateral.
Same protocol as run_eegnet_gpu.py: CAR, train-fold-only channel selection, validation-subject early stopping.
Resumable CSV: outputs_fourclass/table_4class_eegnet.csv
usage: python run_fourclass_eegnet.py [execution imagery]"""
import sys, time, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import run_eegnet_gpu as G

ROOT = Path(r'C:\SEM_7\PhaseProject'); OUT = ROOT / 'outputs_fourclass'; OUT.mkdir(exist_ok=True)
CSV = OUT / 'table_4class_eegnet.csv'; CM = OUT / 'confusion_4class.csv'
CLASSES = ['left_fist', 'right_fist', 'both_fists', 'both_feet']
DEV, SEED = G.DEV, G.SEED
CONFIGS = [(64, 'eegnet'), (16, 'eegnet'), (16, 'compact')]

def load(cond):
    a = np.load(G.CACHE / f'{cond}_unilateral.npz', allow_pickle=True); b = np.load(G.CACHE / f'{cond}_bilateral.npz', allow_pickle=True)
    assert list(a['ch']) == list(b['ch'])
    return (np.concatenate([a['X'], b['X']]), np.concatenate([a['y'], b['y'] + 2]), np.concatenate([a['g'], b['g']]), list(a['ch']))

def select(X, y, k):
    F, _ = f_classif(np.log(np.var(X, axis=2) + 1e-8), y)
    return np.sort(np.argsort(np.nan_to_num(F))[::-1][:k])

def make(kind, ch, T):
    m = G.make_model(kind, ch, T); m.c = nn.Linear(m.c.in_features, 4); return m

def probs(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256): out.append(model(X[i:i + 256]).argmax(1).cpu())
    return torch.cat(out).numpy()

def train_eval(X, y, tr, va, te, kind):
    sd = X[tr].std() + 1e-6; T = lambda i: torch.tensor(X[i] / sd, dtype=torch.float32).unsqueeze(1).to(DEV)
    Xtr, Xva, Xte = T(tr), T(va), T(te); ytr = torch.tensor(y[tr]).to(DEV)
    torch.manual_seed(SEED); np.random.seed(SEED); model = make(kind, X.shape[1], X.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=8, factor=0.5)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best, state, wait = -1, None, 0
    for ep in range(200):
        model.train(); perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(G.augment(Xtr[idx])), ytr[idx]).backward(); opt.step()
        v = (probs(model, Xva) == y[va]).mean(); sch.step(v)
        if v > best: best, state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 30: break
    model.load_state_dict(state); pred = probs(model, Xte); yt = y[te]
    return (dict(accuracy=accuracy_score(yt, pred), balanced_accuracy=balanced_accuracy_score(yt, pred),
                 macro_f1=f1_score(yt, pred, average='macro'), val_acc=best, epochs=ep + 1,
                 params=sum(p.numel() for p in model.parameters())), confusion_matrix(yt, pred, labels=range(4)))

def main(conds):
    done = set()
    if CSV.exists(): d = pd.read_csv(CSV); done = set(zip(d.condition, d.fold, d.n_channels, d.model))
    for cond in conds:
        X, y, g, ch = load(cond); print(cond, X.shape, np.bincount(y), flush=True)
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g), 1):
            a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED + fold).split(X[tr], y[tr], groups=g[tr]))
            tr_i, va_i = tr[a], tr[b]
            for nch, kind in CONFIGS:
                if (cond, fold, nch, kind) in done: continue
                t0 = time.time(); sel = np.arange(64) if nch == 64 else select(X[tr_i], y[tr_i], nch)
                m, cm = train_eval(G.car(X[:, sel, :]), y, tr_i, va_i, te, kind)
                row = dict(condition=cond, fold=fold, n_channels=nch, model=kind, **m, n_test=len(te), chance=0.25, seconds=round(time.time() - t0))
                pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)
                pd.DataFrame([dict(condition=cond, fold=fold, n_channels=nch, model=kind, **{f'{CLASSES[i]}->{CLASSES[j]}': int(cm[i, j]) for i in range(4) for j in range(4)})]).to_csv(CM, mode='a', header=not CM.exists(), index=False)
                print(cond, fold, nch, kind, f"acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f} f1={m['macro_f1']:.3f} {row['seconds']}s", flush=True)

if __name__ == '__main__':
    main(sys.argv[1:] or ['execution', 'imagery'])
