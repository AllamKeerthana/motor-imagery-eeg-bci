"""Subject-independent 5-fold GroupKFold EEGNet on GPU (same folds as run_complete_rigorous_benchmark.py).
Configs: channels {64,16} x Euclidean alignment {off,on} x model {eegnet, compact}.
Leakage controls: channel selection (non-CAR, training-fold subjects only); CAR over used channels;
early stopping on held-out VALIDATION SUBJECTS from the training fold; EA is label-free per-subject.
Metrics: acc, balanced acc, precision, recall, F1 (class 1 and macro), AUC, params, model size, CPU latency.
Resumable CSV: outputs_eegnet_groupkfold/table_c_eegnet_gpu.csv
usage: python run_eegnet_gpu.py [task ...]
"""
import sys, time, copy, io
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats
from scipy.linalg import fractional_matrix_power
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

ROOT = Path(r'C:\SEM_7\PhaseProject'); CACHE = ROOT / 'cache_epochs'; OUT = ROOT / 'outputs_eegnet_groupkfold'
OUT.mkdir(exist_ok=True); CSV = OUT / 'table_c_eegnet_gpu.csv'
TASKS = ['execution_unilateral', 'imagery_unilateral', 'execution_bilateral', 'imagery_bilateral']
SEED = 42; DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIGS = [(64, False, 'eegnet'), (16, False, 'eegnet'), (16, True, 'eegnet'), (64, True, 'eegnet'),
           (16, True, 'compact'), (16, False, 'compact')]

class EEGNet(nn.Module):
    def __init__(s, ch, T, F1=8, D=2, F2=16, drop=0.25):
        super().__init__()
        s.f = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False), nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad(): n = s.f(torch.zeros(1, 1, ch, T)).flatten(1).shape[1]
        s.c = nn.Linear(n, 2)
    def forward(s, x): return s.c(s.f(x).flatten(1))

def make_model(kind, ch, T):
    return EEGNet(ch, T) if kind == 'eegnet' else EEGNet(ch, T, F1=4, D=2, F2=8, drop=0.25)

def select16(X, y, k=16):
    lv = np.log(np.var(X, axis=2) + 1e-8); a, b = lv[y == 0], lv[y == 1]
    t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
    return np.sort(np.argsort(np.abs(np.nan_to_num(t)))[::-1][:k])

def car(X): return X - X.mean(axis=1, keepdims=True)

def euclid_align(X, g):
    X = X.copy()
    for s in np.unique(g):
        m = g == s; Xs = X[m].astype(np.float64)
        R = np.mean([x @ x.T / x.shape[1] for x in Xs], axis=0) + 1e-6 * np.eye(Xs.shape[1])
        w, v = np.linalg.eigh(R); Ri = (v / np.sqrt(np.maximum(w, 1e-12))) @ v.T
        X[m] = np.einsum('ij,tjk->tik', Ri, Xs).astype(np.float32)
    return X

def augment(x):
    x = x + 0.05 * torch.randn_like(x)
    return torch.roll(x, int(np.random.randint(-8, 9)), dims=-1)

def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256): out.append(torch.softmax(model(X[i:i + 256]), 1)[:, 1].cpu())
    return torch.cat(out).numpy()

def train_eval(X, y, tr, va, te, max_epochs=200, patience=30, kind='eegnet'):
    sd = X[tr].std() + 1e-6
    T = lambda idx: torch.tensor(X[idx] / sd, dtype=torch.float32).unsqueeze(1).to(DEV)
    Xtr, Xva, Xte = T(tr), T(va), T(te); ytr = torch.tensor(y[tr]).to(DEV)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = make_model(kind, X.shape[1], X.shape[2]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=8, factor=0.5)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); best, best_state, wait = -1, None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(augment(Xtr[idx])), ytr[idx]).backward(); opt.step()
        va_acc = ((predict(model, Xva) > 0.5).astype(int) == y[va]).mean(); sch.step(va_acc)
        if va_acc > best: best, best_state, wait = va_acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); p = predict(model, Xte); pred = (p > 0.5).astype(int); yt = y[te]
    m = dict(accuracy=accuracy_score(yt, pred), balanced_accuracy=balanced_accuracy_score(yt, pred),
             precision=precision_score(yt, pred, zero_division=0), recall=recall_score(yt, pred, zero_division=0),
             f1=f1_score(yt, pred, zero_division=0), macro_f1=f1_score(yt, pred, average='macro'),
             auc=roc_auc_score(yt, p), val_acc=best, epochs=ep + 1)
    cpu = copy.deepcopy(model).cpu().eval(); x1 = Xte[:1].cpu(); torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(5): cpu(x1)
        t0 = time.perf_counter()
        for _ in range(50): cpu(x1)
    buf = io.BytesIO(); torch.save(cpu.state_dict(), buf)
    m.update(params=sum(q.numel() for q in model.parameters()), size_kb=round(buf.getbuffer().nbytes / 1024, 1),
             latency_ms_cpu=round((time.perf_counter() - t0) / 50 * 1000, 2))
    return m

def main(tasks):
    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV); done = set(zip(d.task_family, d.fold, d.n_channels, d.ea, d.model))
    for fam in tasks:
        z = np.load(CACHE / f'{fam}.npz', allow_pickle=True); X, y, g, ch = z['X'], z['y'], z['g'], list(z['ch'])
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g), 1):
            a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED + fold).split(X[tr], y[tr], groups=g[tr]))
            tr_i, va_i = tr[a], tr[b]
            for nch, ea, kind in CONFIGS:
                if (fam, fold, nch, ea, kind) in done: continue
                t0 = time.time(); sel = np.arange(64) if nch == 64 else select16(X[tr_i], y[tr_i], nch)
                Xs = car(X[:, sel, :])
                if ea: Xs = euclid_align(Xs, g)
                m = train_eval(Xs, y, tr_i, va_i, te, kind=kind)
                row = dict(task_family=fam, fold=fold, n_channels=nch, ea=ea, model=kind, **m, n_test=len(te),
                           seconds=round(time.time() - t0), channels=' '.join(ch[i] for i in sel))
                pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)
                print(fam, fold, nch, 'EA' if ea else '--', kind, f"acc={m['accuracy']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} params={m['params']} {row['seconds']}s", flush=True)

if __name__ == '__main__':
    main(sys.argv[1:] or TASKS)
