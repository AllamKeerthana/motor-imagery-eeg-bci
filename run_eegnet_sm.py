"""EEGNet on the fixed a-priori sensorimotor 16-ch set, same GroupKFold folds/validation scheme as run_eegnet_gpu.py. Output: table_c_eegnet_sm.csv"""
import time, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import run_eegnet_gpu as G
SM = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4', 'P3', 'PZ', 'P4']
CSV = G.OUT / 'table_c_eegnet_sm.csv'
done = set(zip(*[pd.read_csv(CSV)[c] for c in ('task_family', 'fold', 'model')])) if CSV.exists() else set()
for fam in G.TASKS:
    z = np.load(G.CACHE / f'{fam}.npz', allow_pickle=True); X, y, g, ch = z['X'], z['y'], z['g'], [c.strip('.').upper() for c in z['ch']]
    sel = np.array([ch.index(c) for c in SM]); Xs = G.car(X[:, sel, :])
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=g), 1):
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=G.SEED + fold).split(X[tr], y[tr], groups=g[tr])); tr_i, va_i = tr[a], tr[b]
        for kind in ('compact', 'eegnet'):
            if (fam, fold, kind) in done: continue
            t0 = time.time(); m = G.train_eval(Xs, y, tr_i, va_i, te, kind=kind)
            row = dict(task_family=fam, fold=fold, model=kind, n_channels=16, chan_set='sensorimotor', **m, seconds=round(time.time() - t0))
            pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)
            print(fam, fold, kind, round(m['accuracy'], 3), round(m['f1'], 3), row['seconds'], 's', flush=True)
