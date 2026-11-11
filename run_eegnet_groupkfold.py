"""Subject-independent 5-fold GroupKFold EEGNet (64 vs 16 ch) on the SAME folds as run_complete_rigorous_benchmark.py.
- Epoch cache per task (cache_epochs/*.npz), no CAR stored.
- Channel selection: training-fold subjects only, on non-CAR data.
- CAR applied over the channels actually used. Early stopping on held-out validation SUBJECTS from the training fold.
- Resumable: appends one row per (task, fold, n_ch) to outputs_eegnet_groupkfold/table_c_eegnet.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import run_complete_rigorous_benchmark as R
import all_tasks_adaptive_sparse_common as C

ROOT = Path(r'C:\SEM_7\PhaseProject'); CACHE = ROOT / 'cache_epochs'; OUT = ROOT / 'outputs_eegnet_groupkfold'
CACHE.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
CSV = OUT / 'table_c_eegnet.csv'

def load_family(family):
    f = CACHE / f'{family}.npz'
    if f.exists():
        z = np.load(f, allow_pickle=True); return z['X'], z['y'], z['g'], list(z['ch'])
    recs = [(s, *r) for s in range(1, 110) if (r := R.load_subject_raw_epochs(s, R.TASK_FAMILIES[family])) is not None]
    X = np.concatenate([r[1] for r in recs]); y = np.concatenate([r[2] for r in recs])
    g = np.concatenate([[r[0]] * len(r[2]) for r in recs]); ch = recs[0][3]
    np.savez(f, X=X, y=y, g=g, ch=np.array(ch)); return X, y, g, ch

def main(only_tasks=None, max_epochs=150):
    C.seed_everything()
    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV); done = set(zip(d.task_family, d.fold, d.n_channels))
    for family in R.TASK_FAMILIES:
        if only_tasks and family not in only_tasks: continue
        X, y, g, ch = load_family(family)
        print(family, X.shape, flush=True)
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g), 1):
            vs = GroupShuffleSplit(1, test_size=0.15, random_state=C.SEED + fold)
            a, b = next(vs.split(X[tr], y[tr], groups=g[tr])); tr_i, va_i = tr[a], tr[b]
            for nch in (64, 16):
                if (family, fold, nch) in done: continue
                t0 = time.time()
                if nch == 64: sel = np.arange(X.shape[1]); names = ch
                else: sel, names = R.select_top_channels_train_only(X[tr_i], y[tr_i], ch, 16)
                Xs = R.apply_honest_car(X[:, sel, :])
                m = C.fit_and_score(Xs, y, tr_i, va_i, te, epochs=max_epochs)
                row = dict(task_family=family, fold=fold, n_channels=nch, accuracy=m['accuracy'], balanced_accuracy=m['balanced_accuracy'],
                           macro_f1=m['macro_f1'], val_acc=m['validation_accuracy'], n_test=m['n_test_trials'],
                           n_test_subjects=len(np.unique(g[te])), seconds=round(time.time() - t0), channels=' '.join(names))
                pd.DataFrame([row]).to_csv(CSV, mode='a', header=not CSV.exists(), index=False)
                print(row['task_family'], fold, nch, round(row['accuracy'], 4), row['seconds'], 's', flush=True)

if __name__ == '__main__':
    main(only_tasks=sys.argv[1:] or None)
