"""Verification of the streaming pipeline (rt_pipeline.py): causality tests, replay agreement with saved outputs, per-stage timing.
usage: python verify_streaming.py [tests|timing]      (run `timing` only when the machine is otherwise idle)
Writes outputs_final/streaming_verification.json
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt
import pandas as pd
import rt_pipeline as RT
ROOT = Path(os.environ.get('EEG_PROJECT_ROOT', r'C:\SEM_7\PhaseProject')); OUT = ROOT / 'outputs_final' / 'streaming_verification.json'
TASKS = ['execution_bilateral', 'execution_unilateral', 'imagery_bilateral', 'imagery_unilateral']
res = json.load(open(OUT)) if OUT.exists() else {}

def tests():
    rng = np.random.RandomState(0); out = {}
    # 1) filter causality: output at sample i must not depend on samples > i
    buf = rng.randn(16, RT.N_BUF); t0 = 400; b2 = buf.copy(); b2[:, t0 + 1:] += 50 * rng.randn(16, RT.N_BUF - t0 - 1)
    f1, f2 = sosfilt(RT.SOS, buf, axis=-1), sosfilt(RT.SOS, b2, axis=-1)
    out['filter_output_up_to_t_independent_of_future_samples'] = bool(np.abs(f1[:, :t0 + 1] - f2[:, :t0 + 1]).max() < 1e-12)
    out['filter_state_note'] = 'Each decision filters its own 4 s buffer from zero state (first 1 s = warm-up); state is NOT carried between decisions. Buffer demean uses the buffer mean (available at decision time).'
    # 2) online reference: score of window j must not depend on windows after j
    d = np.load(ROOT / 'deploy_rt' / TASKS[0] / 'replay.npz'); m = dict(np.load(ROOT / 'deploy_rt' / TASKS[0] / 'model.npz')); g = d['g']
    first = np.where(g == g[0])[0][:30]; X = d['X'][first].astype(np.float32); p1 = RT.Online(m); s1 = [p1.step(x) for x in X]
    X2 = X.copy(); X2[15:] = rng.randn(*X2[15:].shape).astype(np.float32) * 20; p2 = RT.Online(m); s2 = [p2.step(x) for x in X2]
    out['online_scores_for_windows_0_14_unchanged_when_future_windows_altered'] = bool(np.allclose(s1[:15], s2[:15], atol=1e-9))
    out['online_scores_for_windows_15plus_do_change'] = bool(not np.allclose(s1[15:], s2[15:]))
    # 3) reset per user: scoring a user's windows after another user equals scoring them fresh
    other = np.where(g != g[0])[0][:10]; pa = RT.Online(m); [pa.step(x) for x in d['X'][other].astype(np.float32)]; pa.reset(); sa = [pa.step(x) for x in X[:10]]
    out['reset_between_users_equals_fresh_state'] = bool(np.allclose(sa, s1[:10], atol=1e-9))
    # 4) replay agreement with saved outputs (table_k fold 1) + metrics
    k = pd.read_csv(ROOT / 'outputs_rigorous_benchmark' / 'table_k_streaming_nested.csv'); rep = {}
    for t in TASKS:
        b = ROOT / 'deploy_rt' / t; mm = dict(np.load(b / 'model.npz')); dd = np.load(b / 'replay.npz'); X, y, g = dd['X'], dd['y'], dd['g']
        pipe = RT.Online(mm); sc, prev = [], None
        for ep, s in zip(X, g):
            if s != prev: pipe.reset(); prev = s
            sc.append(pipe.step(ep))
        p = (np.array(sc) > 0).astype(int); tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1); bal = (rc + tn / max(tn + fp, 1)) / 2
        saved = float(k[(k.task_family == t) & (k.fold == 1)].accuracy.iloc[0])
        rep[t] = dict(n_windows=int(len(y)), n_subjects=int(len(np.unique(g))), accuracy=float((p == y).mean()), saved_fold1_accuracy=saved, agrees=bool(abs((p == y).mean() - saved) < 1e-9),
                      balanced_accuracy=float(bal), precision=float(pr), recall=float(rc), f1=float(2 * pr * rc / max(pr + rc, 1e-9)), w0=float(mm['w0']), scale_prior=int(mm.get('scale_prior', 0)))
    out['replay'] = rep; res['tests'] = out; json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(out, indent=1))

def timing(reps=3):
    out = {}
    for t in TASKS:
        b = ROOT / 'deploy_rt' / t; mm = dict(np.load(b / 'model.npz')); dd = np.load(b / 'replay.npz'); X, g = dd['X'][:400], dd['g'][:400]
        runs = []
        for _ in range(reps):
            pipe = RT.Online(mm); prev = None; st = {k: [] for k in ('filter_car', 'cov', 'recenter', 'tangent_lda', 'full_window')}
            for ep, s in zip(X, g):
                if s != prev: pipe.reset(); prev = s
                t0 = time.perf_counter(); w = RT.window(ep); t1 = time.perf_counter(); C = RT.cov(w); t2 = time.perf_counter()
                pipe.S += C; pipe.n += 1; Pm = pipe.P * (np.trace(pipe.S) / pipe.n / np.trace(pipe.P)) if pipe.scale_prior else pipe.P
                M = (pipe.w0 * Pm + pipe.S) / (pipe.w0 + pipe.n); Cw = RT.whiten(C, M); t3 = time.perf_counter(); f = RT.tangent(Cw); _ = f @ pipe.coef + pipe.b; t4 = time.perf_counter()
                for k_, v in zip(st, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)): st[k_].append(v * 1000)
            runs.append(st)
        med = {k: {s: float(np.median([np.__dict__[s](r[k]) if s in ('mean', 'max') else np.percentile(r[k], 50 if s == 'median' else 95) for r in runs])) for s in ('mean', 'median', 'p95', 'max')} for k in runs[0]}
        out[t] = med
    out['note'] = ('Per-window processing times in ms (median over 3 passes of 400 windows). Includes: demean+causal filter+CAR, covariance, online reference update + whitening, '
                   'tangent map + LDA score. Excludes: acquisition/buffering, I/O, Python start-up. Host: ' + os.environ.get('COMPUTERNAME', 'unknown') + '. Not a Raspberry Pi measurement.')
    res['timing'] = out; json.dump(res, open(OUT, 'w'), indent=1); print(json.dumps(out, indent=1)[:3000])

if __name__ == '__main__':
    {'tests': tests, 'timing': timing}[sys.argv[1] if len(sys.argv) > 1 else 'tests']()
