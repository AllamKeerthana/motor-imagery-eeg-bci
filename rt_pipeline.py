"""Streaming motor-imagery pipeline: numpy + scipy ONLY (runs on a Raspberry Pi).
Stages per decision (all causal w.r.t. the decision time):
  buffer (16 sensorimotor ch, 641 samples = 4 s; the first 1 s is filter warm-up)
  -> per-channel demean -> causal 8-30 Hz Butterworth (sosfilt, order 4)
  -> last 481 samples (3 s) -> CAR over the 16 channels -> sample covariance (+5% shrinkage)
  -> ADAPTIVE PREPROCESSING: online label-free re-centering
         M_t = (w0*P + sum_{i<=t} C_i) / (w0 + t),   C~ = M_t^{-1/2} C M_t^{-1/2}
     (P = mean covariance of TRAINING subjects, w0 = prior weight; state is reset for every new user)
  -> tangent-space (log-map, 136 features) -> linear LDA.
usage (replay benchmark):  python rt_pipeline.py deploy_rt/<task>
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
from scipy.signal import sosfilt, butter

FS, N_WIN, N_BUF, K, ALPHA = 160, 481, 641, 16, 0.05
SOS = butter(4, [8, 30], btype='bandpass', fs=FS, output='sos')
TRI = np.triu_indices(K)
TSCALE = np.where(TRI[0] == TRI[1], 1.0, np.sqrt(2.0))


def window(buf):
    """(..., 16, 641) raw -> (..., 16, 481) causally filtered + CAR over the 16 channels"""
    x = buf - buf.mean(-1, keepdims=True)
    x = sosfilt(SOS, x, axis=-1)[..., -N_WIN:]
    return x - x.mean(-2, keepdims=True)


def cov(x):
    C = x @ np.swapaxes(x, -1, -2) / x.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / K
    return (1 - ALPHA) * C + ALPHA * tr * np.eye(K)


def inv_sqrt(M):
    w, v = np.linalg.eigh(M)
    return (v / np.sqrt(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(v, -1, -2)


def tangent(Cw):
    w, v = np.linalg.eigh(Cw)
    L = (v * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(v, -1, -2)
    return L[..., TRI[0], TRI[1]] * TSCALE


def whiten(C, M):
    W = inv_sqrt(M)
    return W @ C @ np.swapaxes(W, -1, -2)


def online_means(C, P, w0, scale_prior=False):
    """C: (n,16,16) covariances of ONE user in arrival order -> running mean M_t for every t.
    scale_prior: rescale the population prior P to the user's own running mean trace (removes amplitude mismatch)."""
    S = np.cumsum(C, axis=0)
    t = np.arange(1, len(C) + 1)[:, None, None]
    Pm = P
    if scale_prior:
        Pm = P[None] * (np.trace(S, axis1=1, axis2=2)[:, None, None] / t) / np.trace(P)
    return (w0 * Pm + S) / (w0 + t)


class Online:
    def __init__(self, model):
        self.coef = model['coef']
        self.b = float(model['intercept'])
        self.P = model['P']
        self.w0 = float(model['w0'])
        self.scale_prior = bool(model.get('scale_prior', 0))
        self.reset()

    def reset(self):
        self.S = np.zeros((K, K))
        self.n = 0

    def step(self, buf):
        C = cov(window(buf))
        self.S += C
        self.n += 1
        Pm = self.P * (np.trace(self.S) / self.n / np.trace(self.P)) if self.scale_prior else self.P
        M = (self.w0 * Pm + self.S) / (self.w0 + self.n)
        return float(tangent(whiten(C, M)) @ self.coef + self.b)


def replay(bundle):
    b = Path(bundle)
    m = dict(np.load(b / 'model.npz'))
    d = np.load(b / 'replay.npz')
    X, y, g = d['X'], d['y'], d['g']
    pipe = Online(m)
    scores, lat, prev = [], [], None
    for ep, s in zip(X, g):
        if s != prev:
            pipe.reset()
            prev = s
        buf = np.zeros((K, N_BUF), np.float32)
        for i in range(0, N_BUF, 16):  # simulated acquisition in 100 ms chunks
            c = ep[:, i:i + 16]
            buf = np.roll(buf, -c.shape[1], axis=1)
            buf[:, -c.shape[1]:] = c
        t0 = time.perf_counter()
        sc = pipe.step(buf)
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(sc)
    p = (np.array(scores) > 0).astype(int)
    tp = ((p == 1) & (y == 1)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    out = dict(bundle=b.name, n_windows=int(len(y)), n_users=int(len(np.unique(g))), accuracy=round(float((p == y).mean()), 4),
               precision=round(float(pr), 4), recall=round(float(rc), 4), f1=round(float(2 * pr * rc / max(pr + rc, 1e-9)), 4),
               decision_latency_ms_mean=round(float(np.mean(lat)), 3), decision_latency_ms_p95=round(float(np.percentile(lat, 95)), 3),
               decision_latency_ms_max=round(float(np.max(lat)), 3), window_s=3.0, rss_mb=round(float(rss), 1),
               host=os.environ.get('COMPUTERNAME', os.uname().nodename if hasattr(os, 'uname') else 'unknown'))
    print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    replay(sys.argv[1])
