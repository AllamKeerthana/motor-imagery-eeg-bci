"""Real-time replay + latency benchmark. Needs ONLY numpy + onnxruntime (runs on a Raspberry Pi).
usage: python pi_infer.py deploy/<bundle> [fp32|int8]
Streams the held-out test epochs chunk by chunk (16 samples = 100 ms) into a ring buffer (16 channels). When a full
481-sample window is available it applies online CAR + scale and runs the ONNX model.
Reports accuracy / precision / recall / F1, per-inference latency (mean, p95), model size, process memory.
"""
import sys, json, time, os
from pathlib import Path
import numpy as np, onnxruntime as ort

b = Path(sys.argv[1]); prec = sys.argv[2] if len(sys.argv) > 2 else 'fp32'
meta = json.load(open(b / 'meta.json')); d = np.load(b / 'replay.npz'); X, y = d['X'], d['y']
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(str(b / f'model_{prec}.onnx'), so, providers=['CPUExecutionProvider'])
N, CH, CHUNK = meta['n_samples'], len(meta['channels']), 16
preds, lat = [], []
for ep in X:
    ring = np.zeros((CH, N), np.float32); filled = 0
    for s in range(0, N, CHUNK):                       # simulated acquisition
        c = ep[:, s:s + CHUNK]; ring = np.roll(ring, -c.shape[1], axis=1); ring[:, -c.shape[1]:] = c; filled += c.shape[1]
    t0 = time.perf_counter()                            # window complete -> decide
    w = ring - ring.mean(0, keepdims=True); w = (w / meta['scale']).astype(np.float32)[None, None]
    out = sess.run(None, {'x': w})[0][0]; lat.append((time.perf_counter() - t0) * 1000); preds.append(int(out[1] > out[0]))
p = np.array(preds); tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
try:
    import resource; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
except ImportError:
    import psutil; rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
print(json.dumps(dict(bundle=b.name, precision_fmt=prec, n=len(y), accuracy=round(float((p == y).mean()), 4), precision=round(float(pr), 4), recall=round(float(rc), 4),
    f1=round(float(2 * pr * rc / max(pr + rc, 1e-9)), 4), latency_ms_mean=round(float(np.mean(lat)), 2), latency_ms_p95=round(float(np.percentile(lat, 95)), 2),
    model_kb=round((b / f'model_{prec}.onnx').stat().st_size / 1024, 1), rss_mb=round(float(rss), 1), host=os.environ.get('COMPUTERNAME', 'unknown')), indent=1))
