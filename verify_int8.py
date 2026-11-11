"""fp32 vs int8 ONNX comparison on identical workloads (same replay windows, same preprocessing, same session options, 1 thread).
Reports: accuracy of each, prediction agreement, max |logit difference|, session.run latency (median of 3 passes of per-window means), file sizes.
Run with the Python that has onnxruntime (cuda_env).  Output: outputs_final/int8_verification.json
"""
import os, json, time
from pathlib import Path
import numpy as np, onnxruntime as ort
ROOT = Path(os.environ.get('EEG_PROJECT_ROOT', r'C:\SEM_7\PhaseProject'))
out = {}
for t in ('execution_bilateral', 'execution_unilateral', 'imagery_bilateral', 'imagery_unilateral'):
    b = ROOT / 'deploy' / f'{t}_compact_sm'; meta = json.load(open(b / 'meta.json')); d = np.load(b / 'replay.npz'); X, y = d['X'], d['y']
    W = X - X.mean(1, keepdims=True); W = (W / meta['scale']).astype(np.float32)[:, None]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; r = {}
    sess = {p: ort.InferenceSession(str(b / f'model_{p}.onnx'), so, providers=['CPUExecutionProvider']) for p in ('fp32', 'int8')}
    logits = {p: np.concatenate([s.run(None, {'x': W[i:i + 1]})[0] for i in range(len(W))]) for p, s in sess.items()}
    pred = {p: logits[p].argmax(1) for p in logits}
    for p in ('fp32', 'int8'):
        lat = []
        for _ in range(3):
            tt = []
            for i in range(min(300, len(W))):
                t0 = time.perf_counter(); sess[p].run(None, {'x': W[i:i + 1]}); tt.append((time.perf_counter() - t0) * 1000)
            lat.append(float(np.mean(tt)))
        r[p] = dict(accuracy=float((pred[p] == y).mean()), run_latency_ms_median_of_3_passes=float(np.median(lat)), file_kb=round((b / f'model_{p}.onnx').stat().st_size / 1024, 2))
    r['prediction_agreement'] = float((pred['fp32'] == pred['int8']).mean()); r['max_abs_logit_diff'] = float(np.abs(logits['fp32'] - logits['int8']).max()); r['n_windows'] = int(len(y))
    out[t] = r
out['note'] = 'Same windows, same preprocessing, 1 intra-op thread; latency = onnxruntime session.run on a single window (excludes CAR/scaling). Static QDQ int8 quantisation with 200 calibration windows from TRAIN subjects. Measured while a GPU training job was running on another core; not a Raspberry Pi result.'
json.dump(out, open(ROOT / 'outputs_final' / 'int8_verification.json', 'w'), indent=1); print(json.dumps(out, indent=1))
