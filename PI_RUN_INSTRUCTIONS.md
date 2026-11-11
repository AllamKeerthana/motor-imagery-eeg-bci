# Raspberry Pi: run the real-time replay benchmark

The pipeline needs only Python 3, numpy and scipy (`pip install numpy scipy`). No torch, mne or onnxruntime.

1. Copy these to the Pi (same folder): `rt_pipeline.py` and the folder `deploy_rt/` (four sub-folders, one per task; each has `model.npz` and `replay.npz`, replay is about 40 MB per task).
2. Run, for each task:

```
python3 rt_pipeline.py deploy_rt/execution_bilateral
python3 rt_pipeline.py deploy_rt/execution_unilateral
python3 rt_pipeline.py deploy_rt/imagery_bilateral
python3 rt_pipeline.py deploy_rt/imagery_unilateral
```

3. Each run prints JSON with accuracy, precision, recall, F1, `decision_latency_ms_mean`, `_p95`, `_max`, memory (`rss_mb`) and the host name. Save the output.

Expected: the same accuracies as on the laptop (0.806, 0.709, 0.635, 0.665 for execution bilateral, execution unilateral, imagery bilateral, imagery unilateral; small float differences possible). Laptop latency is about 0.5 ms; the Pi will be slower. A decision is only useful if latency is far below the 3 s window.

Report these as "measured on Raspberry Pi <model>, single process" and add the Pi model, OS and Python version. Do not quote the laptop latency as a Pi result. Optional: also record CPU temperature and power (`vcgencmd measure_temp`).

Optional ONNX EEGNet check: `deploy/<task>_compact_sm/` holds fp32 and int8 ONNX models; `pi_infer.py` needs `onnxruntime` (`pip install onnxruntime`). fp32 is recommended; int8 was slower and larger.
