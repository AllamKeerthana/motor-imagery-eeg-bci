import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
B = Path(r'C:\SEM_7\PhaseProject\outputs_rigorous_benchmark'); O = Path(r'C:\SEM_7\PhaseProject\outputs_final')
s = pd.read_csv(B / 'table_g_nested_selected.csv'); r = pd.read_csv(B / 'table_g_reference_variants.csv'); e = pd.read_csv(B / 'table_e_channel_sweep.csv')
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); a[i] = min(1, run)
    return a
rows = []
for t, g in s.groupby('task_family'):
    g = g.set_index('fold').accuracy
    for name, ref in (('t-test16 baseline', e[(e.k == 16)]), ('64ch', e[(e.k == 64)])):
        b = ref[ref.task_family == t].set_index('fold').accuracy
        rows.append(dict(task=t, vs=name, nested=g.mean(), other=b.mean(), diff=(g - b).mean(), p_raw=stats.ttest_rel(g, b).pvalue, folds_better=int((g > b).sum())))
d = pd.DataFrame(rows); d['p_holm'] = holm(d.p_raw.values); d.round(4).to_csv(O / 'stats_nested_vs_baselines.csv', index=False)
pd.set_option('display.width', 200); print(d.round(4).to_string())
print(s.groupby('task_family')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean().round(3).to_string())
print(r.groupby(['task_family', 'variant']).accuracy.mean().unstack(0).round(3).to_string())
