import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
B = Path(r'C:\SEM_7\PhaseProject\outputs_rigorous_benchmark'); O = Path(r'C:\SEM_7\PhaseProject\outputs_final')
j = pd.read_csv(B / 'table_j_streaming.csv'); k = pd.read_csv(B / 'table_k_streaming_nested.csv')
acc = j.pivot_table(index=['task_family', 'fold'], columns='condition', values='accuracy')
acc['K_online_nested'] = k.set_index(['task_family', 'fold']).accuracy
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); a[i] = min(1, run)
    return a
rows = []
for t, g in acc.groupby(level=0):
    for a_, b_ in (('K_online_nested', 'N_causal_no_adaptation'), ('K_online_nested', 'B_causal_batch'), ('K_online_nested', 'C_causal_online_w0=5'), ('B_causal_batch', 'N_causal_no_adaptation'), ('B_causal_batch', 'A_zerophase_batch')):
        x, y = g[a_], g[b_]; rows.append(dict(task=t, a=a_, b=b_, acc_a=x.mean(), acc_b=y.mean(), diff=(x - y).mean(), folds_a_better=int((x > y).sum()), p_raw=stats.ttest_rel(x, y).pvalue))
r = pd.DataFrame(rows); r['p_holm_family'] = np.nan
for key, idx in r.groupby(['a', 'b']).groups.items(): r.loc[idx, 'p_holm_family'] = holm(r.loc[idx, 'p_raw'].values)
r.round(4).to_csv(O / 'stats_streaming.csv', index=False); pd.set_option('display.width', 220); print(r.round(4).to_string())
summ = acc.groupby(level=0).mean().round(3)[['A_zerophase_batch', 'B_causal_batch', 'N_causal_no_adaptation', 'C_causal_online_w0=5', 'K_online_nested']]; print(summ.to_string()); summ.to_csv(O / 'table_streaming_summary.csv')
print(k.groupby('task_family')[['accuracy', 'precision', 'recall', 'f1', 'auc']].mean().round(3).to_string())
