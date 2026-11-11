import numpy as np, pandas as pd, glob
from pathlib import Path
from scipy import stats
B = Path(r'C:\SEM_7\PhaseProject\outputs_rigorous_benchmark'); O = Path(r'C:\SEM_7\PhaseProject\outputs_final'); rng = np.random.RandomState(42)
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(B / 'table_i_adaptive_channels_*.csv'))])
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); a[i] = min(1, run)
    return a
rows = []
for t, g in d.groupby('task_family'):
    w = g.pivot_table(index=['fold', 'subject'], columns='arm', values='accuracy').dropna()
    for arm in ('lam0.25', 'lam0.5', 'lam1.0', 'rand'):
        x = (w[arm] - w['sm']).values; ci = np.percentile([rng.choice(x, len(x)).mean() for _ in range(5000)], [2.5, 97.5])
        rows.append(dict(task=t, arm=arm, n_subj=len(w), acc_sm=w['sm'].mean(), acc_arm=w[arm].mean(), diff=x.mean(), ci_lo=ci[0], ci_hi=ci[1],
                         better=int((x > 0).sum()), worse=int((x < 0).sum()), p_raw=stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0))
r = pd.DataFrame(rows); prim = r.arm == 'lam0.5'
r['p_holm_primary'] = np.nan; r.loc[prim, 'p_holm_primary'] = holm(r.loc[prim, 'p_raw'].values)
sec = ~prim; r['p_holm_secondary'] = np.nan; r.loc[sec, 'p_holm_secondary'] = holm(r.loc[sec, 'p_raw'].values)
r.round(4).to_csv(O / 'stats_adaptive_channels.csv', index=False); pd.set_option('display.width', 220); print(r.round(4).to_string())
