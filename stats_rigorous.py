"""Paired statistics + bootstrap CIs with Holm/Bonferroni for outputs_rigorous_benchmark CSVs."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

D = Path(r'C:\SEM_7\PhaseProject\outputs_rigorous_benchmark')
rng = np.random.RandomState(42)

def boot_ci(d, n=5000):
    m = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj

def family(df, comps, unit, test):
    rows = []
    for task, g in df.groupby('task_family'):
        for name, (a, b) in comps.items():
            d = (g[a] - g[b]).values
            if test == 'wilcoxon':
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            else:
                p = stats.ttest_rel(g[a], g[b]).pvalue
            lo, hi = boot_ci(d)
            rows.append(dict(task=task, comparison=name, n=len(d), unit=unit, mean_a=g[a].mean(), mean_b=g[b].mean(),
                             mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, p_raw=p))
    r = pd.DataFrame(rows)
    r['p_bonferroni'] = np.minimum(1, r.p_raw * len(r)); r['p_holm'] = holm(r.p_raw.values)
    return r

a = pd.read_csv(D / 'table_a_subject_independent_5fold.csv')
b = pd.read_csv(D / 'table_b_withinsubject_109subjs.csv')
A = {'fbcsp 16-64': ('acc_16_fbcsp', 'acc_64_fbcsp'), 'riemann 16-64': ('acc_16_riemann', 'acc_64_riemann'),
     'riemann recentered16 - 16': ('acc_16_recentered', 'acc_16_riemann'), 'emotiv14 - 16 (riemann)': ('acc_emotiv_riemann', 'acc_16_riemann')}
B = {'fbcsp pers-fixed': ('fbcsp_pers', 'fbcsp_fixed'), 'riemann pers-fixed': ('riemann_pers', 'riemann_fixed'),
     'emotiv pers8-fixed14': ('emotiv_pers8', 'emotiv_fixed14')}
ra = family(a, A, 'fold (n=5)', 'ttest'); rb = family(b, B, 'subject', 'wilcoxon')
ra.to_csv(D / 'stats_table_a.csv', index=False); rb.to_csv(D / 'stats_table_b.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.float_format', '{:.4f}'.format)
print(ra.drop(columns='unit').to_string()); print(rb.drop(columns='unit').to_string())
