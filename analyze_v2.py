"""Subject-level analysis for the corrected experiments. Paired unit = test SUBJECT (each subject is in exactly one test fold -> 109 independent units).
Reports effect size + 95% bootstrap CI (resampling subjects), Wilcoxon signed-rank p, Holm correction within each declared family (the 4 tasks per comparison),
and an equivalence analysis (TOST on the paired mean difference, margin DELTA = 0.03 accuracy, chosen as a practical margin BEFORE looking at results; also shown for 0.05).
A non-significant difference is NOT treated as equivalence: 'equivalent' only if the 90% CI of the difference lies inside +-DELTA.
Outputs -> outputs_final/v2_*.csv
"""
import os, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT = Path(os.environ.get('EEG_PROJECT_ROOT', r'C:\SEM_7\PhaseProject')); B = ROOT / 'outputs_rigorous_benchmark'; O = ROOT / 'outputs_final'; O.mkdir(exist_ok=True)
TASKS = ['execution_bilateral', 'execution_unilateral', 'imagery_bilateral', 'imagery_unilateral']; DELTA = 0.03; rng = np.random.RandomState(42)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); a[i] = min(1, run)
    return a
def boot(d, q=(2.5, 97.5), n=5000):
    idx = rng.randint(0, len(d), (n, len(d))); return np.percentile(d[idx].mean(1), q)
def paired(d):
    ci = boot(d); ci90 = boot(d, (5, 95)); p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n_subjects=len(d), mean_diff=d.mean(), ci95_lo=ci[0], ci95_hi=ci[1], p_wilcoxon=p, ci90_lo=ci90[0], ci90_hi=ci90[1],
                equivalent_pm0_03=bool(ci90[0] > -DELTA and ci90[1] < DELTA), equivalent_pm0_05=bool(ci90[0] > -0.05 and ci90[1] < 0.05))
def finish(rows, name):
    r = pd.DataFrame(rows); r['p_holm_within_comparison'] = np.nan
    for c, idx in r.groupby('comparison').groups.items(): r.loc[idx, 'p_holm_within_comparison'] = holm(r.loc[idx, 'p_wilcoxon'].values)
    r.round(4).to_csv(O / name, index=False); return r

# ---------- A/B: unified channel experiment ----------
u = pd.concat([pd.read_csv(f) for f in glob.glob(str(B / 'table_u_channels_*.csv'))]) if glob.glob(str(B / 'table_u_channels_*.csv')) else None
if u is not None:
    u['acc'] = (u.tp + u.tn) / u.n; u['cfg'] = u.config.where(~u.config.str.startswith('rand16'), 'rand16')
    S = u.groupby(['task_family', 'cfg', 'subject']).acc.mean().reset_index()               # random sets averaged per subject
    P = S.pivot_table(index=['task_family', 'subject'], columns='cfg', values='acc')
    summ = []
    for t in TASKS:
        for c in ['sm16', 'tt16', 'ttz16', 'rand16', 'tt8', 'tt24', 'tt32', 'all64']:
            if (t, ) and t in P.index.get_level_values(0) and c in P.columns:
                v = P.loc[t][c].dropna().values; g = u[(u.task_family == t) & (u.config.str.startswith(c if c != 'rand16' else 'rand16'))]
                tp, fp, fn, tn = g.tp.sum(), g.fp.sum(), g.fn.sum(), g.tn.sum(); pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                ci = boot(v); summ.append(dict(task=t, config=c, n_subjects=len(v), subject_mean_acc=v.mean(), ci95_lo=ci[0], ci95_hi=ci[1], pooled_trial_acc=(tp + tn) / (tp + fp + fn + tn),
                                               balanced_acc=(rc + tn / max(tn + fp, 1)) / 2, precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9), macro_f1=(2 * pr * rc / max(pr + rc, 1e-9) + 2 * (tn / max(tn + fn, 1)) * (tn / max(tn + fp, 1)) / max((tn / max(tn + fn, 1)) + (tn / max(tn + fp, 1)), 1e-9)) / 2,
                                               n_features=int(g.n_features.iloc[0]), tangent_lda_ms_per_trial=g.tangent_lda_ms_per_trial.mean()))
    pd.DataFrame(summ).round(4).to_csv(O / 'v2_channel_configs_summary.csv', index=False)
    rows = []
    for t in TASKS:
        if t not in P.index.get_level_values(0): continue
        X = P.loc[t].dropna()
        for a, b in (('sm16', 'tt16'), ('sm16', 'ttz16'), ('sm16', 'rand16'), ('sm16', 'all64'), ('ttz16', 'all64'), ('tt16', 'all64')):
            if a in X and b in X: rows.append(dict(task=t, comparison=f'{a} - {b}', **paired((X[a] - X[b]).values)))
    r = finish(rows, 'v2_channel_comparisons.csv'); pd.set_option('display.width', 250); print(r.round(4).to_string())
    # channel-count trade-off (tt ranking prefixes + all64) with incremental gains
    inc = []
    for t in TASKS:
        if t not in P.index.get_level_values(0): continue
        X = P.loc[t].dropna(); ks = [('tt8', 8), ('tt16', 16), ('tt24', 24), ('tt32', 32), ('all64', 64)]
        for (a, ka), (b, kb) in zip(ks[:-1], ks[1:]):
            d = (X[b] - X[a]).values; ci = boot(d); inc.append(dict(task=t, step=f'{ka}->{kb} ch', mean_gain=d.mean(), gain_per_added_channel=d.mean() / (kb - ka), ci95_lo=ci[0], ci95_hi=ci[1],
                                                                    p_wilcoxon=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
    pd.DataFrame(inc).round(4).to_csv(O / 'v2_channel_count_increments.csv', index=False)

# ---------- C: nested v1 vs v2 ----------
v2 = [pd.read_csv(f) for f in glob.glob(str(B / 'table_g2_nested_selected_*.csv'))]
if v2:
    n2 = pd.concat(v2); n1 = pd.read_csv(B / 'table_g_nested_selected.csv')
    cmp = pd.DataFrame({'v1_accuracy': n1.groupby('task_family').accuracy.mean(), 'v2_accuracy': n2.groupby('task_family').accuracy.mean(),
                        'v2_balanced_acc': n2.groupby('task_family').balanced_accuracy.mean(), 'v2_macro_f1': n2.groupby('task_family').macro_f1.mean(), 'v2_folds': n2.groupby('task_family').size()}).reindex(TASKS)
    cmp['diff_v2_minus_v1'] = cmp.v2_accuracy - cmp.v1_accuracy; cmp.round(4).to_csv(O / 'v2_nested_old_vs_corrected.csv'); print(cmp.round(4).to_string())
    sel = n2.groupby(['task_family', 'variant']).size().unstack(0).fillna(0).astype(int); sel.to_csv(O / 'v2_nested_selection_counts.csv'); print(sel)

# ---------- D/E: EEGNet seeds, equal-information model comparison ----------
sf = B.parent / 'outputs_eegnet_groupkfold' / 'table_c_eegnet_seeds.csv'
if sf.exists():
    s = pd.read_csv(sf); base = pd.read_csv(B.parent / 'outputs_eegnet_groupkfold' / 'table_c_eegnet_sm.csv'); base['seed'] = 42; base['ea'] = False
    a = pd.concat([base[['task_family', 'fold', 'model', 'seed', 'ea', 'accuracy', 'f1', 'macro_f1', 'balanced_accuracy']], s[['task_family', 'fold', 'model', 'seed', 'ea', 'accuracy', 'f1', 'macro_f1', 'balanced_accuracy']]])
    t1 = a.groupby(['task_family', 'model', 'ea', 'seed']).accuracy.mean().unstack('seed').round(4); t1['sd_over_seeds'] = t1[[c for c in t1.columns if c in (1, 2, 42)]].std(axis=1).round(4); t1.to_csv(O / 'v2_eegnet_seed_reproducibility.csv'); print(t1.to_string())
ss = B.parent / 'outputs_eegnet_groupkfold' / 'table_c_eegnet_seeds_subjects.csv'
if ss.exists() and u is not None:
    e = pd.read_csv(ss); e['acc'] = (e.tp + e.tn) / e.n; rows = []
    for t in TASKS:
        rim = P.loc[t]['sm16'] if t in P.index.get_level_values(0) else None
        if rim is None: continue
        for kind in ('compact', 'eegnet'):
            for ea, nm in ((True, 'EEGNet+EA(batch)'), (False, 'EEGNet no adaptation (seeds 1,2 mean)')):
                g = e[(e.task_family == t) & (e.model == kind) & (e.ea == ea)].groupby('subject').acc.mean(); j = pd.concat([rim.rename('rim'), g.rename('nn')], axis=1).dropna()
                if len(j) > 20: rows.append(dict(task=t, comparison=f'Riemannian sm16 batch - {kind} {nm}', **paired((j.rim - j.nn).values)))
    if rows: r = finish(rows, 'v2_model_family_comparisons.csv'); print(r.round(4).to_string())
