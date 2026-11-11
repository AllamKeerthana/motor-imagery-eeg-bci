"""Recompute reported p-values from saved per-fold/per-subject outputs and compare with the saved statistics files."""
import numpy as np, pandas as pd, glob
from scipy import stats
B='outputs_rigorous_benchmark/'; O='outputs_final/'
def close(a,b): return bool(np.isclose(a,b,atol=1e-4))
res=[]
# nested vs baselines (fold-level paired t-test, n=5)
s=pd.read_csv(O+'stats_nested_vs_baselines.csv'); g=pd.read_csv(B+'table_g_nested_selected.csv'); e=pd.read_csv(B+'table_e_channel_sweep.csv')
for _,r in s.iterrows():
    a=g[g.task_family==r.task].set_index('fold').accuracy; k=16 if r.vs.startswith('t-test') else 64
    b=e[(e.task_family==r.task)&(e.k==k)].set_index('fold').accuracy
    res.append(('nested vs '+r.vs, r.task, 'fold-level paired t (n=%d)'%len(a), r.p_raw, stats.ttest_rel(a,b).pvalue))
# sweep 16 vs 64
s=pd.read_csv(O+'stats_sweep_16_vs_64.csv')
for _,r in s.iterrows():
    w=e[e.task_family==r.task].pivot(index='fold',columns='k',values='accuracy'); res.append(('sweep 16 vs 64', r.task,'fold-level paired t (n=5)', r.p_raw, stats.ttest_rel(w[16],w[64]).pvalue))
# EEGNet 64 vs 16
s=pd.read_csv(O+'stats_eegnet_64_vs_16.csv'); c=pd.read_csv('outputs_eegnet_groupkfold/table_c_eegnet_gpu.csv'); c=c[(c.model=='eegnet')&(c.ea==False)]
for _,r in s.iterrows():
    w=c[c.task_family==r.task].pivot(index='fold',columns='n_channels',values='accuracy'); res.append(('EEGNet 16 vs 64', r.task,'fold-level paired t (n=5)', r.p_raw, stats.ttest_rel(w[16],w[64]).pvalue))
# few-shot (subject level Wilcoxon)
s=pd.read_csv(O+'stats_fewshot.csv'); f=pd.read_csv(B+'table_f_fewshot.csv'); need=f.groupby(['task_family','fold','subject']).n_per_class.nunique(); keep=need[need==4].index
f=f.set_index(['task_family','fold','subject']).loc[keep].reset_index()
for _,r in s.iterrows():
    w=f[f.task_family==r.task].pivot_table(index=['fold','subject'],columns='n_per_class',values='accuracy'); d=(w[10]-w[0]).values; res.append(('few-shot 10 vs 0', r.task,'subject-level Wilcoxon (n=%d)'%len(d), r.p_raw, stats.wilcoxon(d).pvalue))
# adaptive channels
s=pd.read_csv(O+'stats_adaptive_channels.csv'); a=pd.concat([pd.read_csv(x) for x in glob.glob(B+'table_i_adaptive_channels_*.csv')])
for _,r in s.iterrows():
    w=a[a.task_family==r.task].pivot_table(index=['fold','subject'],columns='arm',values='accuracy').dropna(); d=(w[r.arm]-w['sm']).values
    res.append(('adaptive '+r.arm+' vs sm', r.task,'subject-level Wilcoxon (n=%d)'%len(d), r.p_raw, stats.wilcoxon(d).pvalue if np.any(d!=0) else 1.0))
out=pd.DataFrame(res,columns=['comparison','task','test_unit','reported_p','recomputed_p']); out['reproduced']=[close(x,y) for x,y in zip(out.reported_p,out.recomputed_p)]
out.round(5).to_csv(O+'pvalue_reproduction_check.csv',index=False); print(out.reproduced.value_counts().to_string()); print(out[~out.reproduced].to_string())
