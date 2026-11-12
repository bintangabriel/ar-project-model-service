"""python train.py "Data_Train.xlsx"
Validates the model (5-fold, customers kept together), prints + saves the score report (metrics_report.txt, metrics_calibration.csv),
then trains on all data and saves model.joblib."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from reco import *

def top1(df, s):
    """customers with a paying and a non-paying REMO: how often the top-scored REMO is one who got paid (ties -> expected value) vs a random pick"""
    t=df.assign(s=np.round(s,10)); hit=[]; rnd=[]
    for _,q in t.groupby('NO_AGGR'):
        if q.y.nunique()==2: hit.append(q[q.s==q.s.max()].y.mean()); rnd.append(q.y.mean())
    return np.mean(hit), np.mean(rnd), len(hit)

d=load(sys.argv[1]); X=prep(d); y=d.y.values; g=d.NO_AGGR.values
oof=np.zeros(len(d))
for a,b in GroupKFold(5).split(X,y,g): oof[b]=Ens().fit(X.iloc[a],y[a]).predict_proba(X.iloc[b])[:,1]

auc=roc_auc_score(y,oof); brier=brier_score_loss(y,oof); base=brier_score_loss(y,np.full(len(y),y.mean()))
pw,npairs=pairwise(d,oof); hit,rnd,nc=top1(d,oof)
cal=pd.DataFrame({'pred':oof,'actual':y}).groupby(pd.qcut(oof,5,duplicates='drop'),observed=True).agg(avg_predicted=('pred','mean'),actual_paid=('actual','mean'),n=('actual','size')).round(3).reset_index(drop=True)
cal.index=[f'group {i+1}' for i in range(len(cal))]

rep=f"""MODEL SCORE REPORT  (5-fold validation, all rows of a customer kept in the same fold)
Data: {len(d)} assignments, {d.NO_AGGR.nunique()} customers, paid rate {y.mean():.1%}

1) Does it tell payers from non-payers?
   AUC            : {auc:.3f}   (0.5 = coin flip, 1.0 = perfect)
   Brier score    : {brier:.3f}   (lower is better; always guessing the average = {base:.3f})

2) Are the percentages trustworthy?  (cases sorted by predicted chance, cut into 5 equal groups)
{cal.to_string()}

3) Does it pick a good REMO?  (only customers who had both a paying and a non-paying REMO)
   Paying REMO ranked above a non-paying one : {pw:.1%}   (50% = chance, {npairs} pairs)
   Top-scored REMO actually got paid         : {hit:.1%}   vs random pick {rnd:.1%}   ({nc} customers)
"""
print(rep); open('metrics_report.txt','w').write(rep); cal.to_csv('metrics_calibration.csv')
save(Ens().fit(X,y),d); print('saved model.joblib')
