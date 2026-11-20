"""Optional market benchmark; only timestamped, paired pregame quotes accepted."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

def compare(pred,quotes):
    q=quotes.copy();q['date']=pd.to_datetime(q.date).dt.normalize()
    observed=pd.to_datetime(q.observed_at,utc=True,errors='raise');tip=pd.to_datetime(q.tipoff_at,utc=True,errors='raise')
    if observed.isna().any() or tip.isna().any() or (observed>=tip).any():raise ValueError('Market quote is not demonstrably pregame')
    if (q[['home_decimal','away_decimal']]<=1).any().any():raise ValueError('Invalid decimal odds')
    q['market_prob']=(1/q.home_decimal)/(1/q.home_decimal+1/q.away_decimal);q['observed_at']=observed
    q=q.sort_values('observed_at').drop_duplicates(['date','home','away'],keep='last')
    m=pred.merge(q,on=['date','home','away'],validate='one_to_one');y=m.margin.gt(0).astype(int)
    if m.empty:raise ValueError('No matched quotes')
    metrics={}
    for k in [c for c in m if c.endswith('_prob')]:
        p=np.clip(m[k],1e-6,1-1e-6);metrics[k]={'n':len(m),'accuracy':float(((p>=.5)==y).mean()),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1]))}
    return metrics
