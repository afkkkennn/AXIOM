"""Player projections conditional on appearance. Same-day information is frozen."""
import numpy as np
import pandas as pd

STATS=['Pts','REB','AST','3PM']

def project(raw,test_start):
    r=raw.copy();r['date']=pd.to_datetime(r.Date).dt.normalize();r=r[r.Min>0].sort_values('date');hist={};rows=[]
    for dt,g in r.groupby('date',sort=True):
        for _,x in g.iterrows():
            h=hist.get(x['Player ID'],[])
            if len(h)<3 or dt<pd.Timestamp(test_start):continue
            a=np.array(h);recent=a[-5:];minutes=.6*recent[:,0].mean()+.4*a[:,0].mean();base=a[:,1:].mean(axis=0);rate=minutes*(.55*recent[:,1:].sum(axis=0)/recent[:,0].sum()+.45*a[:,1:].sum(axis=0)/a[:,0].sum());actual=x[STATS].to_numpy(float)
            row={'date':dt,'player_id':x['Player ID'],'actual_minutes':x.Min,'baseline_minutes':a[:,0].mean(),'projected_minutes':minutes}
            for k,b,p,y in zip(STATS,base,rate,actual):row.update({k+'_baseline':b,k+'_rate':p,k+'_actual':y})
            row.update(PRA_baseline=sum(base[:3]),PRA_rate=sum(rate[:3]),PRA_actual=sum(actual[:3])); rows.append(row)
        for _,x in g.iterrows():hist.setdefault(x['Player ID'],[]).append(x[['Min',*STATS]].to_numpy(float))
    pred=pd.DataFrame(rows)
    if pred.empty:raise ValueError('Insufficient player history')
    metrics={k:{m:float(np.abs(pred[k+'_'+m]-pred[k+'_actual']).mean()) for m in ['baseline','rate']} for k in STATS+['PRA']}
    metrics['minutes']={m:float(np.abs(pred[m+'_minutes']-pred.actual_minutes).mean()) for m in ['baseline','projected']};metrics['n']=len(pred)
    return metrics,pred
