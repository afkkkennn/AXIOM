"""Score the immutable forecast log against independently recorded final results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss

def paired_injury_metrics(m):
    versions=['alpha.7-matched-control-T90','alpha.7-injury-shadow-T90']
    part=m[m.model_version.isin(versions)]
    if part.duplicated(['game_id','model_version']).any():raise ValueError('Duplicate paired-model forecasts')
    groups=[part[part.model_version==v].set_index('game_id') for v in versions]
    ids=groups[0].index.intersection(groups[1].index)
    if not len(ids):return {'matched_games':0,'status':'awaiting paired actual results'}
    a,b=[g.loc[ids] for g in groups]
    y=a.home_score.gt(a.away_score).to_numpy(int)
    pa=a.home_probability.to_numpy(float);pb=b.home_probability.to_numpy(float)
    error=lambda p:-(y*np.log(p)+(1-y)*np.log1p(-p))
    diff=error(pb)-error(pa)
    output={'matched_games':len(ids),'control_accuracy':float(((pa>=.5)==y).mean()),'injury_accuracy':float(((pb>=.5)==y).mean()),'control_brier':float(brier_score_loss(y,pa)),'injury_brier':float(brier_score_loss(y,pb)),'control_log_loss':float(error(pa).mean()),'injury_log_loss':float(error(pb).mean()),'delta_log_loss_injury_minus_control':float(diff.mean())}
    day=pd.to_datetime(a.tipoff_at_utc,utc=True).dt.strftime('%Y-%m-%d')
    blocks=pd.DataFrame({'day':day.to_numpy(),'delta':diff}).groupby('day').delta.agg(['sum','count']).to_numpy()
    if len(blocks)>=20:
        rng=np.random.default_rng(71);sample=blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].sum(1)
        output['date_cluster_bootstrap_ci95']=np.quantile(sample[:,0]/sample[:,1],[.025,.975]).tolist()
    output['status']='preliminary' if len(ids)<100 else 'accumulating prospective evidence'
    return output

def evaluate(forecasts,actual):
    if forecasts.forecast_id.duplicated().any() or actual.game_id.duplicated().any():raise ValueError('Duplicate forecast or result IDs')
    m=forecasts.merge(actual,on='game_id',validate='many_to_one');summary={}
    for version,g in m.groupby('model_version'):
        y=(g.home_score>g.away_score).astype(int);p=g.home_probability.astype(float)
        if not p.between(0,1,inclusive='neither').all():raise ValueError('Invalid probabilities')
        margin=g.home_score-g.away_score;total=g.home_score+g.away_score
        summary[version]={'forecast_count':len(g),'unique_games':g.game_id.nunique(),'accuracy':float(((p>=.5)==y).mean()),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'margin_mae':float(np.abs(g.projected_margin-margin).mean()),'total_mae':float(np.abs(g.projected_total-total).mean())}
    return {'settled_forecasts':len(m),'unsettled_forecasts':len(forecasts)-len(m),'by_model':summary,'paired_injury_comparison':paired_injury_metrics(m)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forecasts',default='tracking/predictions.csv');p.add_argument('--actual',default='tracking/actual_results.csv');p.add_argument('--out',default='tracking/performance.json');a=p.parse_args();r=evaluate(pd.read_csv(a.forecasts,dtype={'game_id':str}),pd.read_csv(a.actual,dtype={'game_id':str}));Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
