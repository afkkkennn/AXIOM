"""Score the immutable forecast log against independently recorded final results."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss

def evaluate(forecasts,actual):
    if forecasts.forecast_id.duplicated().any() or actual.game_id.duplicated().any():raise ValueError('Duplicate forecast or result IDs')
    m=forecasts.merge(actual,on='game_id',validate='many_to_one');summary={}
    for version,g in m.groupby('model_version'):
        y=(g.home_score>g.away_score).astype(int);p=g.home_probability.astype(float)
        if not p.between(0,1,inclusive='neither').all():raise ValueError('Invalid probabilities')
        margin=g.home_score-g.away_score;total=g.home_score+g.away_score
        summary[version]={'forecast_count':len(g),'unique_games':g.game_id.nunique(),'accuracy':float(((p>=.5)==y).mean()),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'margin_mae':float(np.abs(g.projected_margin-margin).mean()),'total_mae':float(np.abs(g.projected_total-total).mean())}
    return {'settled_forecasts':len(m),'unsettled_forecasts':len(forecasts)-len(m),'by_model':summary}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forecasts',default='tracking/predictions.csv');p.add_argument('--actual',default='tracking/actual_results.csv');p.add_argument('--out',default='tracking/performance.json');a=p.parse_args();r=evaluate(pd.read_csv(a.forecasts,dtype={'game_id':str}),pd.read_csv(a.actual,dtype={'game_id':str}));Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
