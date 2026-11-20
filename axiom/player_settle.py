"""Settle appearance-conditional player forecasts using published current player boxes.
Absent rows are explicitly not observed, not zero-point predictions.
"""
import datetime as dt,json
from pathlib import Path
import numpy as np
import pandas as pd
from axiom.live_players import current_boxes,key,STATS
from axiom.live import current_teams,ET,UTC
def evaluate(f,a,confirmed_games):
    f=f.copy();a=a.copy();a['player_id']=a['Player ID'].map(key);a['game_id']='espn:'+a['game_id'].map(key)
    if f.forecast_id.duplicated().any() or a.duplicated(['game_id','player_id']).any():raise ValueError('Duplicate forecasts/actual appearances')
    if f.empty:return dict(registered=0,observed_appearances=0,not_observed=0,pending=0,by_stat={})
    if (pd.to_datetime(f.created_at_utc,utc=True)>=pd.to_datetime(f.tipoff_at_utc,utc=True)).any():raise ValueError('Late forecast')
    a=a[['game_id','player_id','Team',*STATS]].rename(columns={'Team':'actual_team',**{k:k+'_actual' for k in STATS}})
    m=f.merge(a,on=['game_id','player_id'],how='left',validate='many_to_one');ok=m.game_id.isin(confirmed_games)&m.actual_team.eq(m.team);m['PRA_actual']=m[['Pts_actual','REB_actual','AST_actual']].sum(axis=1,min_count=3)
    obs=m[ok];metrics={}
    if len(obs):
        for k in STATS+['PRA']:
            metrics[k]=dict(mae=float(np.abs(obs[k+'_mean']-obs[k+'_actual']).mean()),interval_coverage=float(obs[k+'_actual'].between(obs[k+'_p10'],obs[k+'_p90']).mean()))
    return dict(registered=len(f),observed_appearances=len(obs),not_observed=int(m.game_id.isin(confirmed_games).sum()-len(obs)),pending=int((~m.game_id.isin(confirmed_games)).sum()),by_stat=metrics,note='Not observed does not establish DNP; team final and published player data required')
def main():
    path=Path('tracking/player_predictions.csv');f=pd.read_csv(path,dtype={'game_id':str,'player_id':str});out=Path('reports/live_players');out.mkdir(parents=True,exist_ok=True)
    today=pd.Timestamp(dt.datetime.now(UTC).astimezone(ET).date());season=today.year+(today.month>=7)
    p,src=current_boxes(season);t=current_teams(season,[]);ids=set('espn:'+t.game_id.map(key));m=evaluate(f,p,ids);m['player_data_source']=src
    Path('tracking/player_performance.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
if __name__=='__main__':main()
