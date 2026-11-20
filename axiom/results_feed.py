"""Verify logged ESPN game IDs against completed scoreboard events.
No retrospective forecasts are generated. Source is public ESPN, not official NBA.
"""
import argparse,datetime as dt,re
from pathlib import Path
import pandas as pd
import requests
from axiom.settle import evaluate
import json
API='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary'
ALIASES={'GS':'GSW','NY':'NYK','NO':'NOP','SA':'SAS','UTAH':'UTA','WSH':'WAS','BK':'BKN','BRK':'BKN','PHO':'PHX'}
def canonical(t):
    t=str(t).strip().upper();return ALIASES.get(t,t)
def extract(event,game_id,home,away):
    header=event.get('header',{})
    if str(header.get('id'))!=str(game_id):raise ValueError('Unexpected game ID')
    c=header['competitions'][0]
    s=c['status']['type']
    if not s.get('completed') or s.get('name') not in ('STATUS_FINAL','STATUS_FINAL_OVERTIME'):return None
    teams={x['homeAway']:x for x in c['competitors']}
    for side,t in [('home',home),('away',away)]:
        if canonical(teams[side]['team']['abbreviation'])!=canonical(t):raise ValueError('Game team mismatch')
    h,a=int(teams['home']['score']),int(teams['away']['score'])
    if h==a:raise ValueError('Invalid final score')
    return h,a
def settle(ledger,actual,out):
    f=pd.read_csv(ledger,dtype={'game_id':str});a=pd.read_csv(actual,dtype={'game_id':str});rows=[]
    if f.forecast_id.duplicated().any() or a.game_id.duplicated().any():raise ValueError('Duplicate IDs')
    now=dt.datetime.now(dt.timezone.utc)
    for gid,g in f.groupby('game_id'):
        if gid in set(a.game_id):continue
        if not re.fullmatch(r'espn:\d+',gid):raise ValueError('game_id must use espn:<event id>')
        if g[['home','away','tipoff_at_utc']].drop_duplicates().shape[0]!=1:raise ValueError('Inconsistent forecast metadata')
        row=g.iloc[0];tip=dt.datetime.fromisoformat(row.tipoff_at_utc.replace('Z','+00:00'))
        if tip>=now:continue
        created=pd.to_datetime(g.created_at_utc,utc=True,errors='raise')
        if (created>=tip).any():raise ValueError('Late forecast in ledger')
        eid=gid.split(':')[1];r=requests.get(API,params={'event':eid},timeout=(15,45));r.raise_for_status()
        scores=extract(r.json(),eid,row.home,row.away)
        if scores is None:continue
        rows.append(dict(game_id=gid,home_score=scores[0],away_score=scores[1],source_url=r.url,verified_at_utc=now.isoformat()))
    if rows:a=pd.concat([a,pd.DataFrame(rows)],ignore_index=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a.to_csv(out/'actual_results.csv',index=False);m=evaluate(f,a)
    (out/'performance.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ledger',default='tracking/predictions.csv');p.add_argument('--actual',default='tracking/actual_results.csv');p.add_argument('--out',default='reports/live');a=p.parse_args();settle(a.ledger,a.actual,a.out)
