"""Live team forecasts using frozen alpha.6 specification.
Same-day results deliberately excluded, no sports market inputs.
Write only future forecasts to the append-only ledger.
"""
import argparse,datetime as dt,hashlib,io,json,os
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
from scipy.special import expit,logit
from sklearn.linear_model import Ridge,LogisticRegression
from axiom.historical import fetch,normalize,team_features,HASHES,MODELS,REPO,TEAMS,ALIASES
from axiom.ingest import git_blob_sha
from axiom.simulation import score_draws,summarize_scores
from axiom.ledger import append

ET=ZoneInfo('America/New_York')
UTC=dt.timezone.utc
SCOREBOARD='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
MODEL='alpha.6-calibrated-live'
def get(url,**kw):
    r=requests.get(url,timeout=(15,90),**kw);r.raise_for_status();return r
def current_teams(y,manifest):
    api=f'https://api.github.com/repos/{REPO}'
    ref=get(api+'/git/ref/heads/main').json()['object']['sha']
    path=f'nba/team_box/parquet/team_box_{y}.parquet'
    tree=get(api+f'/git/trees/{ref}',params={'recursive':'1'}).json()
    entry=next(x for x in tree['tree'] if x['path']==path)
    url=f'https://raw.githubusercontent.com/{REPO}/{ref}/{path}';raw=get(url).content
    if git_blob_sha(raw)!=entry['sha']:raise ValueError('Current data hash mismatch')
    manifest.append(dict(url=url,git_blob=entry['sha'],sha256=hashlib.sha256(raw).hexdigest(),retrieved_at_utc=dt.datetime.now(UTC).isoformat()))
    return normalize(pd.read_parquet(io.BytesIO(raw)),y,'team_box')
def schedule(now,hours=36):
    games={}
    for offset in range(3):
        date=(now.astimezone(ET).date()+dt.timedelta(days=offset))
        r=get(SCOREBOARD,params={'dates':date.strftime('%Y%m%d'),'limit':100})
        for e in r.json().get('events',[]):
            c=e['competitions'][0];s=c['status']['type'];tip=dt.datetime.fromisoformat(e['date'].replace('Z','+00:00'))
            if s.get('state')!='pre' or s.get('name')!='STATUS_SCHEDULED' or not now<tip<=now+dt.timedelta(hours=hours):continue
            if e.get('season',{}).get('type',2)!=2:continue
            teams={q['homeAway']:ALIASES.get(q['team']['abbreviation'],q['team']['abbreviation']) for q in c['competitors']}
            if set(teams.values())<=TEAMS and len(set(teams.values()))==2:
                games[str(e['id'])]=dict(game_id='espn:'+str(e['id']),date=pd.Timestamp(tip.astimezone(ET).date()),tipoff=tip,home=teams['home'],away=teams['away'])
    return sorted(games.values(),key=lambda g:g['tipoff'])
def dummy_pair(g,season):
    # Only generates pregames for this date; never append dummy rows to actual history.
    rows=[]
    for home,away,flag in [(g['home'],g['away'],'vs.'),(g['away'],g['home'],'@')]:
        rows.append(dict(Date=g['date'],Season=season,Team=home,Opp=away,**{'Match up':f'{home} {flag} {away}'},Pts=101 if flag=='vs.' else 100,FGM=40,FGA=90,FTM=15,FTA=20,OREB=10,DREB=30,TOV=12,**{'3PM':10}))
    return rows
def train_models(f):
    tr=f[f.season<=2024];va=f[f.season==2025]
    if min(len(tr),len(va))<500:raise ValueError('Historical training data incomplete')
    cols=MODELS['ensemble'];m=Ridge(alpha=100).fit(tr[cols],tr.margin)
    sigma=float(np.sqrt(np.mean((tr.margin-m.predict(tr[cols]))**2)))
    vmargin=m.predict(va[cols]);vp=expit(vmargin/(.6*sigma))
    cal=LogisticRegression().fit(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1),va.margin.gt(0))
    tm=Ridge(alpha=100).fit(tr[['rolling_total']],tr.total)
    res=np.column_stack([va.margin.to_numpy()-vmargin,va.total.to_numpy()-tm.predict(va[['rolling_total']])])
    return m,sigma,cal,tm,res
def main():
    p=argparse.ArgumentParser();p.add_argument('--ledger',default='tracking/predictions.csv');p.add_argument('--out',default='reports/live_forecast');p.add_argument('--record',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);now=dt.datetime.now(UTC);today=pd.Timestamp(now.astimezone(ET).date());season=today.year+(today.month>=7)
    games=schedule(now);manifest=[];published=[];skipped=[]
    if not games:
        (out/'forecasts.json').write_text(json.dumps({'status':'no_upcoming_regular_season_games','generated_at':now.isoformat(),'forecasts':[]},indent=2));print('No eligible games');return
    hist=pd.concat([normalize(fetch(y,'team_box',manifest),y,'team_box') for y in HASHES],ignore_index=True)
    f,_=team_features(hist);m,s,cal,tm,res=train_models(f)
    current=current_teams(season,manifest);current=current[current.Date<today]
    latest=current.Date.max()
    if pd.isna(latest) or (today-latest).days>3:raise ValueError(f'Current team data stale or missing: {latest}')
    all_actual=pd.concat([hist,current],ignore_index=True)
    existing=pd.read_csv(a.ledger,dtype={'game_id':str}) if Path(a.ledger).exists() else pd.DataFrame(columns=['forecast_id'])
    ids=set(existing.forecast_id)
    for date in sorted(set(g['date'] for g in games)):
        valid=[]
        for g in [x for x in games if x['date']==date]:
            h=current[(current.Team==g['home'])];aw=current[(current.Team==g['away'])]
            if min(len(h),len(aw))<3 or max((today-h.Date.max()).days,(today-aw.Date.max()).days)>8:
                skipped.append({'game_id':g['game_id'],'reason':'insufficient recent team data'});continue
            valid.append(g)
        if not valid:continue
        placeholder=pd.DataFrame([r for g in valid for r in dummy_pair(g,season)])
        f,_=team_features(pd.concat([all_actual[all_actual.Date<date],placeholder],ignore_index=True))
        x=f[f.date==date].copy()
        if len(x)!=len(valid):raise ValueError('Unexpected pregame feature rows')
        margin=m.predict(x[MODELS['ensemble']]);praw=expit(margin/(.6*s));prob=cal.predict_proba(logit(np.clip(praw,1e-6,1-1e-6)).reshape(-1,1))[:,1];total=tm.predict(x[['rolling_total']])
        for i,row in enumerate(x.itertuples()):
            g=next(z for z in valid if z['home']==row.home and z['away']==row.away)
            ts=dt.datetime.now(UTC)
            if ts>=g['tipoff']:skipped.append({'game_id':g['game_id'],'reason':'tipoff passed during calculation'});continue
            fid=g['game_id']+':'+MODEL
            notes='Team-only baseline; official injury/lineup data unverified; all same-day scores excluded; may omit recent results'
            record=dict(forecast_id=fid,tipoff_at_utc=g['tipoff'].isoformat(),game_id=g['game_id'],model_version=MODEL,home=row.home,away=row.away,home_probability=float(prob[i]),projected_margin=float(margin[i]),projected_total=float(total[i]),data_as_of_utc=now.isoformat(),notes=notes)
            summary=summarize_scores(score_draws(margin[i],total[i],prob[i],res,seed=42))
            published.append({**record,**summary,'latest_box_score_date':str(latest.date()),'recorded_at_utc':ts.isoformat() if a.record and fid not in ids else None,'already_recorded':fid in ids})
            if a.record and fid not in ids:append(a.ledger,record,ts);ids.add(fid)
    report={'status':'ok','generated_at_utc':now.isoformat(),'source_manifest':manifest,'forecasts':published,'skipped':skipped,'warning':'No validated injury adjustments. Do not label these high confidence solely from win probability.'}
    (out/'forecasts.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'ok','latest_box_score_date':str(latest.date()),'forecasts':published,'skipped':skipped},indent=2))
if __name__=='__main__':main()
