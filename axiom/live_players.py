"""Prospective, appearance-conditional NBA player point estimates and joint intervals.
Verify team membership using roster responses. Never infer active from missing injury data.
Withdraw known inactive players; do not automatically allocate extra usage/minutes.
"""
import datetime as dt,hashlib,io,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
from axiom.historical import HASHES,fetch,normalize,player_benchmark,REPO
from axiom.ingest import git_blob_sha
from axiom.live import schedule,get,ET,UTC,ALIASES
from axiom.simulation import player_draws
MODEL='alpha.7-opponent-conditional'
STATS=['Pts','REB','AST','3PM']
ROSTER='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}/roster'
INJURIES='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'
ESPN_ABBR={'GSW':'GS','NYK':'NY','NOP':'NO','SAS':'SA','UTA':'UTAH','WAS':'WSH','BKN':'BKN'}
FIELDS=['forecast_id','created_at_utc','tipoff_at_utc','game_id','model_version','player_id','player','team','opponent','expected_minutes','injury_status','data_as_of_utc']+[f'{k}_{v}' for k in STATS+['PRA'] for v in ['mean','p10','p90']]
def key(x):return str(int(x)) if str(x).replace('.0','').isdigit() else str(x)
def roster_players(obj):
    found={}
    def walk(v):
        if isinstance(v,list):
            for x in v:walk(x)
        elif isinstance(v,dict):
            if v.get('id') is not None and v.get('displayName') and isinstance(v.get('position'),dict):
                found[key(v['id'])]=v['displayName']
            for x in v.values():
                if isinstance(x,(list,dict)):walk(x)
    walk(obj.get('athletes',[]));return found
def injury_players(obj):
    rows={}
    for group in obj.get('injuries',[]):
        for i in group.get('injuries',[]):
            athlete=i.get('athlete',{});pid=athlete.get('id')
            if pid is not None and i.get('status'):rows[key(pid)]=str(i['status'])
    return rows
def current_boxes(season):
    api=f'https://api.github.com/repos/{REPO}';ref=get(api+'/git/ref/heads/main').json()['object']['sha'];tree=get(api+f'/git/trees/{ref}',params={'recursive':'1'}).json()['tree']
    path=f'nba/player_box/parquet/player_box_{season}.parquet';obj=next(x for x in tree if x['path']==path)
    url=f'https://raw.githubusercontent.com/{REPO}/{ref}/{path}';raw=get(url).content
    if git_blob_sha(raw)!=obj['sha']:raise ValueError('Player data hash mismatch')
    d=normalize(pd.read_parquet(io.BytesIO(raw)),season,'player_box')
    return d,dict(url=url,git_blob=obj['sha'],sha256=hashlib.sha256(raw).hexdigest())
def conditional_projection(h,o,l):
    r=np.asarray(h,dtype=float)[-30:];recent=r[-8:]
    if len(r)<5 or (r[:,0]<=0).any():raise ValueError('Insufficient prior appearances')
    mins=.6*recent[:,0].mean()+.4*r[:,0].mean()
    rate=mins*(.6*recent[:,1:].sum(0)/recent[:,0].sum()+.4*r[:,1:].sum(0)/r[:,0].sum())
    adjust=np.ones(4)
    if len(o)>=30 and len(l)>=200:
        o=np.asarray(o)[-200:];l=np.asarray(l)[-3000:];lr=l[:,1:].sum(0)/l[:,0].sum()
        opp=(o[:,1:].sum(0)+900*lr)/(o[:,0].sum()+900)
        adjust=np.clip(opp/np.maximum(lr,1e-6),.85,1.15)
    return float(mins),rate*adjust
def main():
    now=dt.datetime.now(UTC);today=pd.Timestamp(now.astimezone(ET).date());season=today.year+(today.month>=7)
    out=Path('reports/live_players');out.mkdir(parents=True,exist_ok=True);games=schedule(now);rows=[];skips=[];manifest=[]
    if not games:
        (out/'report.json').write_text(json.dumps({'status':'no_upcoming_games','n':0}));return
    old=pd.concat([normalize(fetch(y,'player_box',manifest),y,'player_box') for y in [2024,2025,2026]],ignore_index=True)
    current,source=current_boxes(season);current=current[current.Date<today];latest=current.Date.max()
    if pd.isna(latest) or (today-latest).days>3:raise ValueError('Current player data stale')
    cal_data=old[old.Season<=2025].copy();cal_data.loc[cal_data.Season==2025,'Season']=2026;_,cp=player_benchmark(cal_data)
    res=cp[[k+'_actual' for k in STATS]].to_numpy()-cp[[k+'_opponent' for k in STATS]].to_numpy()
    bins=np.digitize(cp.Pts_opponent,[5,12,20])
    history=pd.concat([old,current],ignore_index=True).sort_values('Date');history['pid']=history['Player ID'].map(key)
    h={pid:g[['Min',*STATS]].to_numpy(float)[-30:] for pid,g in history.groupby('pid')}
    o={t:g[['Min',*STATS]].to_numpy(float)[-200:] for t,g in history.groupby('Opp')}
    l=history[['Min',*STATS]].to_numpy(float)[-3000:];recent=current.sort_values('Date').assign(pid=lambda x:x['Player ID'].map(key)).drop_duplicates('pid',keep='last').set_index('pid')
    try:
        ir=get(INJURIES);inj=injury_players(ir.json());inj_note='ESPN secondary injury feed; absent entries are unverified';inj_hash=hashlib.sha256(ir.content).hexdigest()
    except (requests.RequestException,ValueError,TypeError):
        inj={};inj_note='Secondary injury source unavailable';inj_hash=None
    rosters={};roster_refs={}
    for t in sorted({g[s] for g in games for s in ['home','away']}):
        url=ROSTER.format(team=ESPN_ABBR.get(t,t).lower())
        try:
            rr=get(url);players=roster_players(rr.json())
            if not players:raise ValueError('Empty roster')
            rosters[t]=players;roster_refs[t]={'url':url,'sha256':hashlib.sha256(rr.content).hexdigest()}
        except (requests.RequestException,ValueError,TypeError) as e:skips.append({'team':t,'reason':f'Roster unavailable: {type(e).__name__}'})
    ledger=Path('tracking/player_predictions.csv');previous=pd.read_csv(ledger,dtype={'player_id':str,'game_id':str}) if ledger.exists() else pd.DataFrame(columns=FIELDS);seen=set(previous.forecast_id);new=[]
    for g in games:
        for t,opp in [(g['home'],g['away']),(g['away'],g['home'])]:
            for pid,name in rosters.get(t,{}).items():
                if pid not in h or pid not in recent.index or recent.loc[pid,'Team']!=t or (today-recent.loc[pid,'Date']).days>21:continue
                if len(h[pid])<5:continue
                status=inj.get(pid,'UNVERIFIED')
                if status.lower() in ('out','suspension','suspended'):skips.append({'game_id':g['game_id'],'player_id':pid,'reason':f'Secondary source: {status}'});continue
                mi,pr=conditional_projection(h[pid],o.get(opp,[]),l)
                pool=res[bins==np.digitize(pr[0],[5,12,20])]
                if len(pool)<30:pool=res
                draws=player_draws(pr,pool,seed=42);ts=dt.datetime.now(UTC)
                if ts>=g['tipoff']:continue
                fid=f'{g["game_id"]}:{pid}:{MODEL}'
                r=dict(forecast_id=fid,created_at_utc=ts.isoformat(),tipoff_at_utc=g['tipoff'].isoformat(),game_id=g['game_id'],model_version=MODEL,player_id=pid,player=name,team=t,opponent=opp,expected_minutes=mi,injury_status=status,data_as_of_utc=now.isoformat())
                for j,k in enumerate(STATS+['PRA']):
                    r[k+'_mean']=float((list(pr)+[pr[:3].sum()])[j]);r[k+'_p10'],r[k+'_p90']=map(float,np.quantile(draws[:,j],[.1,.9]))
                rows.append(r)
                if fid not in seen:new.append(r);seen.add(fid)
    if new:pd.concat([previous,pd.DataFrame(new,columns=FIELDS)],ignore_index=True).to_csv(ledger,index=False)
    summary={'n':len(rows),'new':len(new),'known_rosters':len(rosters),'skipped':len(skips),'latest_box_score_date':str(latest.date()),'source':source,'rosters':roster_refs,'injury_source':{'url':INJURIES,'sha256':inj_hash,'note':inj_note},'historical_manifest':manifest,'warning':'Predictions conditional on appearance; roster membership does not establish availability; no injury minutes redistribution'}
    (out/'report.json').write_text(json.dumps(summary,indent=2));pd.DataFrame(rows,columns=FIELDS).to_csv(out/'predictions.csv',index=False);(out/'skipped.json').write_text(json.dumps(skips,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
