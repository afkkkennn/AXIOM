"""Prospective injury challenger. Frozen specifications, no use of target-game outcomes.
Only issue paired T-90 eligible forecasts when both report snapshots exist.
The archive can lag. A skipped game is not a healthy roster.
"""
import datetime as dt,json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import Ridge,LogisticRegression
from axiom.historical import HASHES,fetch,normalize,team_features,MODELS
from axiom.live import UTC,ET,schedule,current_teams,dummy_pair
from axiom.official_injuries import source,eastern,index_reports,snapshot,player_history,build_features,STATUS,EXTRA
from axiom.ledger import append
MODEL='alpha.7-injury-shadow-T90'
CONTROL='alpha.7-matched-control-T90'
OUT=Path('reports/injury_shadow')
def eligible_snapshots(index,g,now):
    tip=pd.Timestamp(g['tipoff']);cut=tip-pd.Timedelta(minutes=90)
    # Wait until the historical cutoff, but never issue after tipoff.
    if pd.Timestamp(now)<cut or pd.Timestamp(now)>=tip:return None
    s=[snapshot(index,g['date'],g['home'],g['away'],t,cut) for t in [g['home'],g['away']]]
    if any(x is None or x.empty for x in s):return None
    # Additional live freshness gate: reports must be no older than 8 hours at cutoff.
    if any(cut-x.report.max()>pd.Timedelta(hours=8) for x in s):return None
    return s
def side_vector(players,team,date,report,season_start):
    """Same past-appearance and box-score proxy as historical specification.
    Uses complete prior game dates only; no future game player records required.
    """
    previous=players[players.Date<date]
    current=previous[(previous.Team==team)&(previous.Date>=season_start)]
    days=sorted(current.Date.unique())[-10:]
    if len(days)<5:return None
    prior=current[current.Date.isin(days)]
    if report.empty:return None
    result=np.zeros(4);known=0
    listed=report.dropna(subset=['person_id','current_status']).drop_duplicates('person_id')
    for r in listed.itertuples():
        status=str(r.current_status).lower()
        if status not in STATUS:continue
        pid=int(r.person_id)
        recent=previous[previous.player_id.astype(int)==pid].sort_values('Date').tail(25)
        if len(recent)<3 or recent.iloc[-1].Team!=team:continue
        mins=prior[prior.player_id.astype(int)==pid].set_index('Date')['min'].reindex(days,fill_value=0).to_numpy()
        played=mins>0
        if not played.any():continue
        typical=mins[played].mean();baseline=played.mean();dm=(STATUS[status]-baseline)*typical
        denominator=recent['min'].sum()+1e-6
        rate=recent.pts.sum()/denominator
        quality=(recent.pts+.7*recent.ast+.25*recent.reb).sum()/denominator
        result+=np.array([dm,dm*rate,dm*quality,1.]);known+=1
    # A report with named players but zero identity/prior matches is not sufficient.
    if len(listed)>0 and known==0:return None
    return result
def train(history,inj):
    d=history.merge(inj,on=['date','home','away'],validate='one_to_one')
    tr=d[d.season<=2024];va=d[d.season==2025]
    if min(len(tr),len(va))<500:raise ValueError('Insufficient injury training data')
    models={}
    for name,cols in [(CONTROL,MODELS['ensemble']),(MODEL,MODELS['ensemble']+EXTRA)]:
        m=Ridge(alpha=100).fit(tr[cols],tr.margin)
        sd=float(np.sqrt(np.mean((tr.margin-m.predict(tr[cols]))**2)))
        vp=expit(m.predict(va[cols])/(.6*sd))
        cal=LogisticRegression().fit(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1),va.margin.gt(0))
        total=Ridge(alpha=100).fit(tr[['rolling_total']],tr.total)
        models[name]=(cols,m,sd,cal,total)
    return models
def main():
    OUT.mkdir(parents=True,exist_ok=True);now=dt.datetime.now(UTC);today=pd.Timestamp(now.astimezone(ET).date());season=today.year+int(today.month>=7)
    games=schedule(now);manifest=[];skip=[];candidates=[]
    def finish(status,rows):
        report={'status':status,'generated_at_utc':now.isoformat(),'new_forecasts':len(rows),'forecasts':rows,'skips':skip,'manifest':manifest,'warning':'Shadow only; not promoted. No missing-report inference. Both model forecasts use identical as-of inputs except injury features.'}
        (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
    upcoming=[g for g in games if 0<(g['tipoff']-now).total_seconds()<=90*60]
    if not upcoming:return finish('no_games_in_T90_window',[])
    reports=source(season,'injury_report',manifest,dynamic=True);idx=index_reports(reports)
    for g in upcoming:
        s=eligible_snapshots(idx,g,now)
        if s is None:skip.append({'game_id':g['game_id'],'reason':'No fresh submitted T-90 snapshots for both teams'})
        else:candidates.append((g,s))
    if not candidates:return finish('no_eligible_report_pairs',[])
    ledger=Path('tracking/predictions.csv');old=pd.read_csv(ledger);seen=set(old.forecast_id)
    candidates=[(g,s) for g,s in candidates if not any(g['game_id']+':'+m in seen for m in [MODEL,CONTROL])]
    if not candidates:return finish('already_recorded',[])
    ts=[];rr=[];ps=[];tips={}
    for y in HASHES:
        raw=fetch(y,'team_box',manifest);ts.append(normalize(raw,y,'team_box'))
        for gid,tip in zip(raw.game_id,eastern(raw.game_date_time)):tips[str(gid)]=tip
        rr.append(source(y,'injury_report',manifest));ps.append(player_history(source(y,'player_game_logs',manifest)))
    historical=pd.concat(ts,ignore_index=True);f,_=team_features(historical)
    inj,_,_=build_features(historical,tips,pd.concat(ps,ignore_index=True),pd.concat(rr,ignore_index=True));models=train(f,inj)
    teams=current_teams(season,manifest);teams=teams[teams.Date<today]
    if teams.empty or (today-teams.Date.max()).days>3:raise ValueError('Current team boxes stale')
    current=player_history(source(season,'player_game_logs',manifest,dynamic=True))
    current=current[current.Date<today]
    if current.empty or (today-current.Date.max()).days>3:raise ValueError('Current player logs stale')
    player=pd.concat([ps[-1],current],ignore_index=True).sort_values('Date');hist=pd.concat([historical,teams],ignore_index=True)
    season_start=pd.Timestamp(season-1,7,1);recorded=[]
    for g,s in candidates:
        sides=[side_vector(player,t,g['date'],rep,season_start) for t,rep in zip([g['home'],g['away']],s)]
        if any(v is None for v in sides):skip.append({'game_id':g['game_id'],'reason':'Insufficient matched player history'});continue
        feat,_=team_features(pd.concat([hist[hist.Date<g['date']],pd.DataFrame(dummy_pair(g,season))],ignore_index=True))
        row=feat[feat.date==g['date']].copy()
        if len(row)!=1:raise ValueError('Invalid pregame feature row')
        for k,v in zip(EXTRA,sides[0]-sides[1]):row[k]=v
        ts_now=dt.datetime.now(UTC)
        if ts_now>=g['tipoff']:skip.append({'game_id':g['game_id'],'reason':'Tipoff elapsed during training'});continue
        report_times=[rep.report.max().isoformat() for rep in s]
        for name,(cols,m,sd,cal,tm) in models.items():
            margin=float(m.predict(row[cols])[0]);p=expit(margin/(.6*sd));prob=float(cal.predict_proba(np.array([[logit(np.clip(p,1e-6,1-1e-6))]]))[0,1])
            rec=dict(forecast_id=g['game_id']+':'+name,game_id=g['game_id'],tipoff_at_utc=g['tipoff'].isoformat(),model_version=name,home=g['home'],away=g['away'],home_probability=prob,projected_margin=margin,projected_total=float(tm.predict(row[['rolling_total']])[0]),data_as_of_utc=ts_now.isoformat(),notes='T-90 paired shadow; report times '+','.join(report_times)+'; no same-day results; archive transcription')
            append(ledger,rec,ts_now);recorded.append(rec)
    finish('recorded' if recorded else 'no_eligible_predictions',recorded)
if __name__=='__main__':main()
