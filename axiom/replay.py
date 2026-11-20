"""Retrospective pregame replay, never writes the live ledger.
Uses reconstructed sources with prior-game dates only; not a timestamped original forecast.
"""
import argparse,datetime as dt,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from axiom.historical import HASHES,fetch,normalize,team_features,MODELS
from axiom.live import current_teams,dummy_pair,train_models,UTC
from axiom.official_injuries import source,eastern,player_history,build_features,index_reports,snapshot,EXTRA
from axiom.injury_shadow import side_vector,train,MODEL,CONTROL

def main():
    p=argparse.ArgumentParser();p.add_argument('--date',default='2026-11-18');p.add_argument('--home',default='LAL');p.add_argument('--away',default='UTA');p.add_argument('--tipoff',default='2026-11-19T03:00:00Z');a=p.parse_args()
    date=pd.Timestamp(a.date);season=date.year+int(date.month>=7);tip=pd.Timestamp(a.tipoff);g=dict(date=date,home=a.home,away=a.away);manifest=[]
    if tip.tzinfo is None:raise ValueError('Tipoff needs timezone')
    hist=[];tips={}
    for y in HASHES:
        raw=fetch(y,'team_box',manifest);hist.append(normalize(raw,y,'team_box'))
        for gid,t in zip(raw.game_id,eastern(raw.game_date_time)):tips[str(gid)]=t
    h=pd.concat(hist,ignore_index=True);f,_=team_features(h);m,s,c,tm,res=train_models(f)
    cur=current_teams(season,manifest);cur=cur[cur.Date<date].copy()
    if min(len(cur[cur.Team==a.home]),len(cur[cur.Team==a.away]))<3:raise ValueError('Insufficient pregame data')
    features,_=team_features(pd.concat([h[h.Date<date],cur,pd.DataFrame(dummy_pair(g,season))],ignore_index=True))
    row=features[features.date==date].copy()
    if len(row)!=1:raise ValueError('Incorrect replay feature row')
    margin=float(m.predict(row[MODELS['ensemble']])[0]);total=float(tm.predict(row[['rolling_total']])[0])
    probability=float(c.predict_proba(np.array([[logit(np.clip(expit(margin/(.6*s)),1e-6,1-1e-6))]]))[0,1])
    result=dict(type='retrospective_replay_not_prospective',date=a.date,home=a.home,away=a.away,latest_game_date=str(cur.Date.max().date()),baseline=dict(home_probability=probability,projected_margin=margin,projected_total=total),team_features={k:float(row[k].iloc[0]) for k in MODELS['ensemble']},previous_games={t:cur[cur.Team==t].sort_values('Date').tail(3)[['Date','Opp','Pts']].assign(Date=lambda x:x.Date.astype(str)).to_dict('records') for t in [a.home,a.away]})
    try:
        ir=source(season,'injury_report',manifest,dynamic=True);idx=index_reports(ir);cut=tip-pd.Timedelta(minutes=90)
        snaps=[snapshot(idx,date,a.home,a.away,t,cut) for t in [a.home,a.away]]
        if any(x is None or x.empty or cut-x.report.max()>pd.Timedelta(hours=8) for x in snaps):raise ValueError('No fresh paired archived injury snapshots')
        irs=[];ps=[]
        for y in HASHES:
            irs.append(source(y,'injury_report',manifest));ps.append(player_history(source(y,'player_game_logs',manifest)))
        inj,_,_=build_features(h,tips,pd.concat(ps,ignore_index=True),pd.concat(irs,ignore_index=True));models=train(f,inj)
        current=player_history(source(season,'player_game_logs',manifest,dynamic=True));current=current[current.Date<date]
        players=pd.concat([ps[-1],current],ignore_index=True)
        sides=[side_vector(players,t,date,rep,pd.Timestamp(season-1,7,1)) for t,rep in zip([a.home,a.away],snaps)]
        if any(x is None for x in sides):raise ValueError('Insufficient identifiable player history')
        for k,v in zip(EXTRA,sides[0]-sides[1]):row[k]=v
        result['paired_models']={}
        for name,(cols,fit,sd,cal,tot) in models.items():
            mm=float(fit.predict(row[cols])[0]);pwin=float(cal.predict_proba(np.array([[logit(np.clip(expit(mm/(.6*sd)),1e-6,1-1e-6))]]))[0,1])
            result['paired_models'][name]=dict(home_probability=pwin,projected_margin=mm,projected_total=float(tot.predict(row[['rolling_total']])[0]))
        result['injury_report_times']=[x.report.max().isoformat() for x in snaps]
    except (ValueError,KeyError,TypeError) as e:
        result['injury_replay_unavailable']=str(e)
    result['limitations']=['Reconstructed historical feeds downloaded after game; no original as-of ingestion proof','No target-game results used as inputs; no market probabilities','Only fixed learned fatigue feature for back-to-back; no distinct overtime fatigue feature','No live ledger entry; not an independent prospective test']
    result['manifest']=manifest
    out=Path('reports/replay');out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='manifest'},indent=2))
if __name__=='__main__':main()
