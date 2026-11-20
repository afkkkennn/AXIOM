"""Official report archive as-of snapshots and paired historical injury benchmark.
Archive data: SportsDataverse transcription of NBA reports, not direct live NBA data.
Unknown, unsubmitted and unmatched cases are separate from healthy.
"""
import datetime as dt,hashlib,io,json
from pathlib import Path
from collections import defaultdict,deque
from itertools import groupby
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import Ridge,LogisticRegression
from axiom.historical import HASHES,fetch,normalize,pair_games,team_features,MODELS
from axiom.ingest import git_blob_sha
from axiom.live import get,schedule,UTC,ET
from axiom.market_matched import canonical
from axiom.challengers import cluster_ci,metric
REPO='sportsdataverse/hoopR-nba-stats-data'
REF='b282d20b76404f4dfb1af2980e763fa831043773'
ARCHIVE={
2020:('4354c253f6a7fb4f6a473f9961b5b8d958d5ec2c','2bfb60aa8074061dacff0c319d376fca677a9b1b'),
2021:('4e737ff5da94b55d1edca4d0c3e3aed18a0ecc','f51707ee729c2bfdd92505ece36b7f98ddc869d4'),
2022:('9682048ced077e1bd5a0aa38288f311c46ee5a0c','96786217ff8063af1696597234efa1e4011ab6d4'),
2023:('a94f77c674d30c5b51496b0f582293fcfc1d5c8f','8781753b0fbe9a2d73d4e566195b4d2028d262e0'),
2024:('703fa829015c4cc911c0c7e9cf48d63d87fe8baa','1372cfe50ee5af0d9e3101a06e22c9c7735c632d'),
2025:('ac08641d824aa9aaf8659aa213c573d551833404','2c2da3db57d1e64dca4d897baac3e19e11e826ab'),
2026:('c5f062c2392b8404f881484171231e6e0d799669','032f4d7395bf595695619be5a46b1270831ca18b'),
2027:('388f45aa703bf9da3b4651d411fe066cc6f84540','23a1501a27c2657550235e96886265ba6669b2e5')}
STATUS={'out':0.,'doubtful':.15,'questionable':.5,'probable':.85,'available':1.}
EXTRA=['injury_minutes','injury_points','injury_quality','injury_rows']
from functools import lru_cache
@lru_cache(maxsize=4)
def tree_hashes(ref):
    api=f'https://api.github.com/repos/{REPO}'
    tree=get(api+f'/git/trees/{ref}',params={'recursive':'1'}).json()
    if tree.get('truncated'):raise ValueError('Archive tree truncated; cannot verify files')
    return {x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
def source(y,kind,manifest,dynamic=False):
    ref=REF
    if dynamic:
        ref=get(f'https://api.github.com/repos/{REPO}/git/ref/heads/main').json()['object']['sha']
    path=f'nba_stats/{kind}/parquet/{kind}_{y}.parquet'
    expected=tree_hashes(ref)[path]
    url=f'https://raw.githubusercontent.com/{REPO}/{ref}/{path}';raw=get(url).content
    actual=git_blob_sha(raw)
    if actual!=expected:raise ValueError(f'Archive hash mismatch at {path}: expected {expected}, got {actual}')
    manifest.append(dict(url=url,git_blob=expected,sha256=hashlib.sha256(raw).hexdigest(),retrieved_at_utc=dt.datetime.now(UTC).isoformat()))
    return pd.read_parquet(io.BytesIO(raw))
def eastern(x):
    a=pd.to_datetime(x,errors='coerce')
    if a.dt.tz is None:return a.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    return a.dt.tz_convert('UTC')
def index_reports(data):
    a=data.copy();a['date']=pd.to_datetime(a.game_date).dt.normalize();a['report']=eastern(a.report_datetime)
    a['home']=a.home_team_abbr.map(canonical);a['away']=a.away_team_abbr.map(canonical);a['tm']=a.team.map(canonical)
    a=a.dropna(subset=['date','report','home','away','tm'])
    return {(d,h,w,t):g.sort_values('report') for (d,h,w,t),g in a.groupby(['date','home','away','tm'],sort=False)}
def snapshot(index,date,home,away,team,cutoff):
    g=index.get((date,home,away,team))
    if g is None:return None
    eligible=g[(g.report<=cutoff)&(g.report>=cutoff-pd.Timedelta(hours=48))]
    if eligible.empty:return None
    latest=eligible.report.max();s=eligible[eligible.report==latest]
    if s.reason.fillna('').str.contains('NOT YET SUBMITTED',case=False).any():return None
    return s
def player_history(raw):
    p=raw.copy();p=p[p.season_type.astype(str).str.lower().str.replace(' ','-').eq('regular-season')].copy()
    p['Date']=pd.to_datetime(p.game_date).dt.normalize();p['Team']=p.team_abbreviation.map(canonical)
    for k in ['min','pts','ast','reb','plus_minus']:p[k]=pd.to_numeric(p[k],errors='coerce')
    p=p.dropna(subset=['player_id','Team','min','pts','ast','reb','plus_minus']);return p[p['min']>0]
def build_features(teams,tipoffs,players,reports):
    idx=index_reports(reports);hist=defaultdict(lambda:deque(maxlen=10));last=defaultdict(lambda:deque(maxlen=25));last_team={}
    byplayer={(day,t):g for (day,t),g in players.groupby(['Date','Team'],sort=False)}
    games,_=pair_games(teams);season=None;rows=[];coverage=defaultdict(lambda:[0,0,0,0])
    status_stats=defaultdict(lambda:[0,0])
    for date,day in groupby(games,key=lambda x:x[0]):
        day=list(day);sy=int(day[0][3].Season)
        if sy!=season:hist.clear();season=sy
        pending=[]
        for _,h,a,hr,ar in day:
            coverage[sy][0]+=1;tip=tipoffs.get(str(hr.game_id))
            if tip is None or pd.isna(tip):continue
            cutoff=tip-pd.Timedelta(minutes=90);vec=[];complete=True
            for t in [h,a]:
                actual=byplayer.get((date,t));prior=list(hist[t]);s=snapshot(idx,date,h,a,t,cutoff)
                if actual is not None:pending.append((t,actual))
                if s is None or len(prior)<5 or actual is None:
                    complete=False;continue
                delta_minutes=delta_points=delta_quality=0.;count=0
                listed=s.dropna(subset=['person_id','current_status']).drop_duplicates('person_id')
                actual_ids=set(actual.player_id.astype(int))
                for r in listed.itertuples():
                    pid=int(r.person_id);status=str(r.current_status).lower()
                    if status not in STATUS:continue
                    status_stats[(sy,status)][0]+=1;status_stats[(sy,status)][1]+=int(pid in actual_ids)
                    if last_team.get(pid)!=t or len(last[pid])<3:continue
                    past=np.array([g.get(pid,0.) for g in prior]);played=past>0
                    if not played.any():continue
                    typical=float(np.mean(past[played]));baseline=float(played.mean());expected=STATUS[status]
                    vals=np.asarray(last[pid],float) # minutes,p,ast,reb,plusminus
                    rate=float(vals[:,1].sum()/(vals[:,0].sum()+1e-6))
                    # Shrunk box-score quality proxy, not RAPM; coefficient learned from train.
                    quality=float((vals[:,1]+.7*vals[:,2]+.25*vals[:,3]).sum()/(vals[:,0].sum()+1e-6))
                    dm=(expected-baseline)*typical
                    delta_minutes+=dm;delta_points+=dm*rate;delta_quality+=dm*quality;count+=1
                coverage[sy][2]+=len(listed);coverage[sy][3]+=count
                vec.append(np.array([delta_minutes,delta_points,delta_quality,float(count)]))
            if complete and len(vec)==2:
                coverage[sy][1]+=1;rows.append(dict(date=date,home=h,away=a,**dict(zip(EXTRA,vec[0]-vec[1]))))
        for t,actual in pending:
            mins={}
            for q in actual[['player_id','min','pts','ast','reb','plus_minus']].itertuples(index=False,name=None):
                pid,mi,*v=q;pid=int(pid);mins[pid]=mi;last[pid].append([mi,*v]);last_team[pid]=t
            hist[t].append(mins)
    return pd.DataFrame(rows),dict(coverage),dict(status_stats)
def run_benchmark(f,inj):
    d=f.merge(inj,on=['date','home','away'],validate='one_to_one');tr=d[d.season<=2024];v=d[d.season==2025];te=d[d.season==2026].copy()
    if min(len(tr),len(v),len(te))<100:raise ValueError(f'Insufficient matched injury snapshots: {len(tr)},{len(v)},{len(te)}')
    y=te.margin.gt(0).to_numpy(int);probs={};metrics={}
    for name,cols in [('matched_team_baseline',MODELS['ensemble']),('injury_augmented',MODELS['ensemble']+EXTRA)]:
        model=Ridge(alpha=100).fit(tr[cols],tr.margin);sd=np.sqrt(np.mean((tr.margin-model.predict(tr[cols]))**2));vp=expit(model.predict(v[cols])/(.6*sd));marg=model.predict(te[cols]);rp=expit(marg/(.6*sd))
        cal=LogisticRegression().fit(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1),v.margin.gt(0));p=cal.predict_proba(logit(np.clip(rp,1e-6,1-1e-6)).reshape(-1,1))[:,1]
        probs[name]=p;te[name+'_prob']=p;metrics[name]={**metric(y,p),'margin_mae':float(np.abs(marg-te.margin).mean()),'coefficients':dict(zip(cols,map(float,model.coef_)))}
    metrics['delta_log_loss_ci95']=cluster_ci(te.date.to_numpy(),y,probs['injury_augmented'],probs['matched_team_baseline']);metrics['split']={'train':len(tr),'calibration':len(v),'test':len(te)}
    return metrics,te
def current_snapshot(out):
    now=dt.datetime.now(UTC);season=now.astimezone(ET).year+int(now.astimezone(ET).month>=7);manifest=[]
    raw=source(season,'injury_report',manifest,dynamic=True);idx=index_reports(raw);rows=[];skipped=[]
    for g in schedule(now):
        for team in [g['home'],g['away']]:
            s=snapshot(idx,g['date'],g['home'],g['away'],team,pd.Timestamp(now))
            if s is None:skipped.append({'game_id':g['game_id'],'team':team,'reason':'No eligible submitted archive snapshot'});continue
            for r in s.itertuples():
                rows.append(dict(game_id=g['game_id'],team=team,tipoff_at_utc=g['tipoff'].isoformat(),report_time_utc=r.report.isoformat(),player=str(r.report_player_name),status=str(r.current_status),reason=str(r.reason)))
    m={'retrieved_at_utc':now.isoformat(),'source_manifest':manifest,'archive_rows':len(raw),'archive_latest_report':str(eastern(raw.report_datetime).max()),'archive_latest_game_date':str(pd.to_datetime(raw.game_date).max()),'rows':len(rows),'unavailable_team_snapshots':skipped,'warning':'An archive may lag the current official report; unlisted players are not automatically confirmed active'}
    out.mkdir(parents=True,exist_ok=True);(out/'current_report.json').write_text(json.dumps(m,indent=2));pd.DataFrame(rows).to_csv(out/'current_report_rows.csv',index=False);print(json.dumps(m,indent=2))
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--current-only',action='store_true');a=parser.parse_args();out=Path('reports/official_injury');out.mkdir(parents=True,exist_ok=True)
    if a.current_only:current_snapshot(out);return
    manifest=[];ts=[];rs=[];ps=[];tipoffs={}
    for y in HASHES:
        raw=fetch(y,'team_box',manifest);ts.append(normalize(raw,y,'team_box'))
        tips=eastern(raw.game_date_time)
        for gid,tip in zip(raw.game_id,tips):tipoffs[str(gid)]=tip
        rs.append(source(y,'injury_report',manifest));ps.append(player_history(source(y,'player_game_logs',manifest)))
    t=pd.concat(ts,ignore_index=True);f,_=team_features(t);inj,c,ss=build_features(t,tipoffs,pd.concat(ps,ignore_index=True),pd.concat(rs,ignore_index=True))
    m,p=run_benchmark(f,inj);m['coverage_by_season']={str(y):dict(all_games=v[0],matched_games=v[1],listed_players=v[2],matched_prior_players=v[3]) for y,v in c.items()}
    m['listed_status_observed_appearance_2026']={st:dict(n=v[0],appeared=v[1],rate=v[1]/v[0] if v[0] else None) for (y,st),v in ss.items() if y==2026}
    m['limitations']=['Reported times from archive interpreted in US Eastern; 90-minute cutoff','Require both team snapshots and prior history; matched subset only','Archive can omit unlisted/clear teams; no inferred healthy status','Identity matches and reconstructed historical source, not record of when AXIOM retrieved it','Fixed status probabilities; quality proxy not causal replacement impact','2025-26 retrospective, not untouched new test']
    (out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));p.to_csv(out/'predictions.csv',index=False);print(json.dumps(m,indent=2))
if __name__=='__main__':main()
