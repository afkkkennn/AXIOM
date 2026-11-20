"""Pinned multi-season data and chronological model comparisons.
Split/fixed model specifications declared before running the test.
"""
import argparse,hashlib,io,json
from pathlib import Path
from collections import defaultdict,deque
from itertools import groupby
import numpy as np
import pandas as pd
import requests
from scipy.special import expit,logit
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss
from axiom.ingest import git_blob_sha
from axiom.team import pair_games,stats,score

REF='e4a0559bf034cdcea9f896889bdfbeb4fab0bf61'
REPO='sportsdataverse/hoopR-nba-data'
HASHES={
2020:('4c9f8a533302a4cb701d8d53aa1d0937165388c3','4536405ccc661ba9aa001ee7e43bfb7f4e9ad1c2'),
2021:('80827c076ebcf0b75e10431d242e26cdc9635125','c3adaad63e88c097d0e627a00c9a5589db77e6cf'),
2022:('56b209c29b778e1abdcb9d86545958fd1cbcee60','2963ddc82fefbb512a2a015365a6b902aa150bb5'),
2023:('8bf97cf2bdf522d1ff12252904fea412519b3c94','741c006a695b9071e938ab2e060699842725a405'),
2024:('aaaeb807397540d80819232fa4e8d3839fee0a81','8b60f5ba6c86bea6c6b454336e7706c6a29f0967'),
2025:('ac316e9feeda7339479c163e4af88c21154ea677','23e0e73a60a1b17d38c49275fe1d16196efbd596'),
2026:('bbbf029793b85be3eb7d90cb548f97204dd31b22','121dcfd61ab4221c357e142d18b73a1cec14723c')}
TEAMS=set('ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS'.split())
ALIASES={'GS':'GSW','NY':'NYK','NO':'NOP','SA':'SAS','UTAH':'UTA','WSH':'WAS','BK':'BKN','BRK':'BKN','PHO':'PHX'}
T_MAP={'team_score':'Pts','field_goals_made':'FGM','field_goals_attempted':'FGA','three_point_field_goals_made':'3PM','free_throws_made':'FTM','free_throws_attempted':'FTA','offensive_rebounds':'OREB','defensive_rebounds':'DREB','total_turnovers':'TOV'}
P_MAP={'athlete_id':'Player ID','minutes':'Min','points':'Pts','rebounds':'REB','assists':'AST','three_point_field_goals_made':'3PM'}
PRIOR=np.array([0,.54,.13,.25,.20,225,.54,.13,.25,.20])
def fetch(y,kind,manifest):
    path=f'nba/{kind}/parquet/{kind}_{y}.parquet';url=f'https://raw.githubusercontent.com/{REPO}/{REF}/{path}'
    r=requests.get(url,timeout=(15,90));r.raise_for_status();raw=r.content
    actual=git_blob_sha(raw);expected=HASHES[y][kind=='player_box']
    if actual!=expected:raise ValueError(f'Hash mismatch {path}: {actual} != {expected}')
    manifest.append(dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),git_blob=actual));return pd.read_parquet(io.BytesIO(raw))

def normalize(d,y,kind):
    d=d[d.season_type.astype(str).isin(['2','Regular Season'])].copy()
    d['Team']=d.team_abbreviation.replace(ALIASES);d['Opp']=d.opponent_team_abbreviation.replace(ALIASES)
    d=d[d.Team.isin(TEAMS)&d.Opp.isin(TEAMS)].copy()
    d['Date']=pd.to_datetime(d.game_date).dt.tz_localize(None).dt.normalize();d['Season']=y
    if kind=='team_box':
        venue=d['team_home_away'] if 'team_home_away' in d else d['home_away']
        if not venue.isin(['home','away']).all():raise ValueError('Unknown venue')
        d['Match up']=d.Team+np.where(venue.eq('home'),' vs. ',' @ ')+d.Opp
        cols=['game_id','Date','Season','Team','Opp','Match up',*T_MAP.values()]
        d=d.rename(columns=T_MAP)[cols]
        for k in T_MAP.values():d[k]=pd.to_numeric(d[k],errors='coerce')
        # Reject whole incomplete games, not individual sides.
        valid=d[list(T_MAP.values())].notna().all(axis=1)&d.FGA.gt(0)&d.Pts.gt(0)
        d=d[~d.game_id.isin(d.loc[~valid,'game_id'])]
        if d.duplicated(['game_id','Team']).any():raise ValueError('Duplicate team record')
    else:
        d=d.rename(columns=P_MAP)
        cols=['game_id','Date','Season','Team','Opp',*P_MAP.values()]
        d=d[cols].copy()
        for k in ['Min','Pts','REB','AST','3PM']:d[k]=pd.to_numeric(d[k],errors='coerce')
        d=d.dropna(subset=['Player ID','Min','Pts','REB','AST','3PM']);d=d[d.Min.gt(0)]
        if d.duplicated(['game_id','Player ID']).any():raise ValueError('Duplicate player record')
    return d

def team_features(raw):
    games,rejected=pair_games(raw)
    hist=defaultdict(lambda:deque(maxlen=25));last={};elo=defaultdict(lambda:1500.);rows=[]
    season=None
    def avg(team):
        h=np.asarray(hist[team])
        if not len(h):return PRIOR.copy()
        w=.94**np.arange(len(h)-1,-1,-1)
        return (w@h+3*PRIOR)/(w.sum()+3)
    for dt,day in groupby(games,key=lambda x:x[0]):
        day=list(day);sy=int(day[0][3].Season)
        if sy!=season:
            if season is not None:
                for t in list(elo):elo[t]=1500+.70*(elo[t]-1500)
                for t in list(hist):hist[t]=deque(list(hist[t])[-8:],maxlen=25)
            season=sy
        pending=[]
        for _,h,a,hr,ar in day:
            hd,ad=avg(h),avg(a);d=hd-ad
            hrst=min(7,(dt-last[h]).days) if h in last else 7
            arst=min(7,(dt-last[a]).days) if a in last else 7
            elo_diff=elo[h]-elo[a];m=float(hr.Pts-ar.Pts)
            rows.append(dict(date=dt,season=sy,home=h,away=a,net=d[0],efg=100*(d[1]-d[6]),turnovers=100*(-d[2]+d[7]),oreb=100*(d[3]-d[8]),ftr=100*(d[4]-d[9]),rest=hrst-arst,b2b=int(hrst==1)-int(arst==1),elo=elo_diff/100,rolling_total=(hd[5]+ad[5])/2,margin=m,total=float(hr.Pts+ar.Pts)))
            prob=1/(1+10**(-(elo_diff+65)/400));change=20*((m>0)-prob)
            pending.append((h,a,stats(hr,ar),stats(ar,hr),change))
        for h,a,hv,av,c in pending:
            hist[h].append(hv);hist[a].append(av);elo[h]+=c;elo[a]-=c;last[h]=last[a]=dt
    f=pd.DataFrame(rows);return f,rejected

MODELS={'net':['net'],'four_factors':['net','efg','turnovers','oreb','ftr'],'elo_rest':['elo','rest','b2b'],'ensemble':['net','efg','turnovers','oreb','ftr','elo','rest','b2b']}
def benchmark(f):
    """Train 2020–24; calibrate 2025; evaluate 2026 once."""
    tr=f[f.season<=2024];v=f[f.season==2025];te=f[f.season==2026].copy()
    if min(len(tr),len(v),len(te))<500:raise ValueError('Insufficient split games')
    y=te.margin.gt(0).astype(int);vy=v.margin.gt(0).astype(int);metrics={}
    for name,cols in MODELS.items():
        m=Ridge(alpha=100).fit(tr[cols],tr.margin)
        s=float(np.sqrt(np.mean((tr.margin-m.predict(tr[cols]))**2)))
        vp=expit(m.predict(v[cols])/(.6*s));tp=expit(m.predict(te[cols])/(.6*s));margin=m.predict(te[cols])
        cal=LogisticRegression().fit(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1),vy)
        cp=cal.predict_proba(logit(np.clip(tp,1e-6,1-1e-6)).reshape(-1,1))[:,1]
        te[name+'_raw_prob']=tp;te[name+'_calibrated_prob']=cp
        metrics[name]={'raw':score(y,tp,margin,te.margin),'calibrated':score(y,cp,margin,te.margin),'validation_log_loss':float(log_loss(vy,cal.predict_proba(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1))[:,1]))}
        metrics[name]['calibrated']['bins']=[dict(n=int(ix.sum()),mean_prob=float(cp[ix].mean()),actual=float(y.to_numpy()[ix].mean())) for i in range(10) if (ix:=(cp>=i/10)&(cp<(i+1)/10)).any()]
    total_fit=Ridge(alpha=100).fit(tr[['rolling_total']],tr.total);ts=total_fit.predict(te[['rolling_total']])
    metrics['totals']={'constant_mae':float(np.abs(te.total-tr.total.mean()).mean()),'rolling_mae':float(np.abs(te.total-te.rolling_total).mean()),'fitted_mae':float(np.abs(te.total-ts).mean())}
    metrics['split']={'train':len(tr),'calibration':len(v),'test':len(te)}
    return metrics,te

def player_benchmark(p):
    """Evaluate known appearances only. Opponent adjustments from prior games, shrunk."""
    hist=defaultdict(lambda:deque(maxlen=30));opps=defaultdict(lambda:deque(maxlen=200));league=deque(maxlen=3000);rows=[]
    cols=['Pts','REB','AST','3PM']
    for dt,g in p.sort_values('Date').groupby('Date',sort=True):
        pending=[]
        for x in g[['Season','Player ID','Team','Opp','Min',*cols]].itertuples(index=False,name=None):
            sy,pid,team,opp,mi,*st=x;z=np.array(st,dtype=float);h=np.array(hist[pid])
            if sy==2026 and len(h)>=5:
                recent=h[-8:];base=h[:,1:].mean(0);mins=.6*recent[:,0].mean()+.4*h[:,0].mean()
                rate=mins*(.6*recent[:,1:].sum(0)/recent[:,0].sum()+.4*h[:,1:].sum(0)/h[:,0].sum())
                l=np.array(league);o=np.array(opps[opp]);adjust=np.ones(4)
                if len(l)>=200 and len(o)>=30:
                    lrate=l[:,1:].sum(0)/l[:,0].sum();orate=(o[:,1:].sum(0)+900*lrate)/(o[:,0].sum()+900)
                    adjust=np.clip(orate/np.maximum(lrate,1e-6),.85,1.15)
                rows.append([str(dt.date()),pid,mi,*z,*base,*rate,*(rate*adjust)])
            pending.append((pid,opp,np.r_[mi,z]))
        for pid,opp,vec in pending:hist[pid].append(vec);opps[opp].append(vec);league.append(vec)
    names=['date','player_id','minutes']+[f'{k}_{m}' for m in ['actual','average','rate','opponent'] for k in cols]
    pred=pd.DataFrame(rows,columns=names)
    for m in ['actual','average','rate','opponent']:pred['PRA_'+m]=sum(pred[k+'_'+m] for k in cols[:3])
    metrics={k:{m:float(np.abs(pred[k+'_'+m]-pred[k+'_actual']).mean()) for m in ['average','rate','opponent']} for k in cols+['PRA']}
    metrics['n']=len(pred);return metrics,pred

def main():
    out=Path('reports/multiseason');out.mkdir(parents=True,exist_ok=True);manifest=[];ts=[];ps=[]
    for y in HASHES:
        ts.append(normalize(fetch(y,'team_box',manifest),y,'team_box'));ps.append(normalize(fetch(y,'player_box',manifest),y,'player_box'))
    t=pd.concat(ts,ignore_index=True);p=pd.concat(ps,ignore_index=True)
    f,rejected=team_features(t);m,pr=benchmark(f);m['players'],pp=player_benchmark(p)
    m['data']={'team_rows':len(t),'player_appearances':len(p),'paired_games':len(f),'rejected_pairs':rejected,'games_by_season':{str(k):int(v) for k,v in f.season.value_counts().items()}}
    m['limitations']=['ESPN source regular-season labels include Cup final where applicable','Player test is appearance-conditional','No historical point-in-time injury or odds archive in this run','2026 holdout now consumed; do not tune and call it untouched']
    pr.to_csv(out/'team_predictions.csv',index=False);pp.to_csv(out/'player_predictions.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
