"""Compare exact alpha.6 champion probabilities on matched archived-market games.
Retrospective only; quote observation timestamps and Bet88 prices are unavailable.
"""
import io,json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import brier_score_loss,log_loss
from axiom.historical import HASHES,fetch,normalize,team_features,MODELS,ALIASES,TEAMS
from axiom.live import train_models,get
from axiom.market_history import URL,SHA
from axiom.ingest import git_blob_sha
from axiom.challengers import cluster_ci
NAMES=dict(zip('ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS'.split(),
'Atlanta Hawks|Boston Celtics|Brooklyn Nets|Charlotte Hornets|Chicago Bulls|Cleveland Cavaliers|Dallas Mavericks|Denver Nuggets|Detroit Pistons|Golden State Warriors|Houston Rockets|Indiana Pacers|Los Angeles Clippers|Los Angeles Lakers|Memphis Grizzlies|Miami Heat|Milwaukee Bucks|Minnesota Timberwolves|New Orleans Pelicans|New York Knicks|Oklahoma City Thunder|Orlando Magic|Philadelphia 76ers|Phoenix Suns|Portland Trail Blazers|Sacramento Kings|San Antonio Spurs|Toronto Raptors|Utah Jazz|Washington Wizards'.split('|')))
def clean(s):return re.sub('[^a-z0-9]','',str(s).lower())
LOOKUP={clean(k):k for k in TEAMS};LOOKUP.update({clean(k):v for k,v in ALIASES.items()})
LOOKUP.update({clean(v):k for k,v in NAMES.items()});LOOKUP.update({clean(v.split()[-1]):k for k,v in NAMES.items()})
LOOKUP.update({clean(v):k for v,k in [('LA Clippers','LAC'),('LA Lakers','LAL'),('Portland','POR'),('Trail Blazers','POR'),('Philadelphia','PHI'),('Golden State','GSW'),('Oklahoma City','OKC'),('San Antonio','SAS'),('New Orleans','NOP'),('New York','NYK'),('Utah','UTA'),('New Jersey Nets','BKN'),('Charlotte Bobcats','CHA')]})
def canonical(x):
    s=clean(x)
    if s in LOOKUP:return LOOKUP[s]
    # City-only names that identify exactly one current franchise.
    options={k for k,v in NAMES.items() if clean(v).startswith(s) and len(s)>=4}
    return options.pop() if len(options)==1 else None
def main():
    manifest=[];t=pd.concat([normalize(fetch(y,'team_box',manifest),y,'team_box') for y in HASHES],ignore_index=True)
    f,_=team_features(t);te=f[f.season==2026].copy();m,s,c,tm,res=train_models(f)
    raw=expit(m.predict(te[MODELS['ensemble']])/(.6*s));te['champion_prob']=c.predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    data=get(URL).content
    if git_blob_sha(data)!=SHA:raise ValueError('Market source hash mismatch')
    d=pd.read_csv(io.BytesIO(data));d['date']=pd.to_datetime(d.Date).dt.normalize();d['home']=d.Home.map(canonical);d['away']=d.Away.map(canonical)
    d=d[d.date.between(te.date.min(),te.date.max())].copy();d['market_prob']=pd.to_numeric(d.market_home_prob,errors='coerce');d=d[d.market_prob.between(0,1,inclusive='neither')]
    unknown=int(d[['home','away']].isna().any(axis=1).sum());d=d.dropna(subset=['home','away'])
    if d.duplicated(['date','home','away']).any():raise ValueError('Duplicate market matches')
    q=te.merge(d[['date','home','away','home_win','market_prob']],on=['date','home','away'],validate='one_to_one')
    if len(q)<100:raise ValueError(f'Insufficient matches ({len(q)}); input teams examples {d.Home.head(5).tolist()}')
    y=q.margin.gt(0).to_numpy(int)
    if not np.array_equal(y,q.home_win.to_numpy(int)):raise ValueError('Market/team result disagreement')
    def stats(p):return dict(n=len(y),accuracy=float(((p>=.5)==y).mean()),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))
    result=dict(matched_games=len(q),champion_test_games=len(te),market_eligible_rows=len(d),unmapped_market_teams=unknown,champion=stats(q.champion_prob),archived_market=stats(q.market_prob),delta_log_loss_ci95_champion_minus_market=cluster_ci(q.date.to_numpy(),y,q.champion_prob.to_numpy(),q.market_prob.to_numpy()),source=dict(url=URL,git_blob=SHA,sha256=hashlib.sha256(data).hexdigest()),limitations=['Matched subset of already evaluated season','Market probabilities not independently timestamped or reconstructed from both sides','No Bet88 claim'])
    out=Path('reports/market_matched');out.mkdir(parents=True,exist_ok=True);q.to_csv(out/'predictions.csv',index=False);(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
