"""Retrospective market baseline. Source quotes lack verified observation times.
Independent scores-only Elo/rest model, matched on the SAME source games.
Not a strict timestamp-audited market test and not a Bet88 evaluation.
"""
import io,json,hashlib
from pathlib import Path
from collections import defaultdict,deque
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss
from scipy.special import logit
from axiom.ingest import git_blob_sha
URL='https://raw.githubusercontent.com/aaryanshah607/nba-line-sheet/76a5e1b0e8e8dd7f6961ce2594c1ca19d698439e/nba_odds_clean.csv'
SHA='c8876c2bb4fe2cff5e49a697522ac091caa5ff7c'
def compute(df):
    required={'Date','Home','Away','home_win','market_home_prob'}
    if missing:=required-set(df.columns):raise ValueError(f'Required columns missing: {missing}. Source columns: {list(df.columns)}')
    d=df.copy();d.Date=pd.to_datetime(d.Date,errors='raise').dt.normalize()
    if d.duplicated(['Date','Home','Away']).any():raise ValueError('Duplicate games')
    d.home_win=pd.to_numeric(d.home_win,errors='raise');d.market_home_prob=pd.to_numeric(d.market_home_prob,errors='coerce')
    if not d.home_win.isin([0,1]).all():raise ValueError('Nonbinary outcome')
    elo=defaultdict(lambda:1500.);last={};h=defaultdict(lambda:deque(maxlen=20));rows=[]
    for day,g in d.groupby('Date',sort=True):
        changes=[]
        for row in g.itertuples():
            a,b=row.Home,row.Away;y=int(row.home_win);p=1/(1+10**(-(elo[a]-elo[b]+65)/400))
            rest=lambda x:min(7,(day-last[x]).days) if x in last else 7
            rate=lambda x:(sum(h[x])+2)/(len(h[x])+4)
            rows.append(dict(date=day,home=a,away=b,y=y,market=row.market_home_prob,elo=(elo[a]-elo[b])/100,rest=rest(a)-rest(b),b2b=int(rest(a)==1)-int(rest(b)==1),form=rate(a)-rate(b)))
            changes.append((a,b,y,20*(y-p)))
        for a,b,y,c in changes:elo[a]+=c;elo[b]-=c;last[a]=last[b]=day;h[a].append(y);h[b].append(1-y)
    f=pd.DataFrame(rows);cols=['elo','rest','b2b','form']
    train=f[f.date<'2024-07-01'];cal=f[(f.date>='2024-07-01')&(f.date<'2025-07-01')];te=f[f.date>='2025-07-01'].copy()
    te=te[te.market.between(0,1,inclusive='neither')].copy()
    if min(len(train),len(cal),len(te))<500:raise ValueError('Insufficient market evaluation sample')
    m=LogisticRegression(max_iter=1000).fit(train[cols],train.y)
    p=m.predict_proba(cal[cols])[:,1];c=LogisticRegression().fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),cal.y)
    te['raw']=m.predict_proba(te[cols])[:,1];te['calibrated']=c.predict_proba(logit(np.clip(te.raw.to_numpy(),1e-6,1-1e-6)).reshape(-1,1))[:,1]
    def metrics(p):return dict(n=len(te),accuracy=float(((p>=.5)==te.y).mean()),brier=float(brier_score_loss(te.y,p)),log_loss=float(log_loss(te.y,p,labels=[0,1])))
    return {'train':len(train),'calibration':len(cal),'matched_test':len(te),'independent_raw':metrics(te.raw),'independent_calibrated':metrics(te.calibrated),'archived_market':metrics(te.market),'limitations':['Source market probabilities not independently reconstructed from both sides','Unverified quote observation times','Source game coverage may omit games','Not Bet88 odds','Scores-only comparator is distinct from alpha.6 box-score ensemble']},te
def main():
    r=requests.get(URL,timeout=(15,90));r.raise_for_status();raw=r.content
    if git_blob_sha(raw)!=SHA:raise ValueError('Source hash mismatch')
    m,p=compute(pd.read_csv(io.BytesIO(raw)));m['source']={'url':URL,'git_blob':SHA,'sha256':hashlib.sha256(raw).hexdigest()}
    out=Path('reports/market_history');out.mkdir(parents=True,exist_ok=True);p.to_csv(out/'predictions.csv',index=False);(out/'metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
