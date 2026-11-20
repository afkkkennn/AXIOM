"""Pregame features, shrinkage, chronological fitting and scoring."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error
from scipy.special import expit, logit

TEAM_COLUMNS = ['Team','Match up','Date','Pts','FGM','FGA','3PM','FTM','FTA','OREB','DREB','TOV']
FEATURES = ['net','efg','turnovers','oreb','ftr']
PRIOR = np.array([0,.54,.13,.25,.20,228,.54,.13,.25,.20])

def pair_games(raw):
    missing=set(TEAM_COLUMNS)-set(raw.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    r=raw.copy(); r['date']=pd.to_datetime(r['Date'],errors='raise').dt.normalize()
    matchup=r['Match up'].str.extract(r'^\s*(\S+)\s+(vs\.|@)\s+(\S+)\s*$')
    if matchup.isna().any().any() or not matchup[0].eq(r.Team).all(): raise ValueError('Malformed matchup')
    r['home']=np.where(matchup[1].eq('vs.'),matchup[0],matchup[2]);r['away']=np.where(matchup[1].eq('@'),matchup[0],matchup[2]);r['is_home']=matchup[1].eq('vs.')
    games=[]; rejected=0
    for (date,home,away),g in r.groupby(['date','home','away'],sort=True):
        if len(g)!=2 or g.is_home.sum()!=1: rejected+=1; continue
        h=g[g.is_home].iloc[0]; a=g[~g.is_home].iloc[0]
        if h.Pts==a.Pts: raise ValueError('Tied final score')
        games.append((date,home,away,h,a))
    return games,rejected

def stats(r,o):
    def poss(t):return t.FGA+.44*t.FTA-t.OREB+t.TOV
    def factors(x,y):
        return [(x.FGM+.5*x['3PM'])/x.FGA,x.TOV/(x.FGA+.44*x.FTA+x.TOV),x.OREB/(x.OREB+y.DREB),x.FTM/x.FGA]
    return np.array([(r.Pts-o.Pts)*100/((poss(r)+poss(o))/2),*factors(r,o),r.Pts+o.Pts,*factors(o,r)],dtype=float)

def features(raw, prior_games=3):
    games,rejected=pair_games(raw); hist={}; rows=[]
    def avg(team):
        h=hist.get(team,[]); return (prior_games*PRIOR+np.sum(h,axis=0))/(prior_games+len(h)) if h else PRIOR.copy()
    from itertools import groupby
    for date,g in groupby(games,key=lambda x:x[0]):
        g=list(g)
        for _,home,away,h,a in g:
            hd,ad=avg(home),avg(away); d=hd-ad
            rows.append(dict(date=date,home=home,away=away,net=d[0],efg=100*(d[1]-d[6]),turnovers=100*(-d[2]+d[7]),oreb=100*(d[3]-d[8]),ftr=100*(d[4]-d[9]),rolling_total=(hd[5]+ad[5])/2,margin=float(h.Pts-a.Pts),total=float(h.Pts+a.Pts)))
        for _,home,away,h,a in g:
            hist.setdefault(home,[]).append(stats(h,a));hist.setdefault(away,[]).append(stats(a,h))
    out=pd.DataFrame(rows);out.attrs['rejected_pairs']=rejected;return out

def score(y,prob,margin,actual_margin):
    prob=np.clip(prob,1e-6,1-1e-6)
    return dict(n=int(len(y)),accuracy=float(np.mean((prob>=.5)==y)),brier=float(brier_score_loss(y,prob)),log_loss=float(log_loss(y,prob,labels=[0,1])),margin_mae=float(mean_absolute_error(actual_margin,margin)))

def fit_predict(train,test,columns,alpha=100):
    if len(train)<10 or len(test)==0: raise ValueError('Insufficient train/test games')
    if columns:
        model=Ridge(alpha=alpha).fit(train[columns],train.margin)
        fit=model.predict(train[columns]); margin=model.predict(test[columns]); coefs={k:float(v) for k,v in zip(columns,model.coef_)}; intercept=float(model.intercept_)
    else:
        intercept=float(train.margin.mean()); fit=np.full(len(train),intercept); margin=np.full(len(test),intercept);coefs={}
    sigma=max(float(np.sqrt(np.mean((train.margin.to_numpy()-fit)**2))),1e-6)
    prob=expit(margin/(.6*sigma))
    return margin,prob,dict(intercept=intercept,coefficients=coefs,training_residual_rmse=sigma)

def run_split(raw,test_start='2026-11-11',validation_start=None):
    f=features(raw); split=pd.Timestamp(test_start); train=f[f.date<split];test=f[f.date>=split].copy();y=(test.margin>0).astype(int).to_numpy(); results={}
    for name,cols in [('home_baseline',[]),('rolling_net',['net']),('four_factors',FEATURES)]:
        margin,p,meta=fit_predict(train,test,cols);test[name+'_prob']=p;test[name+'_margin']=margin
        results[name]={**score(y,p,margin,test.margin),**meta}
        if validation_start:
            v=pd.Timestamp(validation_start);tr=train[train.date<v];val=train[train.date>=v]
            if len(tr)>=10 and len(val)>=20 and val.margin.gt(0).nunique()==2:
                _,vp,_=fit_predict(tr,val,cols);cal=LogisticRegression().fit(logit(np.clip(vp,1e-6,1-1e-6)).reshape(-1,1),val.margin.gt(0))
                vm,tp,_=fit_predict(tr,test,cols);cp=cal.predict_proba(logit(np.clip(tp,1e-6,1-1e-6)).reshape(-1,1))[:,1]
                results[name+'_calibrated']=score(y,cp,vm,test.margin);test[name+'_calibrated_prob']=cp
    results['totals']={'constant_mae':float(mean_absolute_error(test.total,np.full(len(test),train.total.mean()))),'rolling_mae':float(mean_absolute_error(test.total,test.rolling_total))}
    results['split']={'train':len(train),'test':len(test),'rejected_pairs':f.attrs['rejected_pairs'],'test_start':str(split.date()),'validation_start':validation_start}
    return results,test

def walk_forward(raw,test_starts):
    """Expanding history; each fold tests only until the next requested cut date."""
    starts=sorted(pd.Timestamp(d) for d in test_starts)
    raw_dates=pd.to_datetime(raw.Date);out=[]
    if len(set(starts))!=len(starts):raise ValueError('Duplicate cut dates')
    for i,start in enumerate(starts):
        sample=raw[raw_dates<starts[i+1]] if i+1<len(starts) else raw
        results,pred=run_split(sample,str(start.date()));out.append((results,pred))
    return out
