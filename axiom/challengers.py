"""Development folds and retrospective 2026 comparison, not new untouched holdout."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss,brier_score_loss
from axiom.historical import HASHES,fetch,normalize,team_features,MODELS,player_benchmark
from axiom.live import train_models
from axiom.simulation import score_draws,player_draws,interval_coverage

COLS=MODELS['ensemble']
def fit_models(tr,va,te):
    y=tr.margin.gt(0);vy=va.margin.gt(0);out={}
    models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000)),
            'hist_gradient':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=50,l2_regularization=5,learning_rate=.05,random_state=42),
            'random_forest':RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=40,max_features=.8,random_state=42,n_jobs=2)}
    ridge=Ridge(alpha=100).fit(tr[COLS],tr.margin);s=np.sqrt(np.mean((tr.margin-ridge.predict(tr[COLS]))**2))
    pairs={'champion':(expit(ridge.predict(va[COLS])/(.6*s)),expit(ridge.predict(te[COLS])/(.6*s)))}
    for k,m in models.items():
        m.fit(tr[COLS],y);pairs[k]=(m.predict_proba(va[COLS])[:,1],m.predict_proba(te[COLS])[:,1])
    for k,(pv,pt) in pairs.items():
        cal=LogisticRegression().fit(logit(np.clip(pv,1e-6,1-1e-6)).reshape(-1,1),vy)
        out[k]=cal.predict_proba(logit(np.clip(pt,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    out['fixed_blend']=.5*out['champion']+.5*out['logistic']
    return out

def losses(y,p):
    p=np.clip(p,1e-6,1-1e-6);return -y*np.log(p)-(1-y)*np.log(1-p)
def cluster_ci(date,y,p,baseline):
    d=pd.DataFrame({'date':date,'delta':losses(y,p)-losses(y,baseline)}).groupby('date').delta.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(71);ix=rng.integers(len(d),size=(2000,len(d)));s=d[ix].sum(1);return np.quantile(s[:,0]/s[:,1],[.025,.975]).tolist()
def metric(y,p):
    return dict(n=len(y),accuracy=float(((p>=.5)==y).mean()),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))
def coverage_scores(f):
    m,s,c,tm,res=train_models(f);te=f[f.season==2026].copy();marg=m.predict(te[COLS]);total=tm.predict(te[['rolling_total']]);raw=expit(marg/(.6*s));p=c.predict_proba(logit(np.clip(raw,1e-6,1-1e-6)).reshape(-1,1))[:,1]
    intervals=[];totals=[];fail=0
    for i,(a,b,q) in enumerate(zip(marg,total,p)):
        try:d=score_draws(a,b,q,res,n=1200,seed=773+i)
        except ValueError:fail+=1;intervals.append([np.nan,np.nan]);totals.append([np.nan,np.nan]);continue
        intervals.append(np.quantile(d[:,0]-d[:,1],[.1,.9]));totals.append(np.quantile(d.sum(1),[.1,.9]))
    valid=np.isfinite(intervals).all(1)
    return dict(n=int(valid.sum()),failed=fail,margin_80_coverage=interval_coverage(te.margin.to_numpy()[valid],np.asarray(intervals)[valid]),total_80_coverage=interval_coverage(te.total.to_numpy()[valid],np.asarray(totals)[valid]),nominal=.8)
def coverage_players(p):
    # Generate calibration residuals from pregame predictions for 2024-25.
    old=p[p.Season<=2025].copy();old.loc[old.Season==2025,'Season']=2026
    _,cal=player_benchmark(old);_,te=player_benchmark(p);cols=['Pts','REB','AST','3PM'];rng=np.random.default_rng(42)
    pred=te[[k+'_opponent' for k in cols]].to_numpy();actual=te[[k+'_actual' for k in cols]].to_numpy();target=np.column_stack([actual,actual[:,:3].sum(1)])
    residual=cal[[k+'_actual' for k in cols]].to_numpy()-cal[[k+'_opponent' for k in cols]].to_numpy()
    cut=np.array([5,12,20]);bins=np.digitize(cal.Pts_opponent,cut);tb=np.digitize(pred[:,0],cut)
    lows=np.zeros_like(target);highs=np.zeros_like(target)
    for b in range(4):
        pool=residual[bins==b];inds=np.where(tb==b)[0]
        if len(pool)<30:pool=residual
        for start in range(0,len(inds),256):
            ix=inds[start:start+256];draw=np.maximum(0,np.rint(pred[ix,None,:]+pool[rng.integers(len(pool),size=(len(ix),500))]))
            draw[:,:,3]=np.minimum(draw[:,:,3],draw[:,:,0]//3);draw=np.concatenate([draw,draw[:,:,:3].sum(2,keepdims=True)],axis=2)
            lows[ix],highs[ix]=np.quantile(draw,[.1,.9],axis=1)
    return dict(n=len(te),calibration_appearances=len(cal),nominal=.8,coverage={k:float(((target[:,j]>=lows[:,j])&(target[:,j]<=highs[:,j])).mean()) for j,k in enumerate(cols+['PRA'])},mean_width={k:float((highs[:,j]-lows[:,j]).mean()) for j,k in enumerate(cols+['PRA'])},note='Conditional on appearance; residuals clustered by projected points, not player-specific')
def main():
    out=Path('reports/challengers');out.mkdir(parents=True,exist_ok=True);manifest=[];ts=[];ps=[]
    for y in HASHES:
        ts.append(normalize(fetch(y,'team_box',manifest),y,'team_box'));ps.append(normalize(fetch(y,'player_box',manifest),y,'player_box'))
    f,_=team_features(pd.concat(ts,ignore_index=True));p=pd.concat(ps,ignore_index=True);folds={};pred=[]
    for year in [2023,2024,2025,2026]:
        tr=f[f.season<year-1];va=f[f.season==year-1];te=f[f.season==year].copy();y=te.margin.gt(0).to_numpy(int);probs=fit_models(tr,va,te)
        folds[str(year)]={k:{**metric(y,v),'delta_log_loss_ci95_vs_champion':cluster_ci(te.date.to_numpy(),y,v,probs['champion'])} for k,v in probs.items()}
        for k,v in probs.items():te[k+'_prob']=v
        pred.append(te)
    result={'folds':folds,'game_intervals':coverage_scores(f),'player_intervals':coverage_players(p),'note':'All folds previously researched historical data. No novel untouched holdout, and no automatic promotion. Confidence intervals resample game dates; not full season/serial dependency.'}
    pd.concat(pred).to_csv(out/'predictions.csv',index=False);(out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
