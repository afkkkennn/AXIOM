"""Joint empirical residual distributions; no independence assumption."""
import numpy as np

def score_draws(margin,total,prob,paired_residuals,n=10000,seed=42):
    if not 0<prob<1:raise ValueError('Probability strictly between zero and one')
    r=np.asarray(paired_residuals,dtype=float)
    if r.ndim!=2 or r.shape[1]!=2 or len(r)<30 or not np.isfinite(r).all():raise ValueError('Insufficient paired residuals')
    rng=np.random.default_rng(seed)
    # Both scoring outcomes derive from the same residual row; stratify on
    # calibrated win probability so projected winners and simulated winners agree.
    m=margin+r[:,0];t=total+r[:,1]
    home=np.maximum(0,np.rint((t+m)/2)).astype(int)
    away=np.maximum(0,np.rint((t-m)/2)).astype(int)
    home=home+(home==away)*(m>=0);away=away+(home==away)*(m<0)
    positive=np.flatnonzero(home>away);negative=np.flatnonzero(home<away)
    if len(positive)==0 or len(negative)==0:raise ValueError('Residuals lack both outcome scenarios')
    k=rng.binomial(n,prob);ix=np.r_[rng.choice(positive,k,replace=True),rng.choice(negative,n-k,replace=True)];rng.shuffle(ix)
    return np.column_stack([home[ix],away[ix]])

def summarize_scores(draws):
    d=np.asarray(draws);m=d[:,0]-d[:,1];t=d.sum(axis=1)
    return {'simulation_home_win':float((m>0).mean()),'home_mean':float(d[:,0].mean()),'away_mean':float(d[:,1].mean()),'margin_p10_p90':np.quantile(m,[.1,.9]).tolist(),'total_p10_p90':np.quantile(t,[.1,.9]).tolist()}

def player_draws(projection,residuals,n=5000,seed=42):
    p=np.asarray(projection,dtype=float);r=np.asarray(residuals,dtype=float)
    if p.shape!=(4,) or r.ndim!=2 or r.shape[1]!=4 or len(r)<30 or not np.isfinite(r).all():raise ValueError('Invalid joint player residuals')
    rng=np.random.default_rng(seed)
    # Order: points/rebounds/assists/3PM. Resample entire vector.
    x=np.maximum(0,np.rint(p+r[rng.integers(len(r),size=n)])).astype(int)
    x[:,3]=np.minimum(x[:,3],x[:,0]//3)
    return np.column_stack([x,x[:,:3].sum(axis=1)])

def interval_coverage(actual,intervals):
    a=np.asarray(actual);i=np.asarray(intervals)
    return float(((a>=i[:,0])&(a<=i[:,1])).mean())
