import numpy as np
import pandas as pd
import pytest
from axiom.simulation import score_draws,player_draws
from axiom.live import dummy_pair

def test_correlated_scores():
    rng=np.random.default_rng(42);r=rng.multivariate_normal([0,0],[[144,30],[30,225]],size=1000)
    d=score_draws(4,220,.65,r,n=10000)
    assert abs((d[:,0]>d[:,1]).mean()-.65)<.025
    assert (d>=0).all() and (d[:,0]!=d[:,1]).all()

def test_joint_player():
    rng=np.random.default_rng(42);r=rng.normal(size=(200,4))*3
    x=player_draws([18,5,4,2],r)
    assert (x>=0).all() and np.all(x[:,4]==x[:,:3].sum(1)) and np.all(x[:,3]*3<=x[:,0])

def test_dummy_pair():
    g=dict(date=pd.Timestamp('2026-11-21'),home='BOS',away='NYK')
    d=dummy_pair(g,2027)
    assert len(d)==2 and d[0]['Match up']=='BOS vs. NYK' and d[1]['Match up']=='NYK @ BOS'
