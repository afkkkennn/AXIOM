import datetime as dt
import numpy as np
import pandas as pd
import pytest
from axiom.team import features,run_split,walk_forward
from axiom.player import project
from axiom.market import compare
from axiom.impact import expected_delta
from axiom.ledger import append

def teams():
    rows=[]
    for i,date in enumerate(pd.date_range('2026-10-01',periods=35)):
        home,away=('A','B') if i%2==0 else ('B','A')
        for t,o,s,flag in [(home,away,110+i%7,'vs.'),(away,home,104+i%5,'@')]:
            rows.append({'Team':t,'Match up':f'{t} {flag} {o}','Date':date,'Pts':s,'FGM':40,'FGA':85,'3PM':12,'FTM':18,'FTA':20,'OREB':10,'DREB':30,'TOV':12})
    return pd.DataFrame(rows)

def test_no_current_game_leakage():
    raw=teams();before=features(raw);changed=raw.copy();changed.loc[changed.Date==pd.Timestamp('2026-10-20'),'Pts']+=100;after=features(changed)
    cols=['net','efg','turnovers','oreb','ftr','rolling_total'];pd.testing.assert_frame_equal(before.loc[before.date<='2026-10-20',cols],after.loc[after.date<='2026-10-20',cols])

def test_backtest_and_walkforward():
    m,p=run_split(teams(),'2026-10-21');assert m['split']['train']==20 and len(p)==15;assert p.four_factors_prob.between(0,1).all();folds=walk_forward(teams(),['2026-10-21','2026-11-01']);assert len(folds)==2;assert sum(len(x[1]) for x in folds)==15

def test_player():
    p=pd.DataFrame([{'Date':d,'Player ID':1,'Min':30,'Pts':20,'REB':5,'AST':4,'3PM':2} for d in pd.date_range('2026-10-01',periods=5)]);m,_=project(p,'2026-10-04');assert m['n']==2 and m['PRA']['rate']==pytest.approx(0,abs=1e-10)

def test_market_rejects_late():
    q=pd.DataFrame([dict(date='2026-10-21',home='A',away='B',home_decimal=1.8,away_decimal=2.1,observed_at='2026-10-21T23:00:00Z',tipoff_at='2026-10-21T22:00:00Z')]);_,p=run_split(teams(),'2026-10-21')
    with pytest.raises(ValueError):compare(p,q)

def test_impact_no_double_count():
    assert expected_delta(0,32,0,5)==0;assert expected_delta(32,32,0,5)==pytest.approx(-3.3333333333)

def test_ledger(tmp_path):
    row=dict(forecast_id='one',tipoff_at_utc='2026-11-22T00:00:00Z',data_as_of_utc='2026-11-21T00:00:00Z',game_id='1',model_version='test',home='A',away='B',home_probability=.7,projected_margin=5,projected_total=225,notes='synthetic');now=dt.datetime(2026,11,21,12,tzinfo=dt.timezone.utc);f=tmp_path/'ledger.csv';append(f,row,now)
    with pytest.raises(ValueError):append(f,row,now)
    with pytest.raises(ValueError):append(f,{**row,'forecast_id':'two'},dt.datetime(2026,11,23,tzinfo=dt.timezone.utc))
