import pandas as pd
import pytest
from axiom.settle import evaluate
def test_paired_injury_comparison():
    rows=[]
    for game,version,p in [('a','alpha.7-matched-control-T90',.55),('a','alpha.7-injury-shadow-T90',.7),('b','alpha.7-matched-control-T90',.8)]:
        rows.append(dict(forecast_id=game+version,game_id=game,model_version=version,home_probability=p,projected_margin=5,projected_total=220,tipoff_at_utc='2026-11-21T00:00:00Z'))
    actual=pd.DataFrame([dict(game_id=g,home_score=115,away_score=105) for g in ['a','b']])
    m=evaluate(pd.DataFrame(rows),actual)['paired_injury_comparison']
    assert m['matched_games']==1
    assert m['injury_brier']==pytest.approx(.09)
    assert m['delta_log_loss_injury_minus_control']<0
