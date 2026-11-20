import pytest
from axiom.rotation import redistribute,scenario_margin
from axiom.live_players import roster_players,injury_players,conditional_projection

def test_minutes_conservation():
    b={str(i):30 for i in range(8)};r=redistribute(b,['0'],{str(i):40 for i in range(8)})
    assert r['0']==0 and sum(r.values())==pytest.approx(240) and max(r.values())<=40
    assert scenario_margin(b,r,{k:4 if k=='0' else 0 for k in b})==pytest.approx(-2.5)
def test_roster_injury():
    assert roster_players({'athletes':[{'items':[{'id':'1','displayName':'Player','position':{'abbreviation':'G'}}]}]})=={'1':'Player'}
    assert injury_players({'injuries':[{'injuries':[{'athlete':{'id':'1'},'status':'Out'}]}]})=={'1':'Out'}
def test_conditional_projection():
    m,p=conditional_projection([[30,20,5,4,2]]*5,[],[])
    assert m==30 and list(p)==pytest.approx([20,5,4,2])
