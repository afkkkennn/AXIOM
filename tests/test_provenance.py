from axiom.injury_archive import index_html,select
from axiom.results_feed import extract
import pytest

def test_injury_report_cutoff():
    base='https://ak-static.cms.nba.com/referee/injury/'
    html=''.join(f'<a href="{base}Injury-Report_2026-11-19_{h}_00PM.pdf">link</a>' for h in ['06','07','08'])
    r=index_html(html)
    assert len(r)==3
    assert select(r,'2026-11-20T00:00:00Z')['url'].endswith('06_00PM.pdf')
    assert select(r,'2026-11-19T22:00:00Z') is None

def test_results_verification():
    c={'status':{'type':{'completed':True,'name':'STATUS_FINAL'}},'competitors':[{'homeAway':'home','team':{'abbreviation':'GS'},'score':'120'},{'homeAway':'away','team':{'abbreviation':'BOS'},'score':'110'}]}
    e={'header':{'id':'123','competitions':[c]}}
    assert extract(e,'123','GSW','BOS')==(120,110)
    with pytest.raises(ValueError):extract(e,'123','NYK','BOS')
    c['status']['type']['completed']=False
    assert extract(e,'123','GSW','BOS') is None
