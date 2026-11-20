import pytest
from axiom.market_matched import canonical
def test_team_aliases():
    assert canonical('Golden State Warriors')=='GSW'
    assert canonical('LA Clippers')=='LAC'
    assert canonical('LAL')=='LAL'
    assert canonical('Los Angeles') is None
