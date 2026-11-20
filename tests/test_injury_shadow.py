import datetime as dt
import pandas as pd
import numpy as np
from axiom.injury_shadow import eligible_snapshots,side_vector
from axiom.official_injuries import index_reports
UTC=dt.timezone.utc
def test_t90_cutoff():
    day=pd.Timestamp('2026-11-21')
    g=dict(date=day,home='BOS',away='NYK',tipoff=dt.datetime(2026,11,22,0,tzinfo=UTC))
    # 5:00 p.m. ET = 22:00 UTC, 6:00 p.m. ET = 23:00 UTC, after T-90.
    a=pd.DataFrame([dict(game_date=day,report_datetime=f'2026-11-21 {hour}:00',home_team_abbr='BOS',away_team_abbr='NYK',team=t,reason='',person_id=1,current_status=status) for t in ['Boston Celtics','New York Knicks'] for hour,status in [(17,'Questionable'),(18,'Out')]])
    idx=index_reports(a)
    assert eligible_snapshots(idx,g,dt.datetime(2026,11,21,22,tzinfo=UTC)) is None
    s=eligible_snapshots(idx,g,dt.datetime(2026,11,21,23,0,tzinfo=UTC))
    assert s is not None and all(x.current_status.iloc[0]=='Questionable' for x in s)
    assert eligible_snapshots(idx,g,g['tipoff']) is None
def test_long_absence_not_double_counted():
    days=pd.date_range('2026-10-01',periods=12)
    p=pd.DataFrame([dict(Date=d,Team='BOS',player_id=1,min=30.,pts=20.,ast=4.,reb=5.) for d in days[:3]]+[dict(Date=d,Team='BOS',player_id=2,min=25.,pts=10.,ast=2.,reb=3.) for d in days])
    report=pd.DataFrame([dict(person_id=1,current_status='Out'),dict(person_id=2,current_status='Out')])
    x=side_vector(p,'BOS',pd.Timestamp('2026-10-15'),report,pd.Timestamp('2026-07-01'))
    assert x[0]==-25 and x[3]==1
