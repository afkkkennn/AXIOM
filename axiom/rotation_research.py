"""Lagged roster/appearance forecasting. Not contemporaneous injury knowledge.
Construct candidate players entirely from past appearances; evaluate missing returns too.
"""
from collections import defaultdict,deque
from itertools import groupby
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss,log_loss
from axiom.historical import pair_games,HASHES,fetch,normalize
FEATURES=['past3_rate','past10_rate','last_appearance','days_since','minutes_when_active','days_rest']
def make_rows(team,player):
    games,_=pair_games(team)
    groups={(str(gid),t):dict(zip(g['Player ID'].astype(str),g.Min.astype(float))) for (gid,t),g in player.groupby(['game_id','Team'],sort=False)}
    history=defaultdict(lambda:deque(maxlen=10));last_team={};rows=[];coverage=defaultdict(lambda:[0,0,0,0]);season=None
    for date,day in groupby(games,key=lambda x:x[0]):
        day=list(day);sy=int(day[0][3].Season)
        if sy!=season:
            history.clear();season=sy
        pending=[]
        for _,h,a,hr,ar in day:
            for t,record in [(h,hr),(a,ar)]:
                observed=groups.get((str(record.game_id),t))
                if not observed or not 225<=sum(observed.values())<=350:
                    coverage[sy][3]+=1;continue
                hist=list(history[t]);past=len(hist)
                if past>=5:
                    candidates={k for _,minutes in hist for k in minutes if last_team.get(k)==t}
                    coverage[sy][0]+=len(observed);coverage[sy][1]+=sum(k in candidates for k in observed);coverage[sy][2]+=1
                    for pid in candidates:
                        a=np.array([v.get(pid,0) for _,v in hist]);played=a>0;ids=np.flatnonzero(played)
                        if not len(ids):continue
                        rec=a[-5:];last_active=hist[ids[-1]][0]
                        chunks=a[played];expected=.6*chunks[-5:].mean()+.4*chunks.mean()
                        rows.append(dict(date=date,season=sy,team=t,game_id=str(record.game_id),player_id=pid,
                            past3_rate=float(played[-3:].mean()),past10_rate=float(played.mean()),last_appearance=int(played[-1]),
                            days_since=min(40,(date-last_active).days),minutes_when_active=expected,
                            days_rest=min(7,(date-hist[-1][0]).days),baseline_probability=(played.sum()+1)/(past+2),
                            baseline_minutes=float(rec.mean()),appeared=int(pid in observed),actual_minutes=float(observed.get(pid,0))))
                pending.append((t,observed))
        for t,obs in pending:
            history[t].append((date,obs))
            for pid in obs:last_team[pid]=t
    return pd.DataFrame(rows),dict(coverage)
def evaluate(rows,coverage):
    tr=rows[rows.season<=2024];va=rows[rows.season==2025];te=rows[rows.season==2026].copy()
    if min(len(tr),len(va),len(te))<1000:raise ValueError('Insufficient candidate records')
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000)).fit(tr[FEATURES],tr.appeared)
    vp=m.decision_function(va[FEATURES]);cal=LogisticRegression().fit(vp.reshape(-1,1),va.appeared)
    test_p=cal.predict_proba(m.decision_function(te[FEATURES]).reshape(-1,1))[:,1]
    te['learned_probability']=test_p;te['learned_minutes']=test_p*te.minutes_when_active
    def metric(p,mins):
        return dict(brier=float(brier_score_loss(te.appeared,p)),log_loss=float(log_loss(te.appeared,p,labels=[0,1])),
                    unconditional_minutes_mae=float(np.abs(te.actual_minutes-mins).mean()))
    c=coverage[2026]
    return dict(train=len(tr),calibration=len(va),test=len(te),baseline=metric(te.baseline_probability,te.baseline_minutes),
                learned=metric(te.learned_probability,te.learned_minutes),candidate_appearance_coverage=c[1]/c[0],
                observed_appearances=c[0],games_team_sides_evaluated=c[2],incomplete_team_sides_excluded=c[3],
                limitations=['Candidate set from last ten team games at least five earlier games in current season','Appearance record is not verified pregame availability','Unseen returns/trades may be missed','No official injury snapshot used','Reuses historical 2025-26 holdout; not a fresh test']),te
def main():
    manifest=[];ts=[];ps=[]
    for y in HASHES:
        ts.append(normalize(fetch(y,'team_box',manifest),y,'team_box'));ps.append(normalize(fetch(y,'player_box',manifest),y,'player_box'))
    r,c=make_rows(pd.concat(ts,ignore_index=True),pd.concat(ps,ignore_index=True));m,p=evaluate(r,c)
    out=Path('reports/rotation_research');out.mkdir(parents=True,exist_ok=True);p.to_csv(out/'predictions.csv',index=False);(out/'metrics.json').write_text(json.dumps(m,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
