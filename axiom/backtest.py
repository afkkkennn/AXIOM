import argparse,json
from pathlib import Path
import pandas as pd
from axiom.team import run_split
from axiom.player import project
from axiom.market import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--team',required=True);p.add_argument('--player');p.add_argument('--odds');p.add_argument('--test-start',default='2026-11-11');p.add_argument('--validation-start');p.add_argument('--out',default='reports/generated');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    metrics,pred=run_split(pd.read_csv(a.team),a.test_start,a.validation_start);pred.to_csv(out/'team_predictions.csv',index=False)
    if a.player:
        metrics['players'],pp=project(pd.read_csv(a.player),a.test_start);pp.to_csv(out/'player_predictions.csv',index=False)
    if a.odds:metrics['market']=compare(pred,pd.read_csv(a.odds))
    else:metrics['market_status']='Not tested: no verified timestamped odds archive supplied'
    metrics['injury_status']='Scenario calculator only; historical injury-impact backtest requires point-in-time inputs'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
