"""Append pre-tip-off forecasts, never use this module to backfill live records."""
import argparse,csv,datetime as dt,json
from pathlib import Path
FIELDS=['forecast_id','created_at_utc','tipoff_at_utc','game_id','model_version','home','away','home_probability','projected_margin','projected_total','data_as_of_utc','notes']
def timestamp(x):
    z=dt.datetime.fromisoformat(x.replace('Z','+00:00'))
    if z.tzinfo is None:raise ValueError('Timezone required')
    return z

def append(path,row,now=None):
    now=now or dt.datetime.now(dt.timezone.utc);tip=timestamp(row['tipoff_at_utc']);asof=timestamp(row['data_as_of_utc']);p=float(row['home_probability'])
    if not 0<p<1 or not asof<=now<tip:raise ValueError('Forecast must be dated before tipoff and after its data cutoff')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with path.open(newline='') as f:
            if any(r['forecast_id']==row['forecast_id'] for r in csv.DictReader(f)):raise ValueError('Duplicate forecast ID')
    first=not path.exists() or path.stat().st_size==0
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader() if first else None;w.writerow({**row,'created_at_utc':now.isoformat()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('forecast_json');p.add_argument('--ledger',default='tracking/predictions.csv');a=p.parse_args();append(a.ledger,json.loads(Path(a.forecast_json).read_text()))
