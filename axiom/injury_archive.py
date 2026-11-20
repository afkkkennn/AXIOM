"""Index official NBA injury report links; select only report times before tipoff."""
import argparse,datetime as dt,hashlib,json,re
from zoneinfo import ZoneInfo
from urllib.parse import urljoin,urlparse
from html.parser import HTMLParser
from pathlib import Path
import requests
ET=ZoneInfo('America/New_York')
PAT=re.compile(r'Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})_(\d{2})(AM|PM)\.pdf$',re.I)
INDEX='https://official.nba.com/nba-injury-report-2026-27-season/'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            url=dict(attrs).get('href')
            if url:self.urls.append(urljoin(INDEX,url))
def index_html(html):
    parser=Links();parser.feed(html);rows={}
    for u in parser.urls:
        q=urlparse(u)
        if q.scheme!='https' or q.hostname!='ak-static.cms.nba.com':continue
        m=PAT.search(q.path)
        if not m:continue
        day,h,minute,ap=m.groups()
        t=dt.datetime.strptime(f'{day} {h}:{minute}{ap.upper()}','%Y-%m-%d %I:%M%p').replace(tzinfo=ET)
        utc=t.astimezone(dt.timezone.utc).isoformat()
        rows[u]=dict(url=u,report_time_utc=utc)
    return sorted(rows.values(),key=lambda r:r['report_time_utc'])
def select(rows,tipoff,max_age_hours=36):
    tip=dt.datetime.fromisoformat(tipoff.replace('Z','+00:00'))
    if tip.tzinfo is None:raise ValueError('Timezone required')
    eligible=[r for r in rows if dt.timedelta(0)<tip-dt.datetime.fromisoformat(r['report_time_utc'])<=dt.timedelta(hours=max_age_hours)]
    return max(eligible,key=lambda r:r['report_time_utc']) if eligible else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--tipoff');p.add_argument('--out',default='reports/live/injury_index.json');args=p.parse_args()
    r=requests.get(INDEX,timeout=(15,60));r.raise_for_status();rows=index_html(r.text)
    if not rows:raise ValueError('No official PDF links returned; do not infer availability')
    doc={'index_url':INDEX,'retrieved_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'reports':rows,'note':'Report times parsed from official filenames; game-specific status and report completeness still require inspection. Archive link presence is not proof of historical retrieval time.'}
    if args.tipoff:doc['selected']=select(rows,args.tipoff)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2));print(f'Indexed {len(rows)} official report links')
if __name__=='__main__':main()
