"""Reconstruct the opening race to 10 from ESPN's chronological event-level plays.

Defaults to Detroit's most recent 20 completed competitive games, including
previous-season playoffs if fewer than 20 current regular-season games exist.
Reports missing/ambiguous PBP separately, never as an inferred loss.
"""
import argparse,datetime as dt,json,time
from pathlib import Path
import requests

API="https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
FINAL={"STATUS_FINAL","STATUS_FINAL_OVERTIME"}
TEAM_ID={"DET":"8"}
UTC=dt.timezone.utc

def api(path,**params):
    r=requests.get(API+path,params=params,timeout=(15,45),headers={"User-Agent":"AXIOM-research (public ESPN scoring replay)"})
    r.raise_for_status()
    return r.json(),r.url

def is_final(status):
    if not isinstance(status,dict):return False
    v=status.get("type",status)
    return bool(v.get("completed",False)) and v.get("name") in FINAL

def schedule(team,season,kind):
    resp,url=api(f"/teams/{TEAM_ID[team]}/schedule",season=season,seasontype=kind,limit=250)
    events=resp.get("events",[])
    out=[]
    for e in events:
        try:
            comp=e["competitions"][0]
            status=e.get("status",comp.get("status",{}))
            if not is_final(status):continue
            t={x["homeAway"]:x["team"]["abbreviation"] for x in comp["competitors"]}
            if team not in t.values():continue
            moment=dt.datetime.fromisoformat(e["date"].replace("Z","+00:00"))
            out.append({"id":str(e["id"]),"tipoff_utc":moment,"home":t["home"],"away":t["away"],
                "opponent":t["away"] if t["home"]==team else t["home"],
                "location":"home" if t["home"]==team else "away",
                "season":season,"competition":"regular season" if kind==2 else "playoffs",
                "schedule_url":url})
        except (KeyError,IndexError,TypeError) as exc:
            print("Skipping malformed schedule event:",e.get("id"),str(exc))
    return out

def plays_from_summary(s):
    p=s.get("plays",[])
    if isinstance(p,dict):p=p.get("items",p.get("plays",[]))
    return p if isinstance(p,list) else []

def race(game):
    data,url=api("/summary",event=game["id"])
    p=plays_from_summary(data)
    if not p:return {"status":"unverified","reason":"No play-by-play events in ESPN summary","url":url}
    header=data.get("header",{});cs=header.get("competitions",[])
    if not cs or str(header.get("id"))!=game["id"] or not is_final(cs[0].get("status",{})):
        return {"status":"unverified","reason":"Inconsistent game header or final status","url":url}
    teams={x["homeAway"]:x["team"]["abbreviation"] for x in cs[0].get("competitors",[])}
    if teams.get("home")!=game["home"] or teams.get("away")!=game["away"]:
        return {"status":"unverified","reason":"Team mismatch between schedule and summary","url":url}
    q1=[]
    for i,x in enumerate(p):
        period=x.get("period",{})
        period=period.get("number",period) if isinstance(period,dict) else period
        if str(period)!="1":continue
        try:h=int(x["homeScore"]);a=int(x["awayScore"])
        except (KeyError,TypeError,ValueError):continue
        q1.append((i,x,h,a))
    if not q1:
        return {"status":"unverified","reason":"No first-quarter scoreboard data","url":url,"plays_count":len(p)}
    # Only the scoring sequence up to the first team reaching ten matters.
    # Later post-review corrections cannot alter an already-resolved race.
    prev_h,prev_a=0,0
    for _,x,h,a in q1:
        if h<prev_h or a<prev_a:
            return {"status":"unverified","reason":"Nonmonotone score BEFORE the race was decided","url":url,
                    "previous_score":[prev_h,prev_a],"current_score":[h,a],"clock":str(x.get("clock"))}
        prev_h,prev_a=h,a
        if h>=10 and a>=10:
            return {"status":"unverified","reason":"Both reached ten in the same score update","url":url}
        if h>=10 or a>=10:
            winner=game["home"] if h>=10 else game["away"]
            return {"status":"verified","first_to_10":winner,"det_won":winner=="DET",
                "score_at_race_decision":f"{game['away']} {a} - {game['home']} {h}",
                "quarter_clock":x.get("clock",{}).get("displayValue") if isinstance(x.get("clock"),dict) else str(x.get("clock", "")),
                "event":x.get("text",""),"url":url,"q1_scored_events":len(q1),"plays_count":len(p)}
    return {"status":"unverified","reason":"Neither team reached ten in available Q1 play list","url":url}

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--team",default="DET",choices=["DET"])
    pa.add_argument("--count",type=int,default=20)
    pa.add_argument("--as-of-utc",default="2026-11-22T13:29:00+00:00")
    pa.add_argument("--out",default="reports/race_to_10_det_last20.json")
    a=pa.parse_args()
    as_of=dt.datetime.fromisoformat(a.as_of_utc.replace("Z","+00:00"))
    if as_of.tzinfo is None:raise ValueError("as-of must contain timezone")
    all_games=[]
    for season,kind in [(2027,2),(2026,3),(2026,2),(2025,3)]:
        part=schedule(a.team,season,kind)
        print(f"schedule {season} type={kind}: {len(part)} completed")
        all_games.extend(part)
        if len({x["id"] for x in all_games if x["tipoff_utc"]<as_of})>=a.count:break
    unique={x["id"]:x for x in all_games if x["tipoff_utc"]<as_of}
    games=sorted(unique.values(),key=lambda x:x["tipoff_utc"],reverse=True)[:a.count]
    if len(games)<a.count:raise ValueError(f"Only {len(games)} completed games found, expected {a.count}")
    records=[]
    for g in games:
        try:res=race(g)
        except requests.RequestException as ex:
            res={"status":"unverified","reason":f"ESPN fetch failed: {str(ex)[:160]}"}
        record={**{k:v for k,v in g.items() if k!="schedule_url"},**res,
                "tipoff_utc":g["tipoff_utc"].isoformat()}
        records.append(record)
        print(g["tipoff_utc"].date(),g["opponent"],g["location"],res.get("first_to_10", "N/A"),res.get("score_at_race_decision",""),res.get("reason",""))
        time.sleep(.15)
    wins=sum(x.get("det_won",False) for x in records)
    loss=sum(x.get("status")=="verified" and not x["det_won"] for x in records)
    unverified=sum(x["status"]!="verified" for x in records)
    report={"team":a.team,"as_of_utc":as_of.isoformat(),"window":len(records),
        "verified":len(records)-unverified,"race_to_10_wins":wins,"race_to_10_losses":loss,
        "unverified":unverified,"win_pct_verified":round(100*wins/(wins+loss),2) if wins+loss else None,
        "window_current_season_games":sum(x["season"]==2027 for x in records),
        "method":"First chronological Q1 ESPN play with either homeScore or awayScore >=10; winner is side already >=10 while opposing side still below 10. Completed NBA regular season then prior playoffs and prior regular season as required; preseason excluded.",
        "limitations":"This is historical frequency, not forecast probability; absence of PBP is unverified, never counted as a loss.",
        "games":records}
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2))
    print("SUMMARY\n"+json.dumps({k:v for k,v in report.items() if k!="games"},indent=2))
    print("OUTPUT "+str(target))
    if unverified: print(f"WARNING: {unverified} games could not be verified")
if __name__=="__main__":main()
