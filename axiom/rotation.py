"""Explicit rotation scenario. Requires user/analyst-supplied baseline and status.
Not a causal or historically validated estimate of an injury effect."""
import numpy as np
def redistribute(baseline_minutes, unavailable, caps=None, weights=None):
    names=list(baseline_minutes);x=np.array([baseline_minutes[k] for k in names],dtype=float)
    if not np.isfinite(x).all() or (x<0).any() or (x>48).any() or not np.isclose(x.sum(),240):raise ValueError('Supply complete 240-minute regulation baseline')
    if not set(unavailable)<=set(names):raise ValueError('Unknown unavailable player')
    cap=np.array([(caps or {}).get(k,48) for k in names],float)
    if not np.isfinite(cap).all() or (cap<0).any() or (cap>48).any():raise ValueError('Invalid cap')
    mask=np.array([k not in unavailable for k in names]);cap=cap*mask
    if cap.sum()<240-1e-8:raise ValueError('Insufficient eligible minutes')
    z=np.minimum(x,cap);w=np.array([(weights or {}).get(k,max(baseline_minutes[k],1)) for k in names],float)
    if not np.isfinite(w).all() or (w<0).any():raise ValueError('Invalid replacement weights')
    for _ in range(len(names)+2):
        need=240-z.sum()
        if need<1e-8:break
        room=cap-z;active=(room>1e-8)&(w>0)
        if not active.any():raise ValueError('No eligible replacement minutes')
        add=np.zeros_like(z);add[active]=need*w[active]/w[active].sum();z+=np.minimum(add,room)
    if not np.isclose(z.sum(),240):raise ValueError('Unsatisfied team minutes')
    return {k:float(v) for k,v in zip(names,z)}
def scenario_margin(baseline_minutes,scenario_minutes,relative_impact,possessions=100):
    if set(baseline_minutes)!=set(scenario_minutes) or set(relative_impact)!=set(baseline_minutes):raise ValueError('Impact needed for each rotation player')
    if not np.isclose(sum(baseline_minutes.values()),240) or not np.isclose(sum(scenario_minutes.values()),240):raise ValueError('Regulation minutes must sum to 240')
    if possessions<=0:raise ValueError('Invalid possessions')
    return sum((scenario_minutes[k]-baseline_minutes[k])*relative_impact[k] for k in baseline_minutes)/48*possessions/100
