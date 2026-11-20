"""Scenario arithmetic only: not fitted or validated player impact coefficients."""

def margin_delta(baseline_minutes, scenario_minutes, replacement_relative_impact, possessions=100):
    if not (0<=baseline_minutes<=48 and 0<=scenario_minutes<=48 and possessions>0):raise ValueError('Invalid minutes/possessions')
    return (scenario_minutes-baseline_minutes)/48*possessions/100*replacement_relative_impact

def expected_delta(baseline_minutes, minutes_if_active, availability_probability, replacement_relative_impact, possessions=100):
    if not 0<=availability_probability<=1:raise ValueError('Invalid availability probability')
    return margin_delta(baseline_minutes,minutes_if_active*availability_probability,replacement_relative_impact,possessions)
