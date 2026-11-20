# AXIOM Research | 2.0-alpha.5 scaffold

Version-controlled NBA forecasting research, not an app or a validated replacement for alpha.2. This repository is currently public. Never commit secrets or personal information. Dates in source exports are source game dates; convert actual tipoff timestamps to Asia/Manila for daily reporting.

## Reproduce
GitHub Actions runs software tests and the pinned pilot on pushes. The pilot workflow downloads source data, verifies blob hashes, executes the chronological benchmark and uploads metrics, game/player forecasts and data manifest as artifacts. It also supports manual dispatch. Inspect workflow results rather than assuming success.

Optional local commands:
```sh
pip install -r requirements.txt
python -m pytest -q
python -m axiom.ingest
python -m axiom.backtest --team data/raw/pilot/team_boxscores.csv --player data/raw/pilot/boxscores.csv --test-start 2026-11-11
python -m axiom.settle
```

Pass --odds with a verified timestamped file conforming to data/odds_template.csv; use --validation-start with sufficient independent earlier calibration data. The expanding-window function axiom.team.walk_forward accepts successive cut dates.

## Modules
- axiom/ingest.py: pinned pilot downloader, manifest and hash verification.
- axiom/team.py: pregame rolling net rating and Four Factors, totals, chronological evaluation and optional earlier-period calibration.
- axiom/player.py: appearance-conditional average and rate projections.
- axiom/market.py: timestamped pregame paired-price benchmark.
- axiom/impact.py: scenario arithmetic, not a validated injury model.
- axiom/ledger.py: append-only pre-tipoff forecasts.
- axiom/settle.py: independently recorded results and model metrics.
- reports/prior_report.json: conversation-reported metrics, not independently reproduced.
- tests/: synthetic software checks.

## Limitations
Pinned source: https://github.com/jameslyman7/NBA/tree/20ecc62ad43b9f48fa63fc5460525c316fbbeb86/data/exports. Raw data is downloaded rather than redistributed, pending upstream licensing review. No original alpha.2 code/data, timestamped historical market archive, historical pregame injury archive, or multi-season player logs supplied. Current injury snapshot cannot stand in for historical availability. No automatic live forecast publishing. Do not claim verified superiority over alpha.2 or the market. Compare identical games on untouched chronological holdouts; keep calibration separate from testing, measure Brier, log loss, accuracy, margin and total MAE, and retain misses. Assess appearance and minutes separately for props; validate rotation and opponent adjustments and correlated PRA distributions before promotion. No wagering recommendation engine.

## Alpha.6 multi-season experiment
The pinned SportsDataverse ESPN box score files cover season ending years 2020–2026. Run `python -m axiom.historical`. Train on 2020–24, fit calibration only on 2025, evaluate 2026. Specifications fixed before first run. Compare net rating, Four Factors, Elo/rest and ensemble; player average, rate and opponent-adjusted rate. Artifacts include source hashes and individual forecasts. This is experimental until workflow results are inspected. Preserve the consumed 2026 holdout and validate any subsequent tuning on a new prospective period. Player forecasts are conditional on actual appearances, not an availability model.

### Verified alpha.6 benchmark
[Full metrics](reports/alpha6_holdout.json) from [successful workflow](https://github.com/afkkkennn/AXIOM/actions/runs/44682920282): 8,291 games, 1,230 holdout, calibrated ensemble accuracy 68.54%, Brier 0.20543, log loss 0.59738. Opponent-adjusted player model tested on 26,050 appearances. Results are not a head-to-head comparison with the missing alpha.2 model. Candidate for prospective tracking, not a proven market edge. Historical odds analysis and point-in-time injury validation remain separate. See reports/validation_plan.md for promotion gates.

## Alpha.7 operational research
- [Team forecast ledger](tracking/predictions.csv) and [team performance](tracking/performance.json).
- [Player forecast ledger](tracking/player_predictions.csv) and [player performance](tracking/player_performance.json); projections are appearance-conditional. Roster membership does not prove availability.
- Scheduled workflows run every six hours subject to GitHub scheduling and source availability. The first team/player runs recorded 9 and 261 prospective forecasts respectively, with results pending at the time of deployment.
- Joint empirical residual simulations and nonlinear challengers: [verified metrics](reports/alpha7_challengers.json). Alpha.6 calibrated ridge remains champion after retrospective comparisons. No new untouched season has been tested.
- Lagged participation and minutes research: [verified results](reports/alpha7_rotation.json), not a replacement for a point-in-time injury report.
- Matched market comparison: [verified metrics](reports/alpha7_matched_market.json), 553 matched games, market timestamps unverified; no claim of market superiority.
- Timestamped official injury archive research (SportsDataverse transcription) selects T-90 snapshots and explicitly skips unsubmitted/unknown reports. Results must be verified before integrating injury effects into the champion.
- Public inputs only; no private credentials. Repository visibility was public at deployment. No wagering/recommendation engine, no claim of guaranteed accuracy.

### Injury challenger
[Verified retrospective injury benchmark](reports/alpha7_injury_holdout.json): matched sample 1,145 games, injury challenger 69.17% accuracy vs 68.56% baseline; log-loss difference CI includes zero. No promotion yet. `axiom.injury_shadow` issues separately identified paired team-only and injury-aware forecasts using fresh T-90 report snapshots, explicitly skips missing snapshots, and uses the existing result settlement ledger. Workflow polls hourly during usual NBA game hours; the schedule is best effort, source updates may lag, and no eligible forecast is guaranteed. Compare only matched prospective pairs. No same-day scores. Test the live feature implementation and retain coverage counts before promotion.
