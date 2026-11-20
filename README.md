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
