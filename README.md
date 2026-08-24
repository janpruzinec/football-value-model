# football-value-model

Probabilistic forecasting & backtesting system for football over/under markets
(goals, corners, cards). Compares a Dixon-Coles Poisson baseline against
logistic regression and XGBoost, benchmarks against bookmaker implied
probability, and backtests staking strategy (fractional Kelly) with proper
walk-forward validation.

## Pipeline stages

1. `src/ingest.py` — pull historical CSVs from football-data.co.uk into DuckDB
2. `src/features.py` — rolling form, home/away splits, referee tendencies, H2H
3. `src/models/` — poisson (Dixon-Coles), logreg, xgb
4. `src/odds.py` — de-vig bookmaker odds, compute edge vs model probability
5. `src/backtest.py` — walk-forward evaluation: log loss, Brier score, ROI, drawdown
6. `src/staking.py` — fractional Kelly bankroll rules

Live execution (Betfair/Pinnacle API) is a later, separate, optional stage —
not required for the core project.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.ingest      # download + load raw data
python -m src.features    # build feature table
python -m src.backtest    # train + evaluate, walk-forward by season
```

Config (leagues, seasons, edge threshold) lives in `config/leagues.yaml`.
