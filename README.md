# football-value-model

Probabilistic forecasting & backtesting system for football betting markets —
goals over/under and match result (1X2). Compares an independent-Poisson
goal model against logistic regression and XGBoost, benchmarks against
de-vigged bookmaker odds (Bet365, Pinnacle), and backtests fractional-Kelly
staking with proper walk-forward validation.

**[Read the full write-up →](docs/report.html)** (open in a browser)

## Results

Walk-forward log loss, averaged across seasons (lower is better):

| Market | Market (fair) | Best model |
|---|---|---|
| Goals over/under 2.5 (Bet365) | **0.668** | 0.681 (logreg) |
| Match result 1X2 (Pinnacle) | **0.967** | 0.998 (logreg) |

The de-vigged market beats every model on every market tested — expected,
given Pinnacle and Bet365's goals line are among the sharpest prices in
sports betting. See the write-up for the full read on why that's the actual
finding, not a failure.

## Pipeline stages

1. `src/ingest.py` — pull historical CSVs from football-data.co.uk into DuckDB
2. `src/features.py` — rolling form, home/away splits, referee tendencies, H2H
3. `src/models/` — poisson (independent-Poisson goal model), logreg, xgb
4. `src/odds.py` — de-vig bookmaker odds (two-way and three-way markets)
5. `src/backtest.py` — walk-forward eval on goals O/U 2.5 vs Bet365
6. `src/backtest_1x2.py` — walk-forward eval on match result vs Pinnacle,
   including fractional-Kelly staking simulation with a hard stop-loss

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
