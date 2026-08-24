"""Walk-forward backtest: train on past seasons, evaluate on the next season,
roll forward. Compares Poisson / logreg / XGBoost against the bookmaker's
de-vigged line, and simulates fractional-Kelly staking on flagged edges.
"""
import pathlib

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from src.models import logreg, poisson, xgb
from src.odds import devig_two_way

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "matches.duckdb"
CONFIG = yaml.safe_load((ROOT / "config" / "leagues.yaml").read_text())

FEATURE_COLS = None  # populated at runtime from the features table


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("home_", "away_")) or c == "h2h_goal_avg"]


def _season_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["season"].unique())


def kelly_stake(model_prob: float, odds: float, fraction: float, max_stake_pct: float) -> float:
    b = odds - 1
    q = 1 - model_prob
    f_star = (b * model_prob - q) / b
    stake = max(0.0, f_star) * fraction
    return min(stake, max_stake_pct)


def run():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM features").fetchdf()
    con.close()

    df = df.dropna(subset=["B365>2.5", "B365<2.5"]).reset_index(drop=True)
    feat_cols = _feature_cols(df)
    seasons = _season_order(df)

    edge_threshold = CONFIG["edge_threshold"]
    kelly_fraction = CONFIG["kelly_fraction"]
    max_stake_pct = CONFIG["max_stake_pct"]
    stop_loss_pct = CONFIG["stop_loss_pct"]

    results = []
    bankroll = 1.0
    peak = 1.0
    max_drawdown = 0.0
    stopped = False

    # need a few seasons of history before the first walk-forward test
    for i, test_season in enumerate(seasons[3:], start=3):
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == test_season].copy()
        if test.empty or train.empty:
            continue

        pois_fit = poisson.fit(train[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
        test["p_poisson"] = poisson.predict_over_prob(pois_fit, test)

        lr = logreg.fit(train, feat_cols, "over_2_5")
        test["p_logreg"] = logreg.predict_over_prob(lr, test, feat_cols)

        xg = xgb.fit(train, feat_cols, "over_2_5")
        test["p_xgb"] = xgb.predict_over_prob(xg, test, feat_cols)

        fair = devig_two_way(test["B365>2.5"], test["B365<2.5"])
        test = pd.concat([test, fair], axis=1)

        for model_name in ["p_poisson", "p_logreg", "p_xgb"]:
            ll = log_loss(test["over_2_5"], test[model_name].clip(1e-6, 1 - 1e-6))
            bs = brier_score_loss(test["over_2_5"], test[model_name])
            results.append({"season": test_season, "model": model_name, "log_loss": ll, "brier": bs})

        market_ll = log_loss(test["over_2_5"], test["p_over_fair"].clip(1e-6, 1 - 1e-6))
        market_bs = brier_score_loss(test["over_2_5"], test["p_over_fair"])
        results.append({"season": test_season, "model": "market_fair", "log_loss": market_ll, "brier": market_bs})

        # staking simulation using the XGBoost model against market odds
        if not stopped:
            for _, row in test.sort_values("Date").iterrows():
                model_p = row["p_xgb"]
                mkt_p = row["p_over_fair"]
                mkt_edge = model_p - mkt_p
                if abs(mkt_edge) < edge_threshold:
                    continue
                side_odds = row["B365>2.5"] if mkt_edge > 0 else row["B365<2.5"]
                side_prob = model_p if mkt_edge > 0 else (1 - model_p)
                stake_frac = kelly_stake(side_prob, side_odds, kelly_fraction, max_stake_pct)
                if stake_frac <= 0:
                    continue
                stake = bankroll * stake_frac
                won = (row["over_2_5"] == 1) if mkt_edge > 0 else (row["over_2_5"] == 0)
                bankroll += stake * (side_odds - 1) if won else -stake
                peak = max(peak, bankroll)
                drawdown = (peak - bankroll) / peak
                max_drawdown = max(max_drawdown, drawdown)
                if drawdown >= stop_loss_pct:
                    stopped = True
                    break

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("model")[["log_loss", "brier"]].mean().sort_values("log_loss")

    print("=== Calibration (lower is better), averaged across walk-forward seasons ===")
    print(summary.to_string())
    print()
    print(f"=== Staking simulation (XGBoost, edge > {edge_threshold:.0%}, "
          f"{kelly_fraction:.0%} Kelly, max {max_stake_pct:.0%}/bet, "
          f"stop-loss {stop_loss_pct:.0%}) ===")
    print(f"Final bankroll: {bankroll:.3f}x starting  |  Max drawdown: {max_drawdown:.1%}"
          f"  |  Stopped early: {stopped}")

    return results_df, summary, bankroll, max_drawdown


if __name__ == "__main__":
    run()
