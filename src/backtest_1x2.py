"""Walk-forward backtest on the match-result (1X2) market, benchmarked
against Pinnacle (the sharpest publicly-tracked line) rather than Bet365.
"""
import pathlib

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from src.models import logreg, poisson, xgb
from src.odds import devig_three_way

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "matches.duckdb"
CONFIG = yaml.safe_load((ROOT / "config" / "leagues.yaml").read_text())

RESULT_MAP = {"H": 0, "D": 1, "A": 2}
SIDE_COLS = {0: ("PSH", "p_h_fair"), 1: ("PSD", "p_d_fair"), 2: ("PSA", "p_a_fair")}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("home_", "away_")) or c == "h2h_goal_avg"]


def kelly_stake(model_prob: float, odds: float, fraction: float, max_stake_pct: float) -> float:
    b = odds - 1
    q = 1 - model_prob
    f_star = (b * model_prob - q) / b
    return min(max(0.0, f_star) * fraction, max_stake_pct)


def _normalize(proba: np.ndarray) -> np.ndarray:
    return proba / proba.sum(axis=1, keepdims=True)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def run():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM features").fetchdf()
    con.close()

    df = df.dropna(subset=["PSH", "PSD", "PSA", "FTR"]).reset_index(drop=True)
    df["result"] = df["FTR"].map(RESULT_MAP)
    feat_cols = _feature_cols(df)
    seasons = sorted(df["season"].unique())

    edge_threshold = CONFIG["edge_threshold"]
    kelly_fraction = CONFIG["kelly_fraction"]
    max_stake_pct = CONFIG["max_stake_pct"]
    stop_loss_pct = CONFIG["stop_loss_pct"]

    results = []
    bankroll, peak, max_drawdown, stopped = 1.0, 1.0, 0.0, False

    for i, test_season in enumerate(seasons[3:], start=3):
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == test_season].copy()
        if test.empty or train.empty:
            continue

        pois_fit = poisson.fit(train[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
        pois_p = poisson.predict_result_probs(pois_fit, test)

        lr = logreg.fit(train, feat_cols, "result")
        lr_p = logreg.predict_proba(lr, test, feat_cols)[[0, 1, 2]]

        xg = xgb.fit(train, feat_cols, "result")
        xg_p = xgb.predict_proba(xg, test, feat_cols)[[0, 1, 2]]

        fair = devig_three_way(test["PSH"], test["PSD"], test["PSA"])
        test = pd.concat([test.reset_index(drop=True), fair.reset_index(drop=True)], axis=1)

        y_true = test["result"].to_numpy()
        for name, proba in [("poisson", _normalize(pois_p.to_numpy())),
                             ("logreg", _normalize(lr_p.to_numpy())),
                             ("xgb", _normalize(xg_p.to_numpy()))]:
            ll = log_loss(y_true, proba, labels=[0, 1, 2])
            bs = multiclass_brier(y_true, proba)
            results.append({"season": test_season, "model": name, "log_loss": ll, "brier": bs})

        market_proba = test[["p_h_fair", "p_d_fair", "p_a_fair"]].to_numpy()
        results.append({"season": test_season, "model": "market_fair",
                         "log_loss": log_loss(y_true, market_proba, labels=[0, 1, 2]),
                         "brier": multiclass_brier(y_true, market_proba)})

        # staking sim using XGBoost, best-edge side per match, vs Pinnacle
        if not stopped:
            test_xg = xg_p.reset_index(drop=True)
            for idx, row in test.sort_values("Date").iterrows():
                pos = test.index.get_loc(idx)
                model_probs = test_xg.iloc[pos]
                best_side, best_edge = None, edge_threshold
                for side in (0, 1, 2):
                    e = model_probs[side] - row[SIDE_COLS[side][1]]
                    if e > best_edge:
                        best_side, best_edge = side, e
                if best_side is None:
                    continue
                odds_col = SIDE_COLS[best_side][0]
                stake_frac = kelly_stake(model_probs[best_side], row[odds_col],
                                          kelly_fraction, max_stake_pct)
                if stake_frac <= 0:
                    continue
                stake = bankroll * stake_frac
                won = row["result"] == best_side
                bankroll += stake * (row[odds_col] - 1) if won else -stake
                peak = max(peak, bankroll)
                drawdown = (peak - bankroll) / peak
                max_drawdown = max(max_drawdown, drawdown)
                if drawdown >= stop_loss_pct:
                    stopped = True
                    break

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("model")[["log_loss", "brier"]].mean().sort_values("log_loss")

    print("=== 1X2 calibration vs Pinnacle (lower is better) ===")
    print(summary.to_string())
    print()
    print(f"=== Staking sim (XGBoost, best-edge side > {edge_threshold:.0%}, "
          f"{kelly_fraction:.0%} Kelly, max {max_stake_pct:.0%}/bet, stop-loss {stop_loss_pct:.0%}) ===")
    print(f"Final bankroll: {bankroll:.3f}x  |  Max drawdown: {max_drawdown:.1%}  |  Stopped early: {stopped}")

    return results_df, summary, bankroll, max_drawdown


if __name__ == "__main__":
    run()
