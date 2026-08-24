"""Build a leakage-safe feature table from raw matches.

For every match, all features are computed from information available
strictly *before* kickoff (rolling windows are shifted by one match).
"""
import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "matches.duckdb"

ROLL_WINDOWS = [5, 10]


def _long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home + away perspectives)."""
    home = matches.rename(columns={
        "HomeTeam": "team", "AwayTeam": "opponent",
        "FTHG": "goals_for", "FTAG": "goals_against",
        "HS": "shots_for", "AS": "shots_against",
        "HST": "sot_for", "AST": "sot_against",
        "HC": "corners_for", "AC": "corners_against",
    }).copy()
    home["is_home"] = 1
    home["cards_for"] = matches["HY"] + matches["HR"]
    home["cards_against"] = matches["AY"] + matches["AR"]

    away = matches.rename(columns={
        "AwayTeam": "team", "HomeTeam": "opponent",
        "FTAG": "goals_for", "FTHG": "goals_against",
        "AS": "shots_for", "HS": "shots_against",
        "AST": "sot_for", "HST": "sot_against",
        "AC": "corners_for", "HC": "corners_against",
    }).copy()
    away["is_home"] = 0
    away["cards_for"] = matches["AY"] + matches["AR"]
    away["cards_against"] = matches["HY"] + matches["HR"]

    keep = ["match_id", "Date", "league", "season", "team", "opponent", "is_home", "Referee",
            "goals_for", "goals_against", "shots_for", "shots_against",
            "sot_for", "sot_against", "corners_for", "corners_against",
            "cards_for", "cards_against"]
    long_df = pd.concat([home[keep], away[keep]], ignore_index=True)
    return long_df.sort_values(["team", "Date"])


def _add_rolling(long_df: pd.DataFrame) -> pd.DataFrame:
    stat_cols = ["goals_for", "goals_against", "shots_for", "sot_for",
                 "corners_for", "corners_against", "cards_for"]
    g = long_df.groupby("team", group_keys=False)
    for w in ROLL_WINDOWS:
        for col in stat_cols:
            long_df[f"{col}_r{w}"] = g[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

    # home/away-specific form: last 5 games in that venue only
    for venue, flag in [("home", 1), ("away", 0)]:
        mask = long_df["is_home"] == flag
        sub = long_df.loc[mask].sort_values(["team", "Date"])
        roll = sub.groupby("team")["goals_for"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        long_df.loc[mask, f"goals_for_{venue}form5"] = roll
        long_df[f"goals_for_{venue}form5"] = long_df.groupby("team")[f"goals_for_{venue}form5"].ffill()

    return long_df


def _add_referee_tendency(long_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    ref_cards = matches[["match_id", "Date", "Referee", "HY", "AY", "HR", "AR"]].copy()
    ref_cards["total_cards"] = ref_cards[["HY", "AY", "HR", "AR"]].sum(axis=1)
    ref_cards = ref_cards.sort_values(["Referee", "Date"])
    ref_cards["ref_card_avg"] = ref_cards.groupby("Referee")["total_cards"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return long_df.merge(ref_cards[["match_id", "ref_card_avg"]], on="match_id", how="left")


def _add_league_form(long_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["total_goals"] = m["FTHG"] + m["FTAG"]
    m = m.sort_values(["league", "season", "Date"])
    m["league_goal_avg"] = m.groupby(["league", "season"])["total_goals"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return long_df.merge(m[["match_id", "league_goal_avg"]], on="match_id", how="left")


def _add_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    m["pair"] = m.apply(lambda r: "|".join(sorted([r["HomeTeam"], r["AwayTeam"]])), axis=1)
    m["total_goals"] = m["FTHG"] + m["FTAG"]
    m = m.sort_values(["pair", "Date"])
    m["h2h_goal_avg"] = m.groupby("pair")["total_goals"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return m[["match_id", "h2h_goal_avg"]]


def build_features() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    matches = con.execute("SELECT * FROM matches").fetchdf()
    con.close()

    matches = matches.reset_index(drop=True)
    matches["match_id"] = matches.index
    matches = matches.dropna(subset=["FTHG", "FTAG", "Date"])

    long_df = _long_format(matches)
    long_df = _add_rolling(long_df)
    long_df = _add_referee_tendency(long_df, matches)
    long_df = _add_league_form(long_df, matches)

    feat_cols = [c for c in long_df.columns if c.endswith(("_r5", "_r10", "form5")) or c in
                 ("ref_card_avg", "league_goal_avg")]

    home_feats = long_df[long_df["is_home"] == 1][["match_id"] + feat_cols]
    home_feats.columns = ["match_id"] + [f"home_{c}" for c in feat_cols]
    away_feats = long_df[long_df["is_home"] == 0][["match_id"] + feat_cols]
    away_feats.columns = ["match_id"] + [f"away_{c}" for c in feat_cols]

    h2h = _add_h2h(matches)

    out = matches.merge(home_feats, on="match_id", how="left") \
                 .merge(away_feats, on="match_id", how="left") \
                 .merge(h2h, on="match_id", how="left")

    out["total_goals"] = out["FTHG"] + out["FTAG"]
    out["total_corners"] = out["HC"] + out["AC"]
    out["total_cards"] = out["HY"] + out["AY"] + out["HR"] + out["AR"]
    out["over_2_5"] = (out["total_goals"] > 2.5).astype(int)

    return out


def run():
    features = build_features()
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")
    con.close()
    n_missing = features["home_goals_for_r5"].isna().sum()
    print(f"Built {len(features)} rows, {features.shape[1]} columns -> table 'features'")
    print(f"Rows with no home rolling history yet (early-season): {n_missing}")


if __name__ == "__main__":
    run()
