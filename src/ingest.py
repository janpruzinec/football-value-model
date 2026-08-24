"""Download historical match CSVs from football-data.co.uk and load into DuckDB."""
import io
import pathlib

import duckdb
import pandas as pd
import requests
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config" / "leagues.yaml").read_text())
DB_PATH = ROOT / "data" / "processed" / "matches.duckdb"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# columns we actually care about; football-data.co.uk adds/drops columns
# across seasons so we select what's present and leave the rest.
KEEP_COLS = [
    "Div", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC", "HY", "AY", "HR", "AR",
    "Referee", "Attendance",
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
    "B365>2.5", "B365<2.5",
]


def season_codes(start: str, end: str) -> list[str]:
    start_yy, end_yy = int(start[:2]), int(end[:2])
    return [f"{yy:02d}{yy + 1:02d}" for yy in range(start_yy, end_yy + 1)]


def fetch_one(league: str, season: str) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season, league=league)
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200 or not resp.content:
        return None
    try:
        df = pd.read_csv(io.BytesIO(resp.content), encoding="latin1", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return None
    if "HomeTeam" not in df.columns:
        return None
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    df["season"] = season
    df["league"] = league
    return df


def run():
    seasons = season_codes(CONFIG["seasons"]["start"], CONFIG["seasons"]["end"])
    leagues = list(CONFIG["leagues"].keys())

    frames = []
    for league in leagues:
        for season in seasons:
            df = fetch_one(league, season)
            if df is not None:
                frames.append(df)
                print(f"  {league} {season}: {len(df)} rows")
            else:
                print(f"  {league} {season}: not found, skipping")

    if not frames:
        raise RuntimeError("No data fetched — check network access / league codes.")

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["Date"] = pd.to_datetime(all_matches["Date"], dayfirst=True, errors="coerce")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM all_matches")
    con.close()

    print(f"\nTotal: {len(all_matches)} matches -> {DB_PATH}")
    print(f"Leagues: {leagues}")
    print(f"Date range: {all_matches['Date'].min().date()} to {all_matches['Date'].max().date()}")


if __name__ == "__main__":
    run()
