"""Independent-Poisson goal model (standard simplification of Dixon-Coles:
two independent Poisson variables for home/away goals, team attack/defense
strength + home advantage as fixed effects; skips the low-score rho
correlation correction Dixon-Coles adds on top).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

MAX_GOALS = 10


def _team_goal_frame(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "team": matches["HomeTeam"], "opponent": matches["AwayTeam"],
        "goals": matches["FTHG"], "home": 1,
    })
    away = pd.DataFrame({
        "team": matches["AwayTeam"], "opponent": matches["HomeTeam"],
        "goals": matches["FTAG"], "home": 0,
    })
    return pd.concat([home, away], ignore_index=True)


def fit(train_matches: pd.DataFrame):
    long_df = _team_goal_frame(train_matches)
    model = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=long_df, family=sm.families.Poisson(),
    ).fit()
    known_teams = set(train_matches["HomeTeam"]) | set(train_matches["AwayTeam"])
    return {"model": model, "known_teams": known_teams}


def _expected_goals(fitted, home_team: str, away_team: str) -> tuple[float, float]:
    model, known = fitted["model"], fitted["known_teams"]
    # unseen (promoted) teams: fall back to a neutral team already in training
    # data so statsmodels' category encoding doesn't choke on an unknown level
    fallback = next(iter(known))
    h = home_team if home_team in known else fallback
    a = away_team if away_team in known else fallback

    mu_home = model.predict(pd.DataFrame({"team": [h], "opponent": [a], "home": [1]})).iloc[0]
    mu_away = model.predict(pd.DataFrame({"team": [a], "opponent": [h], "home": [0]})).iloc[0]
    return mu_home, mu_away


def predict_result_probs(fitted, matches: pd.DataFrame) -> pd.DataFrame:
    """P(home win), P(draw), P(away win) from the joint scoreline grid."""
    goal_grid = np.arange(0, MAX_GOALS + 1)
    rows = []
    for home_team, away_team in zip(matches["HomeTeam"], matches["AwayTeam"]):
        mu_h, mu_a = _expected_goals(fitted, home_team, away_team)
        ph = poisson.pmf(goal_grid, mu_h)
        pa = poisson.pmf(goal_grid, mu_a)
        joint = np.outer(ph, pa)  # joint[i, j] = P(home scores i, away scores j)
        p_home = np.tril(joint, -1).sum()
        p_draw = np.trace(joint)
        p_away = np.triu(joint, 1).sum()
        rows.append((p_home, p_draw, p_away))
    return pd.DataFrame(rows, index=matches.index, columns=["p_home", "p_draw", "p_away"])


def predict_over_prob(fitted, matches: pd.DataFrame, line: float = 2.5) -> pd.Series:
    goal_grid = np.arange(0, MAX_GOALS + 1)
    probs = []
    for home_team, away_team in zip(matches["HomeTeam"], matches["AwayTeam"]):
        mu_h, mu_a = _expected_goals(fitted, home_team, away_team)
        ph = poisson.pmf(goal_grid, mu_h)
        pa = poisson.pmf(goal_grid, mu_a)
        # distribution of total goals = convolution of the two independent Poissons
        total_dist = np.convolve(ph, pa)
        totals = np.arange(len(total_dist))
        p_over = total_dist[totals > line].sum()
        probs.append(p_over)
    return pd.Series(probs, index=matches.index)
