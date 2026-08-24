"""Convert bookmaker odds to de-vigged (fair) probabilities."""
import pandas as pd


def implied_prob(odds: pd.Series) -> pd.Series:
    return 1.0 / odds


def devig_two_way(odds_over: pd.Series, odds_under: pd.Series) -> pd.DataFrame:
    """Remove overround from a two-outcome market (e.g. over/under 2.5)."""
    p_over_raw = implied_prob(odds_over)
    p_under_raw = implied_prob(odds_under)
    overround = p_over_raw + p_under_raw
    return pd.DataFrame({
        "p_over_fair": p_over_raw / overround,
        "p_under_fair": p_under_raw / overround,
        "overround": overround,
    })


def edge(model_prob: pd.Series, fair_prob: pd.Series) -> pd.Series:
    """Model's edge over the market's own fair (de-vigged) probability."""
    return model_prob - fair_prob
