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


def devig_three_way(odds_h: pd.Series, odds_d: pd.Series, odds_a: pd.Series) -> pd.DataFrame:
    """Remove overround from a three-outcome market (match result 1X2),
    using the basic proportional method (Shin's method is a known, more
    accurate alternative, skipped here for simplicity)."""
    p_h, p_d, p_a = implied_prob(odds_h), implied_prob(odds_d), implied_prob(odds_a)
    overround = p_h + p_d + p_a
    return pd.DataFrame({
        "p_h_fair": p_h / overround,
        "p_d_fair": p_d / overround,
        "p_a_fair": p_a / overround,
        "overround": overround,
    })


def edge(model_prob: pd.Series, fair_prob: pd.Series) -> pd.Series:
    """Model's edge over the market's own fair (de-vigged) probability."""
    return model_prob - fair_prob
