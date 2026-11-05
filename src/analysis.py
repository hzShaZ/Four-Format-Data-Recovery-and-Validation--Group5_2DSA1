"""pandas analysis and NumPy matrix (member 4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_COUNTRIES = ["Philippines", "Indonesia", "Thailand"]
DEFAULT_YEARS = [1982, 1987, 1992, 1997, 2002, 2007]
TOLERANCE = 1e-8


def asia_selected_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Asia rows for one year with population in millions and life-expectancy rank."""
    selected = df.loc[(df["continent"] == "Asia") & (df["year"] == year), :].copy()
    selected["pop_millions"] = selected["pop"] / 1_000_000
    selected["lifeExp_rank"] = (
        selected["lifeExp"].rank(method="min", ascending=False).astype("int64")
    )
    selected = selected.sort_values(["lifeExp_rank", "country"], kind="mergesort")
    return selected.reset_index(drop=True)


def country_history(df: pd.DataFrame, country: str = "Philippines") -> pd.DataFrame:
    """All rows of one country by year, with population in millions and life-expectancy change."""
    rows = df.loc[df["country"] == country, :].sort_values("year").copy()
    if rows.empty:
        raise ValueError(f"no rows found for country {country!r}")
    rows["pop_millions"] = rows["pop"] / 1_000_000
    rows["lifeExp_change"] = rows["lifeExp"] - rows["lifeExp"].iloc[0]
    return rows.reset_index(drop=True)