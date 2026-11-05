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

def population_matrix(df: pd.DataFrame, countries: list[str], years: list[int]) -> np.ndarray:
    """Build a (countries x years) NumPy matrix of populations."""
    lookup = {(c, int(y)): p for c, y, p in zip(df["country"], df["year"], df["pop"])}
    try:
        values = [[lookup[(c, int(y))] for y in years] for c in countries]
    except KeyError as exc:
        raise ValueError(f"no population for {exc.args[0]}") from exc
    return np.array(values, dtype=np.float64)


def index_to_baseline(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast each row's first column (baseline) so that baseline = 100."""
    baseline = matrix[:, [0]]  # shape (rows, 1) stretches across all columns
    return matrix / baseline * 100.0, baseline


def index_to_baseline_loop(matrix: np.ndarray) -> np.ndarray:
    """Independent loop version of index_to_baseline, used to verify it."""
    out = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        base = float(matrix[i, 0])
        for j in range(matrix.shape[1]):
            out[i, j] = float(matrix[i, j]) / base * 100.0
    return out


def population_index_frame(
    index: np.ndarray, countries: list[str], years: list[int]
) -> pd.DataFrame:
    """Index matrix as a table: a country column plus one named column per year."""
    table = pd.DataFrame(index, columns=[str(y) for y in years])
    table.insert(0, "country", countries)
    return table


def asia_population_total_numpy(df: pd.DataFrame, year: int) -> tuple[int, int]:
    """Boolean selection and sum with NumPy only: (Asian row count, population total)."""
    continent = df["continent"].to_numpy()
    years = df["year"].to_numpy()
    pop = df["pop"].to_numpy()
    mask = (continent == "Asia") & (years == year)
    return int(mask.sum()), int(pop[mask].sum())



def _row(name: str, expected: object, actual: object, tolerance: float = 0.0) -> dict:
    """Build one validation.csv row; with a tolerance, numbers may differ by up to it."""
    if tolerance:
        ok = abs(float(actual) - float(expected)) <= tolerance
    else:
        ok = bool(expected == actual)
    return {"check": name, "expected": expected, "actual": actual,
            "tolerance": tolerance, "pass": ok}


def run_analysis(
    clean: pd.DataFrame,
    year: int,
    countries: list[str] | None = None,
    years: list[int] | None = None,
    controls: dict | None = None,
) -> dict:
    """Run all analysis steps; return tables, validation rows and array shapes.

    ``controls`` is the manifest's controls dictionary. Its 2007 control values
    are only compared when the reporting year is 2007.
    """
    countries = countries or DEFAULT_COUNTRIES
    years = years or DEFAULT_YEARS

    asia = asia_selected_year(clean, year)
    history = country_history(clean, "Philippines")
    matrix = population_matrix(clean, countries, years)
    index, baseline = index_to_baseline(matrix)
    loop_index = index_to_baseline_loop(matrix)
    np_rows, np_total = asia_population_total_numpy(clean, year)
    table = population_index_frame(index, countries, years)

    shapes = {"matrix": matrix.shape, "baseline": baseline.shape, "index": index.shape}
    first_pass = bool(len(asia) == 0 or asia["lifeExp_rank"].min() == 1)
    checks = [
        _row(f"asia {year}: row count, pandas vs NumPy mask", len(asia), np_rows),
        _row(f"asia {year}: population total, pandas vs NumPy", int(asia["pop"].sum()), np_total),
        _row(f"asia {year}: best life-expectancy rank is 1", True, first_pass),
        _row("philippines history: years in ascending order", True,
             bool(history["year"].is_monotonic_increasing)),
        _row("population matrix shape before broadcasting",
             (len(countries), len(years)), matrix.shape),
        _row("baseline column shape", (len(countries), 1), baseline.shape),
        _row("index shape after broadcasting", (len(countries), len(years)), index.shape),
        _row("index: broadcast vs loop, max absolute difference", 0.0,
             float(np.max(np.abs(index - loop_index))), TOLERANCE),
        _row("index: 1982 baseline column equals 100", 0.0,
             float(np.max(np.abs(index[:, 0] - 100.0))), TOLERANCE),
    ]
    if controls and year == 2007:
        ph = clean.loc[(clean["country"] == "Philippines") & (clean["year"] == 2007), "pop"]
        checks += [
            _row("control: asia 2007 rows", controls["asia_2007_rows"], len(asia)),
            _row("control: asia 2007 population", controls["asia_2007_population"], np_total),
            _row("control: philippines rows", controls["philippines_rows"], len(history)),
            _row("control: philippines 2007 population",
                 controls["philippines_2007_population"], int(ph.iloc[0])),
        ]
    return {
        "tables": {"asia_selected_year": asia, "country_history": history,
                   "population_index": table},
        "checks": checks,
        "shapes": shapes,
    }


def write_analysis_files(tables: dict, folder: str) -> list[str]:
    """Write each analysis table as <name>.csv into a folder (use the staging folder)."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in tables.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))
    return paths