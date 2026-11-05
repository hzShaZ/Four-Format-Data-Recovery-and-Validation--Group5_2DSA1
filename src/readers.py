"""Readers for the four input formats.

Each reader returns a DataFrame of strings with the columns
source_record, country, continent, year, lifeExp, pop, gdpPercap.
Nothing is converted or validated here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL = ["country", "continent", "year", "lifeExp", "pop", "gdpPercap"]


class FileLevelError(Exception):
    """Problem with a file or its columns. Stops the run."""


def to_canonical_raw(df: pd.DataFrame, mapping: dict, source_file: str) -> pd.DataFrame:
    """Rename columns to canonical names and add source_record."""
    if set(mapping.values()) != set(CANONICAL):
        raise FileLevelError(f"{source_file}: bad column mapping")
    missing = set(mapping) - set(df.columns)
    if missing:
        raise FileLevelError(f"{source_file}: missing columns {sorted(missing)}")
    out = df.rename(columns=mapping)[CANONICAL].copy()
    out.insert(0, "source_record", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def read_csv_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the CSV file."""
    raise NotImplementedError


def read_excel_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the Excel file."""
    raise NotImplementedError


def read_json_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the JSON file."""
    raise NotImplementedError


def read_parquet_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the Parquet file."""
    raise NotImplementedError


READERS = {
    "csv": read_csv_source,
    "xlsx": read_excel_source,
    "json": read_json_source,
    "parquet": read_parquet_source,
}


def read_source(settings: dict, data_dir: str) -> pd.DataFrame:
    """Pick the right reader for a config entry and run it."""
    path = Path(data_dir) / settings["file"]
    fmt = settings.get("format")
    if fmt not in READERS:
        raise FileLevelError(f"{settings['file']}: unsupported format {fmt}")
    if not path.is_file():
        raise FileLevelError(f"{settings['file']}: file not found")
    return READERS[fmt](path, settings)