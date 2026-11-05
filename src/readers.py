"""Readers for the four input formats.

Each reader returns a DataFrame of strings with the columns
source_record, country, continent, year, lifeExp, pop, gdpPercap.
Nothing is converted or validated here.
"""
from __future__ import annotations

import io

import json
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
    """Read the CSV file as strings, skipping comment lines."""
    prefix = settings.get("comment_prefix")
    with open(path, "r", encoding=settings.get("encoding", "utf-8"), newline="") as fh:
        lines = [ln for ln in fh if not (prefix and ln.startswith(prefix))]
    df = pd.read_csv(
        io.StringIO("".join(lines)),
        sep=settings.get("separator", ","),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    return to_canonical_raw(df, settings["mapping"], settings["file"])


def read_excel_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the observations sheet as strings."""
    try:
        df = pd.read_excel(
            path,
            sheet_name=settings["sheet_name"],
            dtype=str,
            keep_default_na=False,
            engine="openpyxl",
        )
    except ValueError as exc:
        raise FileLevelError(f"{settings['file']}: {exc}") from exc
    return to_canonical_raw(df, settings["mapping"], settings["file"])



def flatten_record(record: dict, prefix: str = "") -> dict:
    """Flatten nested dicts into one level using dotted keys (a.b.c)."""
    flat: dict = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_record(value, name))
        else:
            flat[name] = "" if value is None else str(value)
    return flat




def read_json_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the nested JSON records as strings, keeping original order."""
    name = settings["file"]
    try:
        with open(path, "r", encoding=settings.get("encoding", "utf-8")) as fh:
            # parse_float/parse_int/parse_constant keep the original number
            # text (e.g. "1.5", "NaN") so validation sees what was in the file.
            doc = json.load(fh, parse_float=str, parse_int=str, parse_constant=str)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FileLevelError(f"{name}: cannot parse JSON ({exc})") from exc
    records = doc.get(settings.get("record_path", "records")) if isinstance(doc, dict) else None
    if not isinstance(records, list):
        raise FileLevelError(f"{name}: record_path not found or not a list")
    rows = [flatten_record(r) if isinstance(r, dict) else {} for r in records]
    df = pd.DataFrame(rows, dtype=str).fillna("")
    return to_canonical_raw(df, settings["mapping"], name)


def read_parquet_source(path: Path, settings: dict) -> pd.DataFrame:
    """Read the Parquet file, keeping every value as text (no conversion)."""
    name = settings["file"]
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError, ImportError) as exc:
        raise FileLevelError(f"{name}: cannot read Parquet ({exc})") from exc
    df = df.astype(object).where(df.notna(), "").astype(str)
    return to_canonical_raw(df, settings["mapping"], name)

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