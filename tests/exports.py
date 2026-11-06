"""Exports, reload comparison, staging and report writing (member 5)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.readers import CANONICAL
from src.validation import CHECK_COLUMNS

KEYS = ["country", "year"]
TEXT_COLS = ["country", "continent"]
INT_COLS = ["year", "pop"]
FLOAT_COLS = ["lifeExp", "gdpPercap"]
FLOAT_TOLERANCE = 1e-8
EXPORT_NAMES = {
    "csv": "canonical.csv",
    "xlsx": "canonical.xlsx",
    "json": "canonical.json",
    "parquet": "canonical.parquet",
}


def make_check(name: str, expected, actual, tolerance=0, passed: bool = True) -> dict:
    """Build one validation.csv row."""
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "pass": bool(passed),
    }


def make_audit(step: int, source_file: str, operation: str, rule: str,
               rows_before: int, rows_after: int) -> dict:
    """Build one audit_log.csv row."""
    return {
        "step": step,
        "source_file": source_file,
        "operation": operation,
        "rule": rule,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }


def write_rows(rows: list, path, columns: list) -> None:
    """Write a list of dicts to CSV. The header is kept even when the list is empty."""
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def apply_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Keep the schema columns and cast each one to its declared type.

    CSV, Excel and JSON do not store column types, so after a reload pandas
    has to guess them (text can come back as object, whole numbers can come
    back as float). Casting both sides to the declared schema puts the
    comparison on the same types. Parquet keeps its types, so for it the
    cast changes nothing.
    """
    out = df[list(schema)].copy()
    for col, dtype in schema.items():
        out[col] = out[col].astype(dtype)
    return out


def export_canonical(df: pd.DataFrame, staging_dir) -> list[str]:
    """Write canonical.csv, .xlsx, .json and .parquet into the staging folder."""
    out = Path(staging_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = df[CANONICAL]
    paths = {fmt: out / name for fmt, name in EXPORT_NAMES.items()}
    df.to_csv(paths["csv"], index=False)
    df.to_excel(paths["xlsx"], sheet_name="observations", index=False)
    # 15 digits is the most the JSON writer allows; the default is only 10,
    # which would break the 1e-8 float comparison.
    df.to_json(paths["json"], orient="records", indent=2,
               double_precision=15, force_ascii=False)
    df.to_parquet(paths["parquet"], index=False)
    return [str(p) for p in paths.values()]


def reload_export(path, schema: dict) -> pd.DataFrame:
    """Read one exported file back and reapply the declared schema."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, keep_default_na=False)
    elif path.suffix == ".xlsx":
        df = pd.read_excel(path, sheet_name="observations", engine="openpyxl",
                           keep_default_na=False)
    elif path.suffix == ".json":
        df = pd.read_json(path, orient="records", dtype=False, precise_float=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"cannot reload {path.name}")
    return apply_schema(df, schema)


def compare_tables(left: pd.DataFrame, right: pd.DataFrame, label: str,
                   schema: dict, atol: float = FLOAT_TOLERANCE) -> list[dict]:
    """Compare two tables by key. Text and integers exact, floats within atol."""
    a = apply_schema(left, schema).sort_values(KEYS).reset_index(drop=True)
    b = apply_schema(right, schema).sort_values(KEYS).reset_index(drop=True)
    rows = [make_check(f"{label}: row count", len(a), len(b), 0, len(a) == len(b))]
    if len(a) != len(b):
        return rows

    same = a[KEYS].equals(b[KEYS])
    rows.append(make_check(f"{label}: keys", "identical", "identical" if same else "differ", 0, same))
    same = a[TEXT_COLS].equals(b[TEXT_COLS])
    rows.append(make_check(f"{label}: text", "identical", "identical" if same else "differ", 0, same))
    same = bool((a[INT_COLS].to_numpy() == b[INT_COLS].to_numpy()).all())
    rows.append(make_check(f"{label}: integers", "identical", "identical" if same else "differ", 0, same))
    miss_a, miss_b = int(a.isna().sum().sum()), int(b.isna().sum().sum())
    rows.append(make_check(f"{label}: missing values", miss_a, miss_b, 0, miss_a == miss_b))
    for col in FLOAT_COLS:
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        close = bool(np.allclose(x, y, rtol=0, atol=atol, equal_nan=True))
        worst = float(np.max(np.abs(x - y))) if len(x) else 0.0
        rows.append(make_check(f"{label}: {col} max abs difference", 0.0, worst, atol, close))
    return rows


def verify_exports(canonical: pd.DataFrame, paths: list, schema: dict) -> list[dict]:
    """Reload every export and compare it with the canonical table."""
    checks = []
    expected_types = {col: str(pd.api.types.pandas_dtype(t)) for col, t in schema.items()}
    for path in paths:
        kind = Path(path).suffix.lstrip(".")
        label = f"export round trip {kind}"
        try:
            back = reload_export(path, schema)
        except Exception as exc:  # unreadable export counts as a failed check
            checks.append(make_check(f"{label}: reload", "readable", f"{type(exc).__name__}: {exc}", 0, False))
            continue
        actual_types = {col: str(back[col].dtype) for col in schema}
        checks.append(make_check(f"{label}: types after schema", expected_types, actual_types,
                                 0, actual_types == expected_types))
        checks.extend(compare_tables(canonical, back, label, schema))
    return checks


def promote_staging(staging_dir, output_dir) -> None:
    """Copy every staged file into the output folder, then remove staging."""
    staging, out = Path(staging_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for item in staging.iterdir():
        shutil.copy2(item, out / item.name)
    shutil.rmtree(staging)


def write_failed_report(checks: list, message: str, output_dir) -> str:
    """Save the checks so far plus the error message; return the report path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(checks) + [make_check("run failed", "no errors", message, "", False)]
    path = out / "failed_validation_report.csv"
    write_rows(rows, path, CHECK_COLUMNS)
    return str(path)
