"""Exports, reload comparison, staging (member 5)."""
from __future__ import annotations

import pandas as pd


def export_canonical(df: pd.DataFrame, staging_dir: str) -> list[str]:
    """Write canonical.csv/xlsx/json/parquet into the staging folder."""
    raise NotImplementedError