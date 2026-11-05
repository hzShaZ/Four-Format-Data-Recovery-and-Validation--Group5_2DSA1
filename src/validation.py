"""Validation, quarantine, duplicates and conflicts (member 3)."""
from __future__ import annotations

import pandas as pd


class RunResult:
    """Holds quarantine, duplicate log and audit records for one input."""

    def __init__(self) -> None:
        self.quarantine: list = []
        self.duplicates: list = []
        self.audit: list = []


def validate_records(
    raw: pd.DataFrame, rules: dict, source_file: str
) -> tuple[pd.DataFrame, RunResult]:
    """Validate raw strings, quarantine bad rows, drop exact duplicates."""
    raise NotImplementedError