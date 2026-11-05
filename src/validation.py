from __future__ import annotations
"""Validation, quarantine, duplicates and conflicts (member 3)."""
"Input is the raw string table produced by a reader (columns ``source_record``"
"plus the six canonical columns). Values are checked *before* any conversion;"
"rows that break a rule go to quarantine, exact copies of valid rows go to the"
"duplicate log, and a (country, year) key whose rows disagree stops the run."
 
"`rules`` is a plain dictionary (taken from config.json) with the keys:"
 
"* ``allowed_continents`` - list of continent names"
"* ``allowed_years`` - list of integer years"
"* ``life_exp_max`` - upper bound for lifeExp (optional, default 120)"

import math
import re
from decimal import Decimal
from typing import Any
import pandas as pd

from src.readers import CANONICAL, FileLevelError


KEY = ("country", "year")
INT64_MAX = 2**63 - 1
QUARANTINE_COLUMNS = ["source_file", "source_record", "reason"] + CANONICAL
DUPLICATE_COLUMNS = ["source_file", "source_record", "kept_source_record"] + CANONICAL
CHECK_COLUMNS = ["check", "expected", "actual", "tolerance", "pass"]
AUDIT_COLUMNS = ["step", "source_file", "operation", "rule", "rows_before", "rows_after"]
 
# Plain decimal or scientific notation only: no underscores, commas or words.
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_NON_FINITE = {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
 
 
class ConflictingKeyError(FileLevelError):
    """Two valid rows share (country, year) but disagree on another field."""
 
    def __init__(self, message: str, conflicts: list, result: "RunResult") -> None:
        super().__init__(message)
        self.conflicts = conflicts
        self.result = result

class RunResult:
    """Holds quarantine, duplicate log and audit records for one input."""

    def __init__(self, source_file: str = "") -> None:
        self.source_file = source_file
        self.quarantine: list = []
        self.duplicates: list = []
        self.audit: list = []
        self.counts: dict = {}

    def add_audit(self, operation: str, rule: str, before: int, after: int) -> None:
         """Append one audit record; steps are numbered within this input."""
         self.audit.append(
            {
                "step": len(self.audit) + 1,
                "source_file": self.source_file,
                "operation": operation,
                "rule": rule,
                "rows_before": before,
                "rows_after": after,
            }
        )
 
    def quarantine_frame(self) -> pd.DataFrame:
        """Quarantined records as a table in the quarantine.csv layout."""
        return pd.DataFrame(self.quarantine, columns=QUARANTINE_COLUMNS)
 
    def duplicate_frame(self) -> pd.DataFrame:
        """Removed duplicate copies as a table in the duplicate_log.csv layout."""
        return pd.DataFrame(self.duplicates, columns=DUPLICATE_COLUMNS)
 
    def audit_frame(self) -> pd.DataFrame:
        """Audit records as a table in the audit_log.csv layout."""
        return pd.DataFrame(self.audit, columns=AUDIT_COLUMNS)
 
    def reconciliation_checks(self, expected: dict | None = None) -> list:
        """Rows for validation.csv: raw = rejected + duplicates + canonical.
 
        ``expected`` may hold the manifest's expected_per_file counts; each
        count that is present is compared exactly.
        """
        c = self.counts
        name = self.source_file
        total = c["rejected_rows"] + c["exact_duplicates_removed"] + c["canonical_rows"]
        checks = [_check(f"{name}: raw = rejected + duplicates + canonical", c["raw_rows"], total)]
        for field, label in (
            ("raw_rows", "raw rows"),
            ("rejected_rows", "rejected rows"),
            ("valid_before_deduplication", "valid before deduplication"),
            ("exact_duplicates_removed", "exact duplicates removed"),
            ("canonical_rows", "canonical rows"),
        ):
            if expected and field in expected:
                checks.append(_check(f"{name}: {label}", expected[field], c[field]))
        return checks
 
 
def _check(name: str, expected: Any, actual: Any, tolerance: float = 0) -> dict:
    """Build one validation.csv row; counts and text are compared exactly."""
    return {
        "check": name,
        "expected": expected,
        "actual": actual,
        "tolerance": tolerance,
        "pass": bool(expected == actual),
    }
 
 
def _text(value: Any) -> str | None:
    """Trim a field to text; None and NaN become None (missing)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value is pd.NA:
        return None
    return str(value).strip()
 
 
def _original(value: Any) -> str:
    """The field exactly as read, for quarantine output (missing -> empty)."""
    return "" if _text(value) is None else str(value)
 
 
def _number(text: str | None) -> Decimal:
    """Parse text into an exact Decimal, or raise ValueError with the reason."""
    if text is None or text == "":
        raise ValueError("missing value")
    if text.lower() in _NON_FINITE:
        raise ValueError(f"non-finite value {text!r}")
    if not _NUMBER.fullmatch(text):
        raise ValueError(f"not a number {text!r}")
    return Decimal(text)
 
 
def _integer(text: str | None, upper: int) -> int:
    """Parse a whole number (1952 or 1952.0 are fine, 1952.5 is not)."""
    d = _number(text)
    if d.adjusted() > 18 or abs(d) > upper:
        raise ValueError(f"out of range {text!r}")
    if d != d.to_integral_value():
        raise ValueError(f"fractional value {text!r}")
    return int(d)
 
 
def _real(text: str | None) -> float:
    """Parse a finite float, rejecting overflow to infinity."""
    d = _number(text)
    value = float(d)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {text!r}")
    return value
 
 
def check_row(fields: dict, rules: dict) -> tuple[dict | None, list]:
    """Validate one raw record; return (typed values or None, reasons)."""
    reasons: list = []
    out: dict = {}
    allowed_years = set(rules["allowed_years"])
    allowed_continents = set(rules["allowed_continents"])
    life_max = rules.get("life_exp_max", 120)
 
    country = _text(fields["country"])
    if not country:
        reasons.append("country: empty after trimming")
    out["country"] = country
 
    continent = _text(fields["continent"])
    if not continent:
        reasons.append("continent: empty after trimming")
    elif continent not in allowed_continents:
        reasons.append(f"continent: {continent!r} not in allowed set")
    out["continent"] = continent
 
    def numeric(name: str, parse, test, message: str) -> None:
        try:
            value = parse(_text(fields[name]))
        except ValueError as exc:
            reasons.append(f"{name}: {exc}")
            return
        if not test(value):
            reasons.append(f"{name}: {message} ({fields[name]!r})")
            return
        out[name] = value
 
    numeric("year", lambda t: _integer(t, INT64_MAX), lambda v: v in allowed_years,
            "not an allowed year")
    numeric("lifeExp", _real, lambda v: 0 < v <= life_max, f"must be > 0 and <= {life_max}")
    numeric("pop", lambda t: _integer(t, INT64_MAX), lambda v: v > 0, "must be > 0")
    numeric("gdpPercap", _real, lambda v: v > 0, "must be > 0")
 
    if reasons:
        return None, reasons
    return out, reasons
 
 
def validate_records(
    raw: pd.DataFrame, rules: dict, source_file: str
) -> tuple[pd.DataFrame, RunResult]:
    """Validate raw strings, quarantine bad rows, drop exact duplicates.
 
    Returns the typed, sorted canonical table and a RunResult. Raises
    ConflictingKeyError (carrying the RunResult) if a key has contradictory rows.
    """
    needed = set(CANONICAL) | {"source_record"}
    if needed - set(raw.columns):
        raise FileLevelError(f"{source_file}: raw table lacks {sorted(needed - set(raw.columns))}")
 
    result = RunResult(source_file)
    valid: list = []  # (source_record, typed values)
    for rec in raw.to_dict("records"):
        values, reasons = check_row(rec, rules)
        number = int(rec["source_record"])
        if values is None:
            result.quarantine.append(
                {"source_file": source_file, "source_record": number,
                 "reason": "; ".join(reasons), **{c: _original(rec[c]) for c in CANONICAL}}
            )
        else:
            valid.append((number, values))
    result.add_audit("validate", "per-field type, domain and range rules", len(raw), len(valid))
 
    kept: dict = {}  # key -> (source_record, values)
    conflicts: list = []
    for number, values in valid:
        key = tuple(values[k] for k in KEY)
        if key not in kept:
            kept[key] = (number, values)
            continue
        first_number, first = kept[key]
        differs = sorted(c for c in CANONICAL if first[c] != values[c])
        if differs:
            conflicts.append({"key": key, "records": (first_number, number), "fields": differs})
        else:
            result.duplicates.append(
                {"source_file": source_file, "source_record": number,
                 "kept_source_record": first_number, **values}
            )
 
    if conflicts:
        result.add_audit("check_conflicts", "FAILED: same (country, year), different values",
                         len(valid), 0)
        first = conflicts[0]
        raise ConflictingKeyError(
            f"{source_file}: {len(conflicts)} conflicting key(s), e.g. {first['key']} "
            f"records {first['records']} differ in {first['fields']}",
            conflicts, result,
        )
    result.add_audit("check_conflicts", "same (country, year) must not disagree",
                     len(valid), len(valid))
    result.add_audit("deduplicate", "remove exact copies of valid rows",
                     len(valid), len(kept))
 
    clean = pd.DataFrame([v for _, v in kept.values()], columns=CANONICAL)
    clean = clean.astype(
        {"country": "object", "continent": "object", "year": "int64",
         "lifeExp": "float64", "pop": "int64", "gdpPercap": "float64"}
    )
    clean = clean.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    result.add_audit("sort", "order by country, then year", len(clean), len(clean))
 
    result.counts = {
        "raw_rows": len(raw),
        "rejected_rows": len(result.quarantine),
        "valid_before_deduplication": len(valid),
        "exact_duplicates_removed": len(result.duplicates),
        "canonical_rows": len(clean),
    }
    return clean, result
 
 
def domain_checks(clean: pd.DataFrame, rules: dict, source_file: str) -> list:
    """Rows for validation.csv: key uniqueness and domain checks on clean data."""
    allowed_years = set(rules["allowed_years"])
    allowed_continents = set(rules["allowed_continents"])
    life_max = rules.get("life_exp_max", 120)
    keys = list(zip(clean["country"], clean["year"]))
    return [
        _check(f"{source_file}: duplicate (country, year) keys", 0, len(keys) - len(set(keys))),
        _check(f"{source_file}: missing values", 0, int(clean[CANONICAL].isna().sum().sum())),
        _check(f"{source_file}: years outside allowed set", 0,
               len(set(clean["year"]) - allowed_years)),
        _check(f"{source_file}: continents outside allowed set", 0,
               len(set(clean["continent"]) - allowed_continents)),
        _check(f"{source_file}: empty country names", 0, int((clean["country"] == "").sum())),
        _check(f"{source_file}: lifeExp outside (0, {life_max}]", 0,
               int((~clean["lifeExp"].between(0, life_max, inclusive="right")).sum())),
        _check(f"{source_file}: pop <= 0", 0, int((clean["pop"] <= 0).sum())),
        _check(f"{source_file}: gdpPercap <= 0", 0, int((clean["gdpPercap"] <= 0).sum())),
        _check(f"{source_file}: column order", CANONICAL, list(clean.columns)),
    ]
 