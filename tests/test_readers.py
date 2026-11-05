import json
import unittest
from pathlib import Path

from src.readers import CANONICAL, FileLevelError, read_source

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = str(ROOT / "data")


def setting(name: str) -> dict:
    """Get the config entry for one file."""
    return next(s for s in CONFIG["inputs"] if s["file"] == name)


class TestCsvReader(unittest.TestCase):
    def test_csv_rows_and_columns(self) -> None:
        df = read_source(setting("country_records.csv"), DATA)
        self.assertEqual(len(df), 1712)
        self.assertEqual(list(df.columns), ["source_record"] + CANONICAL)

    def test_csv_source_record_starts_at_one(self) -> None:
        df = read_source(setting("country_records.csv"), DATA)
        self.assertEqual(df["source_record"].iloc[0], 1)
        self.assertEqual(df["source_record"].iloc[-1], 1712)

    def test_csv_values_are_strings(self) -> None:
        df = read_source(setting("country_records.csv"), DATA)
        self.assertTrue(all(isinstance(v, str) for v in df["year"]))


class TestExcelAndFailures(unittest.TestCase):
    def test_excel_rows_and_columns(self) -> None:
        df = read_source(setting("country_records.xlsx"), DATA)
        self.assertEqual(len(df), 1712)
        self.assertEqual(list(df.columns), ["source_record"] + CANONICAL)

    def test_missing_file_stops(self) -> None:
        bad = dict(setting("country_records.csv"), file="nope.csv")
        with self.assertRaises(FileLevelError):
            read_source(bad, DATA)

    def test_unsupported_format_stops(self) -> None:
        bad = {"file": "unsupported.txt", "format": "txt", "mapping": {}}
        with self.assertRaises(FileLevelError):
            read_source(bad, DATA)

    def test_missing_column_stops(self) -> None:
        canon = {c: c for c in CANONICAL}
        bad = {"file": "missing_column.csv", "format": "csv", "mapping": canon}
        with self.assertRaises(FileLevelError):
            read_source(bad, DATA)


if __name__ == "__main__":
    unittest.main()