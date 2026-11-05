import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.readers import CANONICAL, FileLevelError, flatten_record, read_source

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = str(ROOT / "data")
EXPECTED_COLUMNS = ["source_record"] + CANONICAL


def setting(name: str) -> dict:
    """Get the config entry for one file."""
    return next(s for s in CONFIG["inputs"] if s["file"] == name)


def write_json(folder: str, name: str, doc: object) -> None:
    """Write a JSON fixture into a temporary folder (never into data/)."""
    Path(folder, name).write_text(json.dumps(doc), encoding="utf-8")


def good_record(country: str = "Togo", year: int = 1987) -> dict:
    """Build one nested record in the layout of country_records.json."""
    return {
        "identity": {"country": country, "continent": "Africa", "year": year},
        "measures": {"lifeExp": 56.941, "pop": 3154264, "gdpPercap": 1202.201361},
    }


class TestFlattenRecord(unittest.TestCase):
    def test_nested_keys_become_dotted(self) -> None:
        flat = flatten_record(good_record())
        self.assertEqual(flat["identity.country"], "Togo")
        self.assertEqual(flat["measures.pop"], "3154264")

    def test_none_becomes_empty_string(self) -> None:
        flat = flatten_record({"identity": {"country": None}})
        self.assertEqual(flat["identity.country"], "")


class TestJsonReader(unittest.TestCase):
    def setUp(self) -> None:
        self.df = read_source(setting("country_records.json"), DATA)

    def test_json_rows_and_columns(self) -> None:
        self.assertEqual(len(self.df), 1712)
        self.assertEqual(list(self.df.columns), EXPECTED_COLUMNS)

    def test_json_source_record_is_one_based_in_file_order(self) -> None:
        self.assertEqual(self.df["source_record"].iloc[0], 1)
        self.assertEqual(self.df["source_record"].iloc[-1], 1712)
        self.assertEqual(self.df["country"].iloc[0], "Togo")
        self.assertEqual(self.df["year"].iloc[0], "1987")

    def test_json_values_are_strings(self) -> None:
        for col in CANONICAL:
            self.assertTrue(all(isinstance(v, str) for v in self.df[col]), col)

    def test_json_keeps_original_number_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec = good_record()
            rec["identity"]["year"] = 1987.5
            write_json(tmp, "x.json", {"records": [rec]})
            cfg = dict(setting("country_records.json"), file="x.json")
            df = read_source(cfg, tmp)
        self.assertEqual(df["year"].iloc[0], "1987.5")


class TestJsonFailures(unittest.TestCase):
    def read_fixture(self, doc: object) -> pd.DataFrame:
        """Write doc to a temp JSON file and read it with the real config."""
        with tempfile.TemporaryDirectory() as tmp:
            write_json(tmp, "x.json", doc)
            cfg = dict(setting("country_records.json"), file="x.json")
            return read_source(cfg, tmp)

    def test_missing_field_in_every_record_stops(self) -> None:
        rec = good_record()
        del rec["measures"]["pop"]
        with self.assertRaises(FileLevelError):
            self.read_fixture({"records": [rec, rec]})

    def test_missing_field_in_one_record_becomes_blank(self) -> None:
        bad = good_record("Fiji", 1992)
        del bad["identity"]["country"]
        df = self.read_fixture({"records": [good_record(), bad]})
        self.assertEqual(len(df), 2)
        self.assertEqual(df["country"].iloc[1], "")

    def test_missing_record_path_stops(self) -> None:
        with self.assertRaises(FileLevelError):
            self.read_fixture({"rows": [good_record()]})

    def test_unparseable_json_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x.json").write_text("{not json", encoding="utf-8")
            cfg = dict(setting("country_records.json"), file="x.json")
            with self.assertRaises(FileLevelError):
                read_source(cfg, tmp)