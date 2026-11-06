import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import main
from src.exports import (compare_tables, export_canonical, promote_staging,
                         verify_exports, write_failed_report)
from src.readers import CANONICAL

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
SCHEMA = MANIFEST["canonical_schema"]


def sample() -> pd.DataFrame:
    """Two clean rows for the round trip tests."""
    return pd.DataFrame({
        "country": ["Chile", "Peru"],
        "continent": ["Americas", "Americas"],
        "year": [2002, 2007],
        "lifeExp": [77.86, 71.421],
        "pop": [15497046, 28674757],
        "gdpPercap": [10778.78, 7408.906],
    })


def run_config(tmp: str, inputs: list, year: int = 2007) -> dict:
    """Config that writes only inside a temporary folder."""
    return {"data_dir": str(ROOT / "data"), "output_dir": str(Path(tmp) / "out"),
            "staging_dir": str(Path(tmp) / "staging"), "reporting_year": year,
            "inputs": inputs}


class TestExports(unittest.TestCase):
    def test_round_trip_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_canonical(sample(), tmp)
            checks = verify_exports(sample(), paths, SCHEMA)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(c["pass"] for c in checks))

    def test_corrupted_export_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_canonical(sample(), tmp)
            csv_path = Path(paths[0])
            csv_path.write_text(csv_path.read_text().replace("71.421", "72.421"))
            checks = verify_exports(sample(), paths, SCHEMA)
            self.assertFalse(all(c["pass"] for c in checks))

    def test_unreadable_export_fails_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_canonical(sample(), tmp)
            Path(paths[2]).write_text("{not json")
            checks = verify_exports(sample(), paths, SCHEMA)
            self.assertFalse(all(c["pass"] for c in checks))

    def test_float_tolerance(self) -> None:
        close, far = sample(), sample()
        close.loc[0, "lifeExp"] += 1e-10
        far.loc[0, "lifeExp"] += 1e-6
        self.assertTrue(all(c["pass"] for c in compare_tables(sample(), close, "t", SCHEMA)))
        self.assertFalse(all(c["pass"] for c in compare_tables(sample(), far, "t", SCHEMA)))

    def test_comparison_ignores_row_order(self) -> None:
        shuffled = sample().iloc[::-1].reset_index(drop=True)
        self.assertTrue(all(c["pass"] for c in compare_tables(sample(), shuffled, "t", SCHEMA)))

    def test_promote_moves_files_and_removes_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            staging, out = Path(tmp) / "staging", Path(tmp) / "out"
            staging.mkdir()
            (staging / "a.csv").write_text("x")
            promote_staging(staging, out)
            self.assertTrue((out / "a.csv").exists())
            self.assertFalse(staging.exists())

    def test_failed_report_has_failed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_failed_report([], "boom", tmp)
            report = pd.read_csv(path)
            self.assertFalse(report["pass"].iloc[-1])


class TestWholeRun(unittest.TestCase):
    def test_normal_run_publishes_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main.run(run_config(tmp, CONFIG["inputs"]))
            out = Path(tmp) / "out"
            self.assertEqual(code, 0)
            self.assertEqual(len(pd.read_csv(out / "canonical.csv")), 1704)
            self.assertEqual(len(pd.read_csv(out / "quarantine.csv")), 24)
            self.assertEqual(len(pd.read_csv(out / "duplicate_log.csv")), 8)
            self.assertEqual(len(pd.read_csv(out / "asia_selected_year.csv")), 33)
            self.assertTrue(pd.read_csv(out / "validation.csv")["pass"].all())
            self.assertFalse((Path(tmp) / "staging").exists())

    def test_reporting_year_1997_needs_no_code_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main.run(run_config(tmp, CONFIG["inputs"], 1997)), 0)
            asia = pd.read_csv(Path(tmp) / "out" / "asia_selected_year.csv")
            self.assertTrue((asia["year"] == 1997).all())

    def test_conflicting_key_stops_and_keeps_old_outputs(self) -> None:
        entry = {"file": "conflicting_key.csv", "format": "csv", "encoding": "utf-8",
                 "separator": ",", "mapping": {c: c for c in CANONICAL}}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            (out / "asia_selected_year.csv").write_text("OLD")
            code = main.run(run_config(tmp, [entry]))
            self.assertEqual(code, 1)
            self.assertEqual((out / "asia_selected_year.csv").read_text(), "OLD")
            self.assertTrue((out / "failed_validation_report.csv").exists())
            self.assertFalse((Path(tmp) / "staging").exists())

    def test_missing_column_file_stops(self) -> None:
        entry = {"file": "missing_column.csv", "format": "csv", "encoding": "utf-8",
                 "separator": ",", "mapping": {c: c for c in CANONICAL}}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main.run(run_config(tmp, [entry])), 1)
            self.assertFalse((Path(tmp) / "out" / "canonical.csv").exists())


if __name__ == "__main__":
    unittest.main()
