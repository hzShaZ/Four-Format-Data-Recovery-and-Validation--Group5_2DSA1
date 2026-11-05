import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    DEFAULT_COUNTRIES,
    DEFAULT_YEARS,
    asia_population_total_numpy,
    asia_selected_year,
    country_history,
    index_to_baseline,
    index_to_baseline_loop,
    population_index_frame,
    population_matrix,
    run_analysis,
    write_analysis_files,
)
from src.readers import CANONICAL, read_source
from src.validation import validate_records

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
DATA = str(ROOT / "data")


def small_table() -> pd.DataFrame:
    """Small canonical table with a life-expectancy tie in Asia 1997."""
    rows = [
        ("Japan", "Asia", 1997, 80.0, 126_000_000, 30000.0),
        ("Korea", "Asia", 1997, 75.0, 46_000_000, 15000.0),
        ("Nepal", "Asia", 1997, 75.0, 23_000_000, 800.0),
        ("India", "Asia", 2007, 64.7, 1_100_000_000, 2500.0),
        ("France", "Europe", 1997, 78.0, 59_000_000, 25000.0),
        ("Philippines", "Asia", 1982, 64.0, 52_000_000, 2000.0),
        ("Philippines", "Asia", 1992, 67.0, 65_000_000, 2300.0),
        ("Philippines", "Asia", 1987, 65.0, 58_000_000, 2100.0),
    ]
    return pd.DataFrame(rows, columns=CANONICAL)


class TestAsiaSelectedYear(unittest.TestCase):
    def test_selects_only_asia_and_the_year(self) -> None:
        out = asia_selected_year(small_table(), 1997)
        self.assertEqual(set(out["country"]), {"Japan", "Korea", "Nepal"})

    def test_rank_uses_min_method_for_ties(self) -> None:
        out = asia_selected_year(small_table(), 1997)
        ranks = dict(zip(out["country"], out["lifeExp_rank"]))
        self.assertEqual(ranks, {"Japan": 1, "Korea": 2, "Nepal": 2})

    def test_sorted_by_rank_then_country(self) -> None:
        out = asia_selected_year(small_table(), 1997)
        self.assertEqual(list(out["country"]), ["Japan", "Korea", "Nepal"])

    def test_population_in_millions(self) -> None:
        out = asia_selected_year(small_table(), 1997)
        self.assertAlmostEqual(out["pop_millions"].iloc[0], 126.0)

    def test_year_without_rows_gives_empty_table(self) -> None:
        self.assertEqual(len(asia_selected_year(small_table(), 1900)), 0)


class TestCountryHistory(unittest.TestCase):
    def test_sorted_by_year_with_change_from_first(self) -> None:
        out = country_history(small_table(), "Philippines")
        self.assertEqual(list(out["year"]), [1982, 1987, 1992])
        self.assertEqual(list(out["lifeExp_change"]), [0.0, 1.0, 3.0])
        self.assertAlmostEqual(out["pop_millions"].iloc[0], 52.0)

    def test_unknown_country_raises(self) -> None:
        with self.assertRaises(ValueError):
            country_history(small_table(), "Atlantis")


class TestNumpyMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array([[100.0, 150.0, 250.0], [200.0, 200.0, 100.0]])

    def test_baseline_shape_and_index_values(self) -> None:
        index, baseline = index_to_baseline(self.matrix)
        self.assertEqual(baseline.shape, (2, 1))
        self.assertEqual(index.shape, (2, 3))
        self.assertTrue(np.allclose(index[0], [100.0, 150.0, 250.0], atol=1e-8, rtol=0))
        self.assertTrue(np.allclose(index[1], [100.0, 100.0, 50.0], atol=1e-8, rtol=0))

    def test_broadcast_matches_independent_loop(self) -> None:
        index, _ = index_to_baseline(self.matrix)
        loop = index_to_baseline_loop(self.matrix)
        self.assertTrue(np.allclose(index, loop, atol=1e-8, rtol=0))

    def test_population_matrix_values_and_order(self) -> None:
        table = small_table()
        matrix = population_matrix(table, ["Philippines"], [1982, 1987, 1992])
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(list(matrix[0]), [52_000_000, 58_000_000, 65_000_000])

    def test_missing_combination_raises(self) -> None:
        with self.assertRaises(ValueError):
            population_matrix(small_table(), ["Philippines"], [1982, 2007])

    def test_index_frame_has_named_year_columns(self) -> None:
        frame = population_index_frame(self.matrix, ["A", "B"], [1982, 1987, 1992])
        self.assertEqual(list(frame.columns), ["country", "1982", "1987", "1992"])


class TestNumpyTotals(unittest.TestCase):
    def test_boolean_selection_matches_pandas(self) -> None:
        table = small_table()
        rows, total = asia_population_total_numpy(table, 1997)
        asia = asia_selected_year(table, 1997)
        self.assertEqual(rows, len(asia))
        self.assertEqual(total, int(asia["pop"].sum()))
        self.assertEqual(total, 126_000_000 + 46_000_000 + 23_000_000)


class TestWriteFiles(unittest.TestCase):
    def test_writes_csv_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_analysis_files({"asia_selected_year": small_table()}, tmp)
            self.assertEqual(len(paths), 1)
            text = Path(paths[0]).read_text(encoding="utf-8")
        self.assertTrue(text.startswith("country,continent,year"))
        

class TestRealData(unittest.TestCase):
    """Run the analysis on the real cleaned JSON data."""

    @classmethod
    def setUpClass(cls) -> None:
        entry = next(s for s in CONFIG["inputs"] if s["file"] == "country_records.json")
        rules = {"allowed_continents": MANIFEST["allowed_continents"],
                 "allowed_years": MANIFEST["allowed_years"]}
        raw = read_source(entry, DATA)
        cls.clean, _ = validate_records(raw, rules, entry["file"])

    def test_clean_table_has_1704_rows(self) -> None:
        self.assertEqual(len(self.clean), 1704)

    def test_asia_2007_matches_manifest_controls(self) -> None:
        controls = MANIFEST["controls"]
        asia = asia_selected_year(self.clean, 2007)
        self.assertEqual(len(asia), controls["asia_2007_rows"])
        self.assertEqual(int(asia["pop"].sum()), controls["asia_2007_population"])
        self.assertEqual(asia_population_total_numpy(self.clean, 2007)[1],
                         controls["asia_2007_population"])

    def test_philippines_history_matches_controls(self) -> None:
        history = country_history(self.clean, "Philippines")
        self.assertEqual(len(history), MANIFEST["controls"]["philippines_rows"])
        self.assertEqual(int(history["pop"].iloc[-1]),
                         MANIFEST["controls"]["philippines_2007_population"])

    def test_changing_year_to_1997_regenerates_analysis(self) -> None:
        asia = asia_selected_year(self.clean, 1997)
        self.assertTrue((asia["year"] == 1997).all())
        self.assertEqual(len(asia), len(asia_selected_year(self.clean, 1997)))
        self.assertEqual(asia_population_total_numpy(self.clean, 1997)[1],
                         int(asia["pop"].sum()))

    def test_run_analysis_all_checks_pass_for_2007_and_1997(self) -> None:
        for year in (2007, 1997):
            result = run_analysis(self.clean, year, DEFAULT_COUNTRIES, DEFAULT_YEARS,
                                  MANIFEST["controls"])
            failed = [c["check"] for c in result["checks"] if not c["pass"]]
            self.assertEqual(failed, [], year)
        self.assertEqual(result["shapes"]["index"], (3, 6))

    def test_control_checks_only_added_for_2007(self) -> None:
        n_2007 = len(run_analysis(self.clean, 2007, controls=MANIFEST["controls"])["checks"])
        n_1997 = len(run_analysis(self.clean, 1997, controls=MANIFEST["controls"])["checks"])
        self.assertEqual(n_2007 - n_1997, 4)


if __name__ == "__main__":
    unittest.main()