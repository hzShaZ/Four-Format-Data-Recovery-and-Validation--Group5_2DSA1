"""Checks for row validation, quarantine, duplicates and conflicts (member 3).

Fixtures are built in memory, so nothing generated is written under data/.
Run with:  python -m unittest tests.test_validation -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from src.readers import CANONICAL, FileLevelError, read_csv_source, read_source
from src.validation import (
    QUARANTINE_COLUMNS,
    ConflictingKeyError,
    RunResult,
    check_row,
    domain_checks,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
RULES = {
    "allowed_continents": MANIFEST["allowed_continents"],
    "allowed_years": MANIFEST["allowed_years"],
    "life_exp_max": 120,
}
GOOD = {"country": "Chile", "continent": "Americas", "year": "1997",
        "lifeExp": "74.126", "pop": "14599929", "gdpPercap": "10118.05318"}


def raw_table(*rows: dict) -> pd.DataFrame:
    """Build a raw string table; each row overrides fields of GOOD."""
    data = [{**GOOD, **r} for r in rows]
    df = pd.DataFrame(data, columns=CANONICAL)
    df.insert(0, "source_record", range(1, len(df) + 1))
    return df


def reasons_for(**override: str) -> list:
    """Validate a single record and return its rejection reasons."""
    clean, result = validate_records(raw_table(override), RULES, "t.csv")
    return [] if len(clean) else [result.quarantine[0]["reason"]]


class TestRowRules(unittest.TestCase):
    def test_good_row_is_typed_and_trimmed(self) -> None:
        values, why = check_row({**GOOD, "country": "  Chile \t"}, RULES)
        self.assertEqual(why, [])
        self.assertEqual(values["country"], "Chile")
        self.assertIsInstance(values["year"], int)
        self.assertIsInstance(values["pop"], int)
        self.assertIsInstance(values["lifeExp"], float)

    def test_missing_key_parts(self) -> None:
        for bad in ({"country": ""}, {"country": "   "}, {"year": ""}, {"year": None}):
            with self.subTest(bad=bad):
                self.assertEqual(len(reasons_for(**bad)), 1)

    def test_missing_other_required_values(self) -> None:
        for name in ("continent", "lifeExp", "pop", "gdpPercap"):
            with self.subTest(name=name):
                self.assertIn(name, reasons_for(**{name: ""})[0])

    def test_continent_must_be_in_allowed_set(self) -> None:
        self.assertIn("continent", reasons_for(continent="Atlantis")[0])
        self.assertIn("continent", reasons_for(continent="europe")[0])

    def test_unparseable_numbers(self) -> None:
        for text in ("abc", "12abc", "1,234", "1_000", "--5", "0x10", "1.2.3", "."):
            for name in ("year", "lifeExp", "pop", "gdpPercap"):
                with self.subTest(name=name, text=text):
                    self.assertIn(name, reasons_for(**{name: text})[0])

    def test_fractional_year_and_population(self) -> None:
        self.assertIn("fractional", reasons_for(year="1997.5")[0])
        self.assertIn("fractional", reasons_for(pop="100.5")[0])
        self.assertIn("fractional", reasons_for(pop="1e-1")[0])

    def test_integral_text_forms_are_accepted(self) -> None:
        values, why = check_row({**GOOD, "year": "1997.0", "pop": "1.4599929e7"}, RULES)
        self.assertEqual(why, [])
        self.assertEqual((values["year"], values["pop"]), (1997, 14599929))

    def test_year_must_be_allowed(self) -> None:
        for year in ("1953", "2008", "0", "-1997"):
            with self.subTest(year=year):
                self.assertIn("year", reasons_for(year=year)[0])

    def test_non_finite_values(self) -> None:
        for text in ("nan", "NaN", "inf", "-inf", "Infinity", "1e999"):
            for name in ("year", "lifeExp", "pop", "gdpPercap"):
                with self.subTest(name=name, text=text):
                    self.assertIn(name, reasons_for(**{name: text})[0])
        self.assertIn("non-finite", reasons_for(lifeExp="nan")[0])

    def test_out_of_range_values(self) -> None:
        for name, text in [("lifeExp", "0"), ("lifeExp", "-3"), ("lifeExp", "120.01"),
                           ("lifeExp", "150"), ("pop", "0"), ("pop", "-1"),
                           ("pop", "9223372036854775808"), ("pop", "1e30"),
                           ("pop", "1e999999999"), ("gdpPercap", "0"), ("gdpPercap", "-5")]:
            with self.subTest(name=name, text=text):
                self.assertIn(name, reasons_for(**{name: text})[0])

    def test_range_boundaries_are_accepted(self) -> None:
        for override in ({"lifeExp": "120"}, {"lifeExp": "0.001"}, {"pop": "1"},
                         {"pop": "9223372036854775807"}, {"gdpPercap": "0.0001"}):
            with self.subTest(override=override):
                self.assertEqual(reasons_for(**override), [])

    def test_python_missing_markers(self) -> None:
        self.assertIn("lifeExp", reasons_for(lifeExp=None)[0])
        self.assertIn("pop", reasons_for(pop=float("nan"))[0])

    def test_native_numbers_are_accepted(self) -> None:
        values, why = check_row({**GOOD, "year": 1997, "lifeExp": 74.126, "pop": 14599929}, RULES)
        self.assertEqual(why, [])
        self.assertEqual(values["pop"], 14599929)


class TestQuarantine(unittest.TestCase):
    def test_source_record_is_one_based_position(self) -> None:
        raw = raw_table({}, {"year": "1952", "pop": "x"}, {"year": "1957"},
                        {"year": "1962", "lifeExp": "0"})
        clean, result = validate_records(raw, RULES, "t.csv")
        self.assertEqual([q["source_record"] for q in result.quarantine], [2, 4])
        self.assertEqual(len(clean), 2)

    def test_original_values_are_preserved_untrimmed(self) -> None:
        raw = raw_table({"country": "  Chile  ", "pop": " 12.5 "})
        _, result = validate_records(raw, RULES, "t.csv")
        row = result.quarantine[0]
        self.assertEqual(row["country"], "  Chile  ")
        self.assertEqual(row["pop"], " 12.5 ")
        self.assertEqual(row["source_file"], "t.csv")

    def test_all_reasons_reported_for_one_row(self) -> None:
        _, result = validate_records(raw_table({"year": "x", "pop": "-1"}), RULES, "t.csv")
        reason = result.quarantine[0]["reason"]
        self.assertIn("year", reason)
        self.assertIn("pop", reason)

    def test_quarantine_frame_layout(self) -> None:
        _, result = validate_records(raw_table({"pop": "x"}), RULES, "t.csv")
        frame = result.quarantine_frame()
        self.assertEqual(list(frame.columns), QUARANTINE_COLUMNS)
        self.assertEqual(QUARANTINE_COLUMNS[:3], ["source_file", "source_record", "reason"])
        self.assertEqual(QUARANTINE_COLUMNS[3:], CANONICAL)

    def test_empty_quarantine_frame_keeps_columns(self) -> None:
        _, result = validate_records(raw_table({}), RULES, "t.csv")
        self.assertEqual(list(result.quarantine_frame().columns), QUARANTINE_COLUMNS)
        self.assertEqual(len(result.quarantine_frame()), 0)

    def test_rejected_row_never_reaches_clean_table(self) -> None:
        clean, _ = validate_records(raw_table({}, {"year": "1952", "country": ""}), RULES, "t")
        self.assertEqual(len(clean), 1)

    def test_missing_raw_column_is_file_level_error(self) -> None:
        with self.assertRaises(FileLevelError):
            validate_records(raw_table({}).drop(columns=["pop"]), RULES, "t.csv")


class TestDuplicates(unittest.TestCase):
    def test_exact_duplicate_removed_and_logged(self) -> None:
        raw = raw_table({}, {"year": "1952"}, {})
        clean, result = validate_records(raw, RULES, "t.csv")
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(result.duplicates), 1)
        dup = result.duplicates[0]
        self.assertEqual((dup["source_record"], dup["kept_source_record"]), (3, 1))
        self.assertEqual(len(result.duplicate_frame()), 1)

    def test_every_removed_copy_is_logged(self) -> None:
        clean, result = validate_records(raw_table({}, {}, {}, {}), RULES, "t.csv")
        self.assertEqual(len(clean), 1)
        self.assertEqual([d["source_record"] for d in result.duplicates], [2, 3, 4])

    def test_equal_numbers_in_different_text_are_duplicates(self) -> None:
        raw = raw_table({}, {"year": "1997.0", "lifeExp": "74.1260", "country": " Chile"})
        clean, result = validate_records(raw, RULES, "t.csv")
        self.assertEqual((len(clean), len(result.duplicates)), (1, 1))

    def test_duplicate_of_a_rejected_row_is_not_a_duplicate(self) -> None:
        raw = raw_table({"pop": "x"}, {"pop": "x"})
        clean, result = validate_records(raw, RULES, "t.csv")
        self.assertEqual((len(clean), len(result.duplicates), len(result.quarantine)), (0, 0, 2))

    def test_counts_reconcile(self) -> None:
        raw = raw_table({}, {}, {"year": "1952"}, {"year": "1957", "pop": "-1"})
        clean, result = validate_records(raw, RULES, "t.csv")
        c = result.counts
        self.assertEqual(c["raw_rows"], c["rejected_rows"] + c["exact_duplicates_removed"] + len(clean))
        self.assertTrue(all(check["pass"] for check in result.reconciliation_checks(c)))

    def test_reconciliation_flags_wrong_expectation(self) -> None:
        _, result = validate_records(raw_table({}, {}), RULES, "t.csv")
        wrong = dict(result.counts, exact_duplicates_removed=2)
        failed = [c for c in result.reconciliation_checks(wrong) if not c["pass"]]
        self.assertEqual(len(failed), 1)


class TestConflicts(unittest.TestCase):
    def test_same_key_different_value_stops(self) -> None:
        raw = raw_table({}, {"lifeExp": "75.0"})
        with self.assertRaises(ConflictingKeyError) as ctx:
            validate_records(raw, RULES, "t.csv")
        err = ctx.exception
        self.assertIsInstance(err, FileLevelError)
        self.assertEqual(err.conflicts[0]["key"], ("Chile", 1997))
        self.assertEqual(err.conflicts[0]["records"], (1, 2))
        self.assertEqual(err.conflicts[0]["fields"], ["lifeExp"])

    def test_conflict_in_any_field_is_detected(self) -> None:
        for name, other in [("continent", "Europe"), ("pop", "1"), ("gdpPercap", "2")]:
            with self.subTest(name=name), self.assertRaises(ConflictingKeyError):
                validate_records(raw_table({}, {name: other}), RULES, "t.csv")

    def test_conflict_is_not_resolved_by_keeping_first(self) -> None:
        raw = raw_table({}, {"pop": "5"}, {})
        with self.assertRaises(ConflictingKeyError):
            validate_records(raw, RULES, "t.csv")

    def test_error_carries_partial_result_for_the_failure_report(self) -> None:
        raw = raw_table({"pop": "x"}, {}, {"lifeExp": "75"})
        with self.assertRaises(ConflictingKeyError) as ctx:
            validate_records(raw, RULES, "t.csv")
        result = ctx.exception.result
        self.assertIsInstance(result, RunResult)
        self.assertEqual(len(result.quarantine), 1)
        self.assertIn("FAILED", result.audit[-1]["rule"])

    def test_invalid_row_cannot_create_a_conflict(self) -> None:
        clean, _ = validate_records(raw_table({}, {"lifeExp": "-1"}), RULES, "t.csv")
        self.assertEqual(len(clean), 1)

    def test_same_country_different_year_is_not_a_conflict(self) -> None:
        clean, _ = validate_records(raw_table({}, {"year": "2002", "pop": "9"}), RULES, "t.csv")
        self.assertEqual(len(clean), 2)

    def test_conflicting_key_file_stops(self) -> None:
        settings = {"file": "conflicting_key.csv", "format": "csv", "encoding": "utf-8",
                    "separator": ",", "mapping": {c: c for c in CANONICAL}}
        try:
            raw = read_csv_source(DATA / "conflicting_key.csv", settings)
        except NotImplementedError:
            self.skipTest("CSV reader (member 1) is not implemented yet")
        with self.assertRaises(ConflictingKeyError):
            validate_records(raw, RULES, "conflicting_key.csv")


class TestOutputShape(unittest.TestCase):
    def test_clean_table_types_order_and_sorting(self) -> None:
        raw = raw_table({"country": "Peru"}, {"year": "1952"}, {"country": "Peru", "year": "1952"},
                        {"country": "Bolivia", "year": "2007"})
        clean, _ = validate_records(raw, RULES, "t.csv")
        self.assertEqual(list(clean.columns), CANONICAL)
        self.assertEqual(clean["year"].dtype, "int64")
        self.assertEqual(clean["pop"].dtype, "int64")
        self.assertEqual(clean["lifeExp"].dtype, "float64")
        self.assertEqual(list(zip(clean["country"], clean["year"])),
                         [("Bolivia", 2007), ("Chile", 1952), ("Peru", 1952), ("Peru", 1997)])
        self.assertEqual(list(clean.index), [0, 1, 2, 3])

    def test_audit_records_are_a_list_of_complete_dicts(self) -> None:
        _, result = validate_records(raw_table({}, {"pop": "x"}), RULES, "t.csv")
        self.assertIsInstance(result.audit, list)
        for rec in result.audit:
            self.assertEqual(set(rec), {"step", "source_file", "operation", "rule",
                                        "rows_before", "rows_after"})
        self.assertEqual([r["step"] for r in result.audit], list(range(1, len(result.audit) + 1)))
        self.assertEqual((result.audit[0]["rows_before"], result.audit[0]["rows_after"]), (2, 1))

    def test_domain_checks_pass_on_clean_and_fail_on_tampered(self) -> None:
        clean, _ = validate_records(raw_table({}, {"year": "1952"}), RULES, "t.csv")
        self.assertTrue(all(c["pass"] for c in domain_checks(clean, RULES, "t.csv")))
        bad = clean.copy()
        bad.loc[0, "pop"] = -5
        bad.loc[1, "year"] = 1999
        failed = {c["check"] for c in domain_checks(bad, RULES, "t.csv") if not c["pass"]}
        self.assertEqual(len(failed), 2)

    def test_reordered_records_give_the_same_result(self) -> None:
        rows = [{"year": str(y), "country": c} for y in (1952, 1957, 1962) for c in ("A", "B", "C")]
        rows += [rows[0], rows[4], {"year": "x"}, {"country": ""}]
        base = raw_table(*rows)
        want, want_res = validate_records(base, RULES, "t.csv")
        for seed in range(5):
            shuffled = base.sample(frac=1, random_state=seed)
            shuffled["source_record"] = range(1, len(shuffled) + 1)
            got, got_res = validate_records(shuffled.reset_index(drop=True), RULES, "t.csv")
            pd.testing.assert_frame_equal(got, want)
            self.assertEqual(got_res.counts, want_res.counts)


def format_settings(entry: dict) -> dict:
    """Turn a manifest file entry into a reader settings dictionary."""
    settings = {k: v for k, v in entry.items() if k != "source_to_canonical"}
    settings["mapping"] = entry["source_to_canonical"]
    return settings


class TestRealFiles(unittest.TestCase):
    """Run the whole reader -> validation chain on the four supplied inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tables: dict = {}
        cls.results: dict = {}
        cls.skipped: dict = {}
        for entry in MANIFEST["files"]:
            try:
                raw = read_source(format_settings(entry), str(DATA))
            except (NotImplementedError, ImportError) as exc:
                cls.skipped[entry["file"]] = repr(exc)
                continue
            cls.tables[entry["file"]], cls.results[entry["file"]] = validate_records(
                raw, RULES, entry["file"])

    def test_at_least_csv_and_excel_ran(self) -> None:
        missing = [n for n in ("country_records.csv", "country_records.xlsx")
                   if n not in self.tables]
        if missing and all("NotImplementedError" in self.skipped.get(n, "") for n in missing):
            self.skipTest(f"readers not implemented yet: {missing}")
        self.assertEqual(missing, [])

    def test_reconciliation_for_each_available_input(self) -> None:
        expected = MANIFEST["expected_per_file"]
        for name, result in self.results.items():
            with self.subTest(file=name):
                self.assertEqual(result.counts["raw_rows"], expected["raw_rows"])
                self.assertEqual(result.counts["rejected_rows"], expected["rejected_rows"])
                self.assertEqual(result.counts["valid_before_deduplication"],
                                 expected["valid_before_deduplication"])
                self.assertEqual(result.counts["exact_duplicates_removed"],
                                 expected["exact_duplicates_removed"])
                self.assertEqual(result.counts["canonical_rows"], expected["canonical_rows"])
                self.assertTrue(all(c["pass"] for c in result.reconciliation_checks(expected)))
                self.assertTrue(all(c["pass"] for c in domain_checks(
                    self.tables[name], RULES, name)))

    def test_controls_for_each_available_input(self) -> None:
        ctl = MANIFEST["controls"]
        for name, df in self.tables.items():
            with self.subTest(file=name):
                self.assertEqual(df["country"].nunique(), ctl["countries"])
                phil = df.loc[df["country"] == "Philippines"]
                self.assertEqual(len(phil), ctl["philippines_rows"])
                self.assertEqual(int(phil.loc[phil["year"] == 2007, "pop"].iloc[0]),
                                 ctl["philippines_2007_population"])
                asia = df.loc[(df["continent"] == "Asia") & (df["year"] == 2007)]
                self.assertEqual(len(asia), ctl["asia_2007_rows"])
                self.assertEqual(int(asia["pop"].sum()), ctl["asia_2007_population"])

    def test_available_inputs_agree_exactly(self) -> None:
        frames = list(self.tables.values())
        for other in frames[1:]:
            pd.testing.assert_frame_equal(frames[0], other, check_exact=True)

    def test_quarantine_rows_differ_in_position_but_not_in_content(self) -> None:
        if not self.results:
            self.skipTest("no reader is implemented yet")
        contents = {
            name: sorted(tuple(q[c] for c in CANONICAL) for q in res.quarantine)
            for name, res in self.results.items()
        }
        first = next(iter(contents.values()))
        self.assertTrue(all(c == first for c in contents.values()))
        self.assertEqual(len(first), MANIFEST["expected_per_file"]["rejected_rows"])

    def test_total_quarantine_and_duplicates_across_available_inputs(self) -> None:
        n = len(self.results)
        self.assertEqual(sum(len(r.quarantine) for r in self.results.values()), 6 * n)
        self.assertEqual(sum(len(r.duplicates) for r in self.results.values()), 2 * n)


if __name__ == "__main__":
    unittest.main()