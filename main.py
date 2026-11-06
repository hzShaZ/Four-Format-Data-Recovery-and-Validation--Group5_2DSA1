"""Entry point: python main.py --config config.json"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from src.analysis import run_analysis, write_analysis_files
from src.exports import (
    compare_tables, export_canonical, make_audit, make_check, promote_staging,
    verify_exports, write_failed_report, write_rows,
)
from src.readers import read_source
from src.validation import (
    AUDIT_COLUMNS, CHECK_COLUMNS, DUPLICATE_COLUMNS, QUARANTINE_COLUMNS,
    ConflictingKeyError, domain_checks, validate_records,
)


def ensure_passed(checks: list) -> None:
    """Stop the run if any check so far has failed."""
    bad = [c["check"] for c in checks if not c["pass"]]
    if bad:
        raise RuntimeError(f"{len(bad)} check(s) failed, first: {bad[0]}")


def build_tables(config: dict, manifest: dict, checks: list, audit: list,
                 quarantine: list, duplicates: list) -> dict:
    """Read and validate every input; return one clean table per file."""
    expected = manifest["expected_per_file"]
    tables = {}
    for settings in config["inputs"]:
        name = settings["file"]
        raw = read_source(settings, config["data_dir"])
        audit.append(make_audit(0, name, "read", settings["format"], len(raw), len(raw)))
        try:
            clean, result = validate_records(raw, manifest, name)
        except ConflictingKeyError as exc:
            # keep what was found before the stop, so the failure report has it
            quarantine.extend(exc.result.quarantine)
            audit.extend(exc.result.audit)
            raise
        quarantine.extend(result.quarantine)
        duplicates.extend(result.duplicates)
        audit.extend(result.audit)
        checks.extend(result.reconciliation_checks(expected))
        checks.extend(domain_checks(clean, manifest, name))
        tables[name] = clean
    return tables


def run(config: dict) -> int:
    """Run the whole pipeline. Return 0 on success and 1 on any failure."""
    staging = Path(config["staging_dir"])
    output = Path(config["output_dir"])
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    checks, audit, quarantine, duplicates = [], [], [], []
    try:
        manifest = json.loads((Path(config["data_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        schema = manifest["canonical_schema"]
        n_files = len(config["inputs"])

        tables = build_tables(config, manifest, checks, audit, quarantine, duplicates)
        checks.append(make_check("all inputs: quarantined records",
                                 manifest["expected_per_file"]["rejected_rows"] * n_files,
                                 len(quarantine), 0,
                                 len(quarantine) == manifest["expected_per_file"]["rejected_rows"] * n_files))
        checks.append(make_check("all inputs: duplicate copies removed",
                                 manifest["expected_per_file"]["exact_duplicates_removed"] * n_files,
                                 len(duplicates), 0,
                                 len(duplicates) == manifest["expected_per_file"]["exact_duplicates_removed"] * n_files))

        names = list(tables)
        for name in names[1:]:
            checks.extend(compare_tables(tables[names[0]], tables[name],
                                         f"agreement {names[0]} vs {name}", schema))
        ensure_passed(checks)
        canonical = tables[names[0]]
        audit.append(make_audit(0, "all inputs", "compare four formats",
                                "keys, text, integers exact; floats atol 1e-8",
                                len(canonical) * n_files, len(canonical)))

        year = config.get("reporting_year", 2007)
        analysis = run_analysis(canonical, year, controls=manifest["controls"])
        checks.extend(analysis["checks"])
        write_analysis_files(analysis["tables"], staging)
        audit.append(make_audit(0, "canonical", "analysis", f"reporting year {year}",
                                len(canonical), len(canonical)))
        print("matrix shapes:", analysis["shapes"])

        paths = export_canonical(canonical, staging)
        checks.extend(verify_exports(canonical, paths, schema))
        audit.append(make_audit(0, "canonical", "export and reload", "four formats, schema reapplied",
                                len(canonical), len(canonical)))
        ensure_passed(checks)

        for number, record in enumerate(audit, start=1):
            record["step"] = number
        write_rows(quarantine, staging / "quarantine.csv", QUARANTINE_COLUMNS)
        write_rows(duplicates, staging / "duplicate_log.csv", DUPLICATE_COLUMNS)
        write_rows(checks, staging / "validation.csv", CHECK_COLUMNS)
        write_rows(audit, staging / "audit_log.csv", AUDIT_COLUMNS)
        promote_staging(staging, output)
        (output / "failed_validation_report.csv").unlink(missing_ok=True)
    except Exception as exc:
        report = write_failed_report(checks, f"{type(exc).__name__}: {exc}", output)
        shutil.rmtree(staging, ignore_errors=True)
        print(f"FAILED: {exc}\nReport: {report}", file=sys.stderr)
        return 1
    print(f"All checks passed ({len(checks)} checks). Outputs in {output}")
    return 0


def main() -> int:
    """Read the config file and run the pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = json.load(fh)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
