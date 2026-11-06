# Four-Format Data Recovery and Validation (Group 5, 2DSA1)

DSA4153 Week 10 Group Challenge. Recovers the same Gapminder dataset from
CSV, XLSX, JSON and Parquet, validates and reconciles them, then produces
analysis outputs.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python main.py --config config.json
```

## Tests
```
python -m unittest discover tests -v
```

## Members
1. Shan - CSV and Excel readers (branch m1-csv-excel)
2. Win - JSON and Parquet readers (branch m2-json-parquet)
3. Mac - validation, quarantine, duplicates (branch m3-validation)
4. Gab - pandas and NumPy analysis (branch m4-analysis)
5. Marc - exports, staging, main.py, audit log (branch m5-exports-cli)




AI Assistance Disclosure
Generative AI tools were utilized during the development of this project for:

- Code debugging, error analysis, and syntax troubleshooting across parsing modules.

- Assisting in the drafting of unit test structures and edge-case test scenarios.

- Reviewing documentation clarity and formatting.

All outputs, logic implementations, and final analytical conclusions were independently reviewed, validated, and verified by the group members.
