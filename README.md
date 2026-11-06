# Four-Format Data Recovery and Validation (Group 5, 2DSA1)

DSA4153 Week 10 Group Challenge. Recovers the same Gapminder dataset from
CSV, XLSX, JSON and Parquet, validates and reconciles them, then produces
analysis outputs.

Repository: https://github.com/hzShaZ/Four-Format-Data-Recovery-and-Validation--Group5_2DSA1

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

## Code reviews

Coder and reviewer pairs:

- Shan (1) is reviewed by Win (2)
- Win (2) is reviewed by Mac (3)
- Mac (3) is reviewed by Gab (4)
- Gab (4) is reviewed by Marc (5)
- Marc (5) is reviewed by Shan (1)


## AI Assistance Disclosure

Generative AI tools were used during this project for:

- Debugging, error analysis and syntax troubleshooting in the parsing modules.
- Drafting unit test structures and edge-case test scenarios.
- Reviewing documentation clarity and formatting.

All code and results were reviewed, run and verified by the group members.
