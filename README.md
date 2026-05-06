# Financial Transaction Anomaly Detection

This project focuses on detecting suspicious financial transactions 
in a real-world banking scenario, similar to anti-money laundering (AML) systems.

The system was developed using real banking data from a data competition, 
simulating a production-like fraud detection workflow.

It provides an end-to-end machine learning pipeline that transforms raw 
customer, account, and transaction data into actionable risk signals.

## Key Highlights

- End-to-end ML pipeline for financial anomaly detection
- Supports both domestic (NTD) and foreign currency transaction flows
- Modular pipeline design with reusable feature engineering components
- Strong recall performance on highly imbalanced fraud dataset
- Reproducible artifacts for analysis and deployment

It supports two pipelines:

- `nt`: Taiwan-dollar transaction pipeline
- `foreign`: Foreign-currency transaction pipeline

Each pipeline follows the same high-level flow:

1. Load source tables
2. Generate EDA summaries
3. Prepare and clean data
4. Build feature tables
5. Train a model
6. Save artifacts for later analysis

## Feature Engineering Overview

The feature engineering in this project is designed to capture abnormal behavioral patterns in financial transactions, rather than relying on raw values alone.

In particular, the features aim to identify:

- unusually high transaction frequency
- inconsistent debit/credit behavior
- abnormal transaction-to-balance relationships
- irregular timing patterns between transactions
- unusual activity across multiple currencies

Key feature groups in the `nt` pipeline include:

- account status indicator features, such as whether certain warning or restriction flags were ever triggered
- transaction count statistics per account, including daily mean, variance, and total transaction counts
- transaction amount statistics, including average, standard deviation, and total transferred amount
- debit versus credit behavior features, such as count ratios and amount differences between transaction directions
- balance-related features, such as differences between current balance and reference balances
- ratio features, such as transaction amount relative to account balance
- transaction timing features, such as average and standard deviation of time gaps between transactions

Key feature groups in the `foreign` pipeline include:

- foreign account profile features, such as account status flags, transaction counters, and account-level debit or credit totals
- currency portfolio features, such as total foreign balance, maximum held balance, number of active currencies, and indicators for major currencies like USD, JPY, AUD, and CNY
- foreign transaction activity features, such as transaction counts, active days, memo-code diversity, channel diversity, and transactions per active day
- directional transaction features, such as debit versus credit count ratios, amount ratios, and amount gaps
- transaction amount and exchange-rate statistics, such as mean, max, min, sum, and settlement-to-amount ratios
- transaction time-gap features based on intervals between consecutive foreign transactions


## Training Result

One of the current recorded runs for the `nt` pipeline produced the following model artifact:

- `artifacts/main_pipeline/nt/model/model_bundle.pkl`

Rounded evaluation metrics:

| Metric | Value |
| --- | ---: |
| Train Accuracy | 0.9995 |
| Train Recall | 1.0000 |
| Train Precision | 0.9801 |
| Test Accuracy | 0.9990 |
| Test Recall | 0.9286 |
| Test Precision | 0.7647 |
| Test Positive Predictions | 34 |

These results suggest that the model performs very strongly on the available NT dataset, with especially high recall on the positive class while maintaining strong overall accuracy.

One of the current recorded runs for the `foreign` pipeline produced the following model artifact:

- `artifacts/main_pipeline/foreign/model/model_bundle.pkl`

Rounded evaluation metrics:

| Metric | Value |
| --- | ---: |
| Train Accuracy | 0.9995 |
| Train Recall | 1.0000 |
| Train Precision | 0.9231 |
| Test Accuracy | 0.9967 |
| Test Recall | 1.0000 |
| Test Precision | 0.5000 |
| Test Positive Predictions | 2 |

Because the currently available foreign abnormal sample is very small, these metrics should be interpreted carefully. They are useful as a pipeline verification result, but they are less stable than the NT results.

## Project Structure

```text
.
├── main.py
├── README.md
├── data/
│   ├── abnormal/
│   └── normal/
├── artifacts/
│   └── main_pipeline/
├── foreign_pipeline/
│   ├── assemble.py
│   ├── config.py
│   ├── io.py
│   ├── model.py
│   ├── prepare.py
│   ├── run_foreign_pipeline.py
│   └── train.py
├── notebooks/
├── process/
└── nt_pipeline/
    ├── account_features.py
    ├── config.py
    ├── eda.py
    ├── foreign_join.py
    ├── io.py
    ├── join_nt.py
    ├── nt_assemble.py
    ├── nt_features.py
    ├── nt_model.py
    ├── prepare.py
    ├── run_foreign_pipeline.py
    ├── run_nt_pipeline.py
    ├── train_foreign.py
    ├── train_nt.py
    └── transaction_features.py
```

## What the Main Entry Point Does

The recommended entry point is `main.py`.

It can run:

- only the `nt` pipeline
- only the `foreign` pipeline
- both pipelines in one execution

For each selected pipeline, `main.py`:

- validates required input files
- loads raw or precomputed feature tables
- writes EDA reports as JSON
- saves cleaned intermediate CSV files
- builds training features
- trains the model
- exports a serialized model bundle and evaluation metrics

## Data Layout

By default, the code expects the dataset root to be `Data`, but this repository currently stores sample data under `data`.

If you use the files already included in this repository, pass `--data-root ./data`.

### NT pipeline inputs

The `nt` pipeline expects these files under the data root:

- `normal/SAMASTER.csv`
- `normal/SATXNREC.csv`
- `normal/CFMASTER.csv`
- `normal/CFACCOUNT.csv`
- `abnormal/SAMASTER_SAR.csv`
- `abnormal/SATXNREC_SAR.csv`
- `abnormal/CFMASTER_SAR.csv`
- `abnormal/CFACCOUNT_SAR.csv`

### Foreign pipeline inputs

The `foreign` pipeline reads raw source tables directly from the data root and generates its own feature tables in code.

Specifically, it looks for:

- `normal/FSTXN.csv`
- `abnormal/FSTXN_SAR.csv`
- `normal/FSCST.csv`
- `abnormal/FSCST_SAR.csv`
- `normal/CFMASTER.csv`
- `normal/CFACCOUNT.csv`
- `abnormal/CFMASTER_SAR.csv`
- `abnormal/CFACCOUNT_SAR.csv`

## Environment Setup

Create a Python virtual environment with `venv` or Conda, then install the libraries used by the project.

This repository does not currently include a dependency lock file such as `requirements.txt` or `pyproject.toml`, so you will need to install the required packages manually.

The code imports these main libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib category_encoders
```

## Usage

### CLI arguments

`main.py` supports the following arguments:

- `--data-root`: root directory of the input data. Default: `Data`
- `--output-root`: output directory for generated artifacts. Default: `artifacts/main_pipeline`
- `--pipeline`: which pipeline to run. Choices: `nt`, `foreign`, `both`. Default: `both`

### Run both pipelines

```bash
python main.py --data-root ./data --pipeline both
```

### Run only the NT pipeline

```bash
python main.py --data-root ./data --pipeline nt
```

### Run only the foreign pipeline

```bash
python main.py --data-root ./data --pipeline foreign
```

### Write outputs to a custom location

```bash
python main.py --data-root ./data --output-root ./artifacts/experiment_01 --pipeline nt
```

## Output Artifacts

Results are written under:

- `artifacts/main_pipeline/nt/` for the NT pipeline
- `artifacts/main_pipeline/foreign/` for the foreign pipeline

Each pipeline output directory may contain:

```text
<pipeline-output>/
├── cleaned/
├── eda/
│   ├── raw_eda.json
│   ├── cleaned_eda.json
│   └── training_eda.json
├── features/
└── model/
    ├── model_bundle.pkl
    └── metrics.json
```

### Output details

- `cleaned/`: cleaned intermediate CSV files
- `eda/raw_eda.json`: summary of source inputs
- `eda/cleaned_eda.json`: summary after preparation and cleaning
- `eda/training_eda.json`: summary of the final training table
- `features/`: generated training feature tables
- `model/model_bundle.pkl`: serialized training bundle
- `model/metrics.json`: model evaluation metrics

## Pipeline Notes

### NT pipeline

The NT pipeline builds multiple intermediate feature tables, including:

- `feature_NT_account.csv`
- `feature_NT_account_SAR.csv`
- `feature_NT_transaction.csv`
- `feature_NT_transaction_SAR.csv`
- `feature_NT_all.csv`
- `feature_NT_cus_acc.csv`

The final training table used by the NT model is:

- `features/feature_NT_cus_acc.csv`

### Foreign pipeline

The foreign pipeline builds intermediate feature tables and the final training table:

- `features/fscst_features.csv`
- `features/fscst_sar_features.csv`
- `features/fstxn_features.csv`
- `features/fstxn_sar_features.csv`
- `features/feature_for_cus_acc.csv`

## Legacy Scripts

The repository also includes:

- `nt_pipeline/run_nt_pipeline.py`
- `nt_pipeline/run_foreign_pipeline.py`

These are older standalone entry points. For normal usage, prefer `main.py`, which provides a unified interface for both pipelines.

## Notes and Limitations

- The default `--data-root` in code is `Data`, but this repository uses `data`, so pass `--data-root ./data` unless you rename the folder.
- The foreign pipeline is now organized under `foreign_pipeline/`, parallel to `nt_pipeline/`.
- The foreign pipeline now generates its feature tables from raw `FSTXN` and `FSCST` data inside the codebase before training.
- This repository currently includes sample artifacts under `artifacts/main_pipeline/`, which can be useful as a reference for expected outputs.
