import json
from pathlib import Path

import pandas as pd


def summarize_dataframe(name: str, df: pd.DataFrame, max_columns: int = 20) -> dict:
    missing = df.isna().sum()
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
    summary = {
        "name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": dtypes,
        "missing_values": {
            column: int(count)
            for column, count in missing[missing > 0].sort_values(ascending=False).head(max_columns).items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "sar" in df.columns:
        summary["label_distribution"] = {
            str(key): int(value) for key, value in df["sar"].value_counts(dropna=False).to_dict().items()
        }
    return summary


def write_eda_report(report: dict[str, dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
