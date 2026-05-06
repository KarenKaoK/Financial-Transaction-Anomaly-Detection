import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COMMON_CURRENCIES = ["USD", "JPY", "AUD", "CNY", "ZAR", "HKD", "EUR", "GBP", "NZD", "SGD"]
COMMON_CHANNELS = ["MB", "NB", "XL", "FX", "SP", ""]


def to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def standard_txn_time(time_value: object) -> str:
    if pd.isna(time_value):
        return "00:00:00"
    digits = "".join(ch for ch in str(time_value) if ch.isdigit()).zfill(6)[-6:]
    return f"{digits[:2]}:{digits[2:4]}:{digits[4:]}"


def prepare_transaction_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in ["ACC_RANDOM", "ID_RANDOM"]:
        if column in output.columns:
            output[column] = output[column].astype(str).str.replace(",", "", regex=False)

    numeric_columns = [
        "FSTXN_BR_CODE",
        "FSTXN_TXN_TIME",
        "FSTXN_MEMO_CODE",
        "FSTXN_DB_CR_STAT",
        "FSTXN_TXN_AMT",
        "FSTXN_SAM_BAL",
        "FSTXN_TXN_STAT",
        "FSTXN_PB_STATUS",
        "FSTXN_CASH_AMT",
        "FSTXN_FX_CASH_AMT",
        "FSTXN_TXN_XCRT",
        "FSTXN_CHG_XCRT",
        "FSTXN_CST_XCRT",
        "FSTXN_TMU_SETTLE_AMT",
        "FSTXN_TMU_TXN_KIND",
        "FSTXN_TMU_CST_XCRT",
        "FSTXN_TMU_TXN_XCRT",
        "FSTXN_SKL_FLAG",
        "FSTXN_NCAPT_FLAG",
    ]
    for column in numeric_columns:
        if column in output.columns:
            output[column] = to_numeric_series(output[column])

    output["FSTXN_ENTRY_DATE"] = pd.to_datetime(output["FSTXN_ENTRY_DATE"], errors="coerce")
    output["standard_time"] = output["FSTXN_TXN_TIME"].apply(standard_txn_time)
    return output


def check_merge(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    common_columns = set(base.columns) & set(feature.columns)
    allowed_overlap = {"ACC_RANDOM", "ID_RANDOM"}
    unexpected_overlap = common_columns - allowed_overlap
    if unexpected_overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(unexpected_overlap)}")
    return base.merge(feature, on="ACC_RANDOM", how="left")


def feature_transaction_profile(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby("ACC_RANDOM")
    output = grouped.agg(
        txn_row_count=("ACC_RANDOM", "size"),
        active_days=("FSTXN_ENTRY_DATE", lambda x: x.dt.date.nunique()),
        currency_nunique=("FSTXN_CURCD", "nunique"),
        channel_nunique=("FSTXN_CHANNEL", "nunique"),
        memo_code_nunique=("FSTXN_MEMO_CODE", "nunique"),
    ).reset_index()
    output["txn_per_active_day"] = output["txn_row_count"] / output["active_days"].replace(0, np.nan)
    output["txn_per_active_day"] = output["txn_per_active_day"].fillna(0.0)
    return output


def feature_amount_statistics(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["cash_total_amt"] = output["FSTXN_CASH_AMT"].fillna(0.0) + output["FSTXN_FX_CASH_AMT"].fillna(0.0)
    output["settle_to_txn_ratio"] = output["FSTXN_TMU_SETTLE_AMT"].fillna(0.0) / (
        output["FSTXN_TXN_AMT"].abs().fillna(0.0) + np.finfo(float).eps
    )
    columns = [
        "FSTXN_TXN_AMT",
        "FSTXN_SAM_BAL",
        "FSTXN_CASH_AMT",
        "FSTXN_FX_CASH_AMT",
        "cash_total_amt",
        "FSTXN_TXN_XCRT",
        "FSTXN_CHG_XCRT",
        "FSTXN_CST_XCRT",
        "FSTXN_TMU_SETTLE_AMT",
        "settle_to_txn_ratio",
    ]
    stats = ["mean", "std", "max", "min", "sum"]
    for column in columns:
        for stat in stats:
            output[f"{column}_{stat}"] = output.groupby("ACC_RANDOM")[column].transform(stat)

    result_columns = [f"{column}_{stat}" for column in columns for stat in stats]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + result_columns)
    return output[["ACC_RANDOM"] + result_columns]


def feature_directional_statistics(data: pd.DataFrame) -> pd.DataFrame:
    base = data[["ACC_RANDOM"]].drop_duplicates().copy()
    result = base

    for direction, prefix in [(1, "db_"), (2, "cr_")]:
        subset = data[data["FSTXN_DB_CR_STAT"] == direction]
        grouped = subset.groupby("ACC_RANDOM").agg(
            txn_count=("ACC_RANDOM", "size"),
            txn_amt_sum=("FSTXN_TXN_AMT", "sum"),
            txn_amt_mean=("FSTXN_TXN_AMT", "mean"),
            cash_amt_sum=("FSTXN_CASH_AMT", "sum"),
        ).reset_index()
        grouped = grouped.rename(
            columns={
                "txn_count": f"{prefix}txn_count",
                "txn_amt_sum": f"{prefix}txn_amt_sum",
                "txn_amt_mean": f"{prefix}txn_amt_mean",
                "cash_amt_sum": f"{prefix}cash_amt_sum",
            }
        )
        result = result.merge(grouped, on="ACC_RANDOM", how="left")

    result["db_cr_count_ratio"] = result["db_txn_count"].fillna(0.0) / (result["cr_txn_count"].fillna(0.0) + np.finfo(float).eps)
    result["db_cr_amount_ratio"] = result["db_txn_amt_sum"].fillna(0.0) / (result["cr_txn_amt_sum"].fillna(0.0) + np.finfo(float).eps)
    result["db_cr_amount_diff"] = result["db_txn_amt_sum"].fillna(0.0) - result["cr_txn_amt_sum"].fillna(0.0)
    return result


def feature_currency_indicators(data: pd.DataFrame) -> pd.DataFrame:
    base = data[["ACC_RANDOM"]].drop_duplicates().copy()
    result = base
    for currency in COMMON_CURRENCIES:
        grouped = (
            data.assign(flag=(data["FSTXN_CURCD"].astype(str) == currency).astype(int))
            .groupby("ACC_RANDOM", as_index=False)["flag"]
            .max()
            .rename(columns={"flag": f"txn_has_currency_{currency}"})
        )
        result = result.merge(grouped, on="ACC_RANDOM", how="left")
    return result


def feature_channel_indicators(data: pd.DataFrame) -> pd.DataFrame:
    base = data[["ACC_RANDOM"]].drop_duplicates().copy()
    result = base
    channel_series = data["FSTXN_CHANNEL"].fillna("").astype(str).str.strip()
    for channel in COMMON_CHANNELS:
        label = "blank" if channel == "" else channel.lower()
        grouped = (
            data.assign(flag=(channel_series == channel).astype(int))
            .groupby("ACC_RANDOM", as_index=False)["flag"]
            .max()
            .rename(columns={"flag": f"txn_has_channel_{label}"})
        )
        result = result.merge(grouped, on="ACC_RANDOM", how="left")
    return result


def feature_time_gap(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["standard_time"] = pd.to_datetime(output["standard_time"], format="%H:%M:%S", errors="coerce").dt.time
    output["full_datetime"] = pd.to_datetime(
        output["FSTXN_ENTRY_DATE"].dt.strftime("%Y-%m-%d").fillna("1970-01-01") + " " + output["standard_time"].astype(str),
        errors="coerce",
    )
    output = output.sort_values(["ACC_RANDOM", "full_datetime"])
    output["time_diff_seconds"] = output.groupby("ACC_RANDOM")["full_datetime"].diff().dt.total_seconds()
    result = output.groupby("ACC_RANDOM").agg(
        mean_time_gap_seconds=("time_diff_seconds", "mean"),
        std_time_gap_seconds=("time_diff_seconds", "std"),
        min_time_gap_seconds=("time_diff_seconds", "min"),
    ).reset_index()
    return result.fillna(0.0)


def build_transaction_features_from_df(data: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    prepared = prepare_transaction_data_frame(data)
    base = prepared[["ACC_RANDOM", "ID_RANDOM"]].drop_duplicates(subset=["ACC_RANDOM"])
    feature_frames = [
        feature_transaction_profile(prepared),
        feature_amount_statistics(prepared),
        feature_directional_statistics(prepared),
        feature_currency_indicators(prepared),
        feature_channel_indicators(prepared),
        feature_time_gap(prepared),
    ]

    output = base
    for feature_frame in feature_frames:
        output = check_merge(output, feature_frame)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_transaction_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path, low_memory=False)
    return build_transaction_features_from_df(data, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build foreign-currency transaction features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_transaction_features(args.input, args.output)
