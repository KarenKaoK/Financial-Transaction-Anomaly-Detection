import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def check_merge(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    common_columns = set(base.columns) & set(feature.columns)
    allowed_overlap = {"ACC_RANDOM", "ID_RANDOM"}
    unexpected_overlap = common_columns - allowed_overlap
    if unexpected_overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(unexpected_overlap)}")
    return base.merge(feature, on="ACC_RANDOM", how="left")


def standard_txn_time(time_value: object) -> str:
    time_str = str(time_value).zfill(6)
    return f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"


def feature_ori(data: pd.DataFrame) -> pd.DataFrame:
    output = data.drop_duplicates(subset=["ACC_RANDOM", "SATXN_BR_CODE"])
    return output[["ACC_RANDOM", "SATXN_BR_CODE"]]


def trans_count(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    grouped = data.groupby(["ACC_RANDOM", "SATXN_ENTRY_DATE"]).size().reset_index(name="count")
    result = grouped.groupby("ACC_RANDOM").agg({"count": ["mean", lambda x: x.std(ddof=0), "sum"]}).reset_index()
    result.columns = ["ACC_RANDOM", f"{prefix}mean_count", f"{prefix}var_count", f"{prefix}sum_count"]
    return result


def trans_txn_amt(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    grouped = (
        data.groupby(["ACC_RANDOM", "SATXN_ENTRY_DATE"])["SATXN_TXN_AMT"]
        .sum()
        .reset_index(name="trans_txn_amt")
    )
    result = (
        grouped.groupby("ACC_RANDOM")
        .agg({"trans_txn_amt": ["mean", lambda x: x.std(ddof=0), "sum"]})
        .reset_index()
    )
    result.columns = ["ACC_RANDOM", f"{prefix}mean_txn_amt", f"{prefix}std_txn_amt", f"{prefix}sum_txn_amt"]
    return result


def trans_sam_bal(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    grouped = (
        data.groupby(["ACC_RANDOM", "SATXN_ENTRY_DATE"])["SATXN_SAM_BAL"]
        .sum()
        .reset_index(name="trans_sam_bal")
    )
    result = (
        grouped.groupby("ACC_RANDOM")
        .agg({"trans_sam_bal": ["mean", lambda x: x.std(ddof=0)]})
        .reset_index()
    )
    result.columns = ["ACC_RANDOM", f"{prefix}mean_trans_sam_bal", f"{prefix}std_trans_sam_bal"]
    return result


def trans_sam_bal_ratio(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    output = data.copy()
    output["ratio"] = output["SATXN_TXN_AMT"] / (output["SATXN_SAM_BAL"] + np.finfo(float).eps)
    result = (
        output.groupby("ACC_RANDOM")
        .agg({"ratio": ["mean", lambda x: x.std(ddof=0)]})
        .reset_index()
    )
    result.columns = ["ACC_RANDOM", f"{prefix}mean_trans_sam_bal_ratio", f"{prefix}std_trans_sam_bal_ratio"]
    return result


def trans_time_diff(data: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    output = data.copy()
    output["SATXN_ENTRY_DATE"] = output["SATXN_ENTRY_DATE"].astype(str).str.strip()
    output["SATXN_ENTRY_DATE"] = pd.to_datetime(output["SATXN_ENTRY_DATE"], format="%Y/%m/%d")
    output["standard_time"] = pd.to_datetime(output["standard_time"], format="%H:%M:%S").dt.time
    output["full_datetime"] = pd.to_datetime(
        output["SATXN_ENTRY_DATE"].astype(str) + " " + output["standard_time"].astype(str)
    )
    output = output.sort_values(["ACC_RANDOM", "full_datetime"])
    output["time_diff"] = output.groupby("ACC_RANDOM")["full_datetime"].diff()
    output["time_diff_second"] = output["time_diff"].dt.total_seconds()

    result = (
        output.groupby("ACC_RANDOM")
        .agg({"time_diff_second": ["mean", lambda x: x.std(ddof=0)]})
        .reset_index()
    )
    result.columns = ["ACC_RANDOM", f"{prefix}mean_time_diff", f"{prefix}std_time_diff"]
    return result


def filter_trans_stat(data: pd.DataFrame) -> pd.DataFrame:
    output = data[~data["SATXN_TXN_STAT"].isin([1, 9])]
    output = output[["ACC_RANDOM"]].drop_duplicates(subset="ACC_RANDOM", keep="first")
    return output


def prepare_transaction_data(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    return prepare_transaction_data_frame(data)


def prepare_transaction_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ACC_RANDOM"] = data["ACC_RANDOM"].astype(str).str.replace(",", "", regex=False)
    data["ID_RANDOM"] = data["ID_RANDOM"].astype(str).str.replace(",", "", regex=False)
    data["SATXN_TXN_AMT"] = data["SATXN_TXN_AMT"].astype(str).str.replace(",", "", regex=False).astype(float)
    data["standard_time"] = data["SATXN_TXN_TIME"].apply(standard_txn_time)
    data["SATXN_SAM_BAL"] = data["SATXN_SAM_BAL"].astype(str).str.replace(",", "", regex=False).astype(float)
    return data


def build_transaction_features_from_df(data: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    data = prepare_transaction_data_frame(data)
    base = data[["ACC_RANDOM", "ID_RANDOM"]].drop_duplicates(subset=["ACC_RANDOM"])

    output = base
    output = check_merge(output, feature_ori(data))

    all_count = trans_count(data)
    output = check_merge(output, all_count)

    count_drcr1 = trans_count(data[data["SATXN_DB_CR_STAT"] == 1], "drcr1_")
    count_drcr2 = trans_count(data[data["SATXN_DB_CR_STAT"] == 2], "drcr2_")
    output = check_merge(output, count_drcr1)
    output = check_merge(output, count_drcr2)

    ratio_features = base.copy()
    ratio_features = ratio_features.merge(count_drcr1[["ACC_RANDOM", "drcr1_sum_count"]], on="ACC_RANDOM", how="left")
    ratio_features = ratio_features.merge(count_drcr2[["ACC_RANDOM", "drcr2_sum_count"]], on="ACC_RANDOM", how="left")
    ratio_features["count_drcr_ratio"] = ratio_features["drcr1_sum_count"] / (
        ratio_features["drcr2_sum_count"] + np.finfo(float).eps
    )
    output = check_merge(output, ratio_features[["ACC_RANDOM", "count_drcr_ratio"]])

    txn_amt_all = trans_txn_amt(data)
    txn_amt_drcr1 = trans_txn_amt(data[data["SATXN_DB_CR_STAT"] == 1], "drcr1_")
    txn_amt_drcr2 = trans_txn_amt(data[data["SATXN_DB_CR_STAT"] == 2], "drcr2_")
    output = check_merge(output, txn_amt_all)
    output = check_merge(output, txn_amt_drcr1)
    output = check_merge(output, txn_amt_drcr2)

    amount_ratio = base.copy()
    amount_ratio = amount_ratio.merge(
        txn_amt_drcr1[["ACC_RANDOM", "drcr1_sum_txn_amt"]], on="ACC_RANDOM", how="left"
    )
    amount_ratio = amount_ratio.merge(
        txn_amt_drcr2[["ACC_RANDOM", "drcr2_sum_txn_amt"]], on="ACC_RANDOM", how="left"
    )
    amount_ratio["sum_txn_amt_ratio"] = amount_ratio["drcr1_sum_txn_amt"] / (
        amount_ratio["drcr2_sum_txn_amt"] + np.finfo(float).eps
    )
    amount_ratio["sum_txn_amt_diff"] = amount_ratio["drcr1_sum_txn_amt"] - amount_ratio["drcr2_sum_txn_amt"]
    output = check_merge(output, amount_ratio[["ACC_RANDOM", "sum_txn_amt_ratio", "sum_txn_amt_diff"]])

    output = check_merge(output, trans_sam_bal(data))
    output = check_merge(output, trans_sam_bal(data[data["SATXN_DB_CR_STAT"] == 1], "drcr1_"))
    output = check_merge(output, trans_sam_bal(data[data["SATXN_DB_CR_STAT"] == 2], "drcr2_"))
    output = check_merge(output, trans_sam_bal_ratio(data))
    output = check_merge(output, trans_time_diff(data))
    output = check_merge(output, trans_time_diff(data[data["SATXN_DB_CR_STAT"] == 1], "drcr1_"))
    output = check_merge(output, trans_time_diff(data[data["SATXN_DB_CR_STAT"] == 2], "drcr2_"))

    filtered_accounts = filter_trans_stat(data)
    output = output.merge(filtered_accounts, how="inner", on="ACC_RANDOM")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_transaction_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    data = prepare_transaction_data(input_path)
    return build_transaction_features_from_df(data, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Taiwan deposit transaction features.")
    parser.add_argument("--input", type=Path, required=True, help="Input SATXNREC csv path.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature csv path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_transaction_features(args.input, args.output)
