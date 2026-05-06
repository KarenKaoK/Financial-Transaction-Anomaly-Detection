import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


COMMON_CURRENCIES = ["USD", "JPY", "AUD", "CNY", "ZAR", "HKD", "EUR", "GBP", "NZD", "SGD"]


def to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_account_data(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in ["ACC_RANDOM", "ID_RANDOM"]:
        if column in output.columns:
            output[column] = output[column].astype(str).str.replace(",", "", regex=False)

    numeric_columns = [
        "FSCST_BR_CODE",
        "FSCST_CUST_STAT",
        "FSCST_MS_STAT",
        "FSCST_NOTICE_KIND",
        "FSCST_CHK_SHEET_CODE",
        "FSCST_PB_LOSE_STAT",
        "FSCST_CHOP_LOSE_STAT",
        "FSCST_TXN_STOP_STAT",
        "FSCST_DEAD_STAT",
        "FSCST_COURT_STAT",
        "FSCST_WARN_STAT",
        "FSCST_CUR_CNT",
        "FSCST_TI_CNT",
        "FSCST_TD_TXN_DB_AMT",
        "FSCST_TD_TXN_CR_AMT",
        "FSCST_TD_CASH_DB_AMT",
        "FSCST_TD_CASH_CR_AMT",
        "FSCST_COUNTER_FLG",
    ]
    for column in numeric_columns:
        if column in output.columns:
            output[column] = to_numeric_series(output[column])

    balance_columns = [f"FSCST_CUR_BAL_{index}" for index in range(1, 31)]
    for column in balance_columns:
        if column in output.columns:
            output[column] = to_numeric_series(output[column])
    return output


def feature_profile(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "FSCST_BR_CODE",
        "FSCST_MS_STAT",
        "FSCST_NOTICE_KIND",
        "FSCST_CHK_SHEET_CODE",
        "FSCST_CUR_CNT",
        "FSCST_TI_CNT",
        "FSCST_COUNTER_FLG",
    ]
    output = data.drop_duplicates(subset=["ACC_RANDOM"])
    return output[["ACC_RANDOM"] + [column for column in columns if column in output.columns]]


def feature_flags(data: pd.DataFrame) -> pd.DataFrame:
    flag_columns = [
        "FSCST_CUST_STAT",
        "FSCST_PB_LOSE_STAT",
        "FSCST_CHOP_LOSE_STAT",
        "FSCST_TXN_STOP_STAT",
        "FSCST_DEAD_STAT",
        "FSCST_COURT_STAT",
        "FSCST_WARN_STAT",
    ]
    output = data[["ACC_RANDOM"]].copy()
    derived_columns = []
    for column in flag_columns:
        if column not in data.columns:
            continue
        new_column = f"{column}_ever_flagged"
        output[new_column] = data.groupby("ACC_RANDOM")[column].transform(lambda x: int((x.fillna(0) >= 1).any()))
        derived_columns.append(new_column)

    output = output.drop_duplicates(subset=["ACC_RANDOM"] + derived_columns)
    return output[["ACC_RANDOM"] + derived_columns]


def feature_amount_statistics(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "FSCST_TD_TXN_DB_AMT",
        "FSCST_TD_TXN_CR_AMT",
        "FSCST_TD_CASH_DB_AMT",
        "FSCST_TD_CASH_CR_AMT",
    ]
    output = data.copy()
    output["fscst_net_flow"] = output.get("FSCST_TD_TXN_CR_AMT", 0) - output.get("FSCST_TD_TXN_DB_AMT", 0)
    output["fscst_cash_flow"] = output.get("FSCST_TD_CASH_CR_AMT", 0) - output.get("FSCST_TD_CASH_DB_AMT", 0)
    columns.extend(["fscst_net_flow", "fscst_cash_flow"])

    stats = ["mean", "max", "min", "sum"]
    for column in columns:
        if column not in output.columns:
            continue
        for stat in stats:
            output[f"{column}_{stat}"] = output.groupby("ACC_RANDOM")[column].transform(stat)

    result_columns = [
        f"{column}_{stat}"
        for column in columns
        if column in output.columns
        for stat in stats
    ]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + result_columns)
    return output[["ACC_RANDOM"] + result_columns]


def feature_currency_portfolio(data: pd.DataFrame) -> pd.DataFrame:
    output = data[["ACC_RANDOM"]].copy()
    code_columns = [f"FSCST_CUR_CODE_{index}" for index in range(1, 31)]
    balance_columns = [f"FSCST_CUR_BAL_{index}" for index in range(1, 31)]

    codes = data[code_columns].copy().fillna("").astype(str)
    balances = data[balance_columns].copy().fillna(0.0).astype(float)

    output["portfolio_total_balance"] = balances.sum(axis=1)
    output["portfolio_max_balance"] = balances.max(axis=1)
    output["portfolio_nonzero_currency_count"] = (balances > 0).sum(axis=1)
    output["portfolio_mean_nonzero_balance"] = balances.replace(0, np.nan).mean(axis=1).fillna(0.0)

    for currency in COMMON_CURRENCIES:
        has_currency = ((codes == currency) & (balances > 0).values).any(axis=1).astype(int)
        output[f"has_currency_{currency}"] = has_currency

    derived_columns = [column for column in output.columns if column != "ACC_RANDOM"]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + derived_columns)
    return output


def feature_balance_ratios(data: pd.DataFrame) -> pd.DataFrame:
    output = data[["ACC_RANDOM"]].copy()
    credit = data.get("FSCST_TD_TXN_CR_AMT", pd.Series(0, index=data.index)).fillna(0.0)
    debit = data.get("FSCST_TD_TXN_DB_AMT", pd.Series(0, index=data.index)).fillna(0.0)
    cash_credit = data.get("FSCST_TD_CASH_CR_AMT", pd.Series(0, index=data.index)).fillna(0.0)
    cash_debit = data.get("FSCST_TD_CASH_DB_AMT", pd.Series(0, index=data.index)).fillna(0.0)

    output["txn_credit_debit_ratio"] = credit / (debit + np.finfo(float).eps)
    output["cash_credit_debit_ratio"] = cash_credit / (cash_debit + np.finfo(float).eps)
    output["cash_to_total_txn_ratio"] = (cash_credit + cash_debit) / (credit + debit + np.finfo(float).eps)
    derived_columns = [column for column in output.columns if column != "ACC_RANDOM"]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + derived_columns)
    return output


def check_merge(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    common_columns = set(base.columns) & set(feature.columns)
    allowed_overlap = {"ACC_RANDOM", "ID_RANDOM"}
    unexpected_overlap = common_columns - allowed_overlap
    if unexpected_overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(unexpected_overlap)}")
    return base.merge(feature, on="ACC_RANDOM", how="left")


def build_account_features_from_df(data: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    prepared = prepare_account_data(data)
    base = prepared[["ACC_RANDOM", "ID_RANDOM"]].drop_duplicates(subset=["ACC_RANDOM"])
    feature_frames = [
        feature_profile(prepared),
        feature_flags(prepared),
        feature_amount_statistics(prepared),
        feature_currency_portfolio(prepared),
        feature_balance_ratios(prepared),
    ]

    output = base
    for feature_frame in feature_frames:
        output = check_merge(output, feature_frame)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_account_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path, low_memory=False)
    return build_account_features_from_df(data, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build foreign-currency account features.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_account_features(args.input, args.output)
