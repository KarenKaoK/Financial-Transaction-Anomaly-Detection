import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def check_merge(base: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    common_columns = set(base.columns) & set(feature.columns)
    allowed_overlap = {"ACC_RANDOM", "ID_RANDOM"}
    unexpected_overlap = common_columns - allowed_overlap
    if unexpected_overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(unexpected_overlap)}")
    return base.merge(feature, on="ACC_RANDOM", how="left")


def feature_ori(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SAMST_BR_CODE",
        "SAMST_WASH_NOCHK_MARK",
        "SAMST_FRUIT",
        "SAMST_OPEN_KIND",
        "SAMST_INT_CODE",
        "SAMST_NET_PASSWD_ERR_TIMES",
        "SAMST_NOTICE_KIND",
        "SAMST_CHK_SHEET_CODE",
        "SAMST_ACCOUNT_PURPOSE",
    ]
    dedup_columns = ["ACC_RANDOM"] + [col for col in columns if col != "SAMST_BR_CODE"]
    output = data.drop_duplicates(subset=dedup_columns)
    return output[["ACC_RANDOM"] + columns]


def feature_ever_one(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "SAMST_CUST_STAT",
        "SAMST_PASSBOOK_STATUS",
        "SAMST_CRNT_SAVING_STK_FLAG",
        "SAMST_OD_STAT",
        "SAMST_PB_LOSE_STAT",
        "SAMST_CHOP_LOSE_STAT",
        "SAMST_VISA_DEBIT_FLAG",
        "SAMST_TXN_STOP_STAT",
        "SAMST_DEAD_STAT",
        "SAMST_COURT_STAT",
        "SAMST_HOLD_STAT",
        "SAMST_WARN_STAT",
        "SAMST_OD_OV_STAT",
        "SAMST_WASH_DB_OV_FLAG",
        "SAMST_WASH_CR_OV_FLAG",
        "SAMST_TD_CASH_CR_FLAG",
    ]
    output = data[["ACC_RANDOM"]].copy()
    new_columns = []
    for column in columns:
        new_column = f"{column}_ever_one"
        output[new_column] = data.groupby("ACC_RANDOM")[column].transform(
            lambda x: 1 if any(val >= 1 for val in x.values if not isinstance(val, str)) else 0
        )
        new_columns.append(new_column)
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + new_columns)
    return output[["ACC_RANDOM"] + new_columns]


def feature_count_one(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["SAMST_TXN_STOP_STAT", "SAMST_WASH_DB_OV_FLAG", "SAMST_WASH_CR_OV_FLAG"]
    output = data[["ACC_RANDOM"]].copy()
    new_columns = []
    for column in columns:
        new_column = f"count_geq_one_{column}"
        count_geq_one = data.groupby("ACC_RANDOM")[column].transform("sum")
        output[new_column] = (count_geq_one >= 1).astype(int)

        grouped_sum = output.groupby("ACC_RANDOM")[new_column].sum().reset_index()
        grouped_sum.columns = ["ACC_RANDOM", f"sum_{new_column}"]
        output = output.merge(grouped_sum, on="ACC_RANDOM", how="left")
        new_columns.append(f"sum_{new_column}")
        output = output.drop_duplicates(subset=["ACC_RANDOM"] + new_columns)
    return output[["ACC_RANDOM"] + new_columns]


def feature_category_company(data: pd.DataFrame) -> pd.DataFrame:
    output = data[["ACC_RANDOM", "SAMST_ACC_CHAR_CODE"]].copy()
    output["SAMST_ACC_CHAR_CODE"] = 0
    condition = output["SAMST_ACC_CHAR_CODE"].between(71, 79)
    output.loc[condition, "SAMST_ACC_CHAR_CODE"] = 1
    output = output.drop_duplicates(subset=["ACC_RANDOM", "SAMST_ACC_CHAR_CODE"])
    return output[["ACC_RANDOM", "SAMST_ACC_CHAR_CODE"]]


def feature_statistics(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["bal_L_DAY_diff"] = output["SAMST_BAL"] - output["SAMST_L_DAY_BAL"]
    output["bal_pb_diff"] = output["SAMST_BAL"] - output["SAMST_PB_BAL"]

    columns = [
        "SAMST_BAL",
        "SAMST_PB_BAL",
        "SAMST_TD_ATM_TRAN_AMT",
        "SAMST_TM_ATM_TRAN_AMT",
        "SAMST_TD_ATM_WDRAW_AMT",
        "SAMST_TD_NON_CON_TFR_AMT",
        "SAMST_TD_CASH_CR_AMT",
        "SAMST_TD_CASH_DB_AMT",
        "SAMST_TD_RT_TXN_CNT",
        "SAMST_TD_RT_TXN_AMT",
        "SAMST_TD_AGBR_DB_AMT",
        "bal_L_DAY_diff",
        "bal_pb_diff",
    ]
    stats = ["mean", "std", "max", "min"]

    for column in columns:
        for stat in stats:
            output[f"{column}_{stat}"] = output.groupby("ACC_RANDOM")[column].transform(stat)

    result_columns = [f"{column}_{stat}" for column in columns for stat in stats]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + result_columns)
    return output[["ACC_RANDOM"] + result_columns]


def feature_ratio_to_bal(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    ratio_columns = [
        "SAMST_TD_ATM_TRAN_AMT",
        "SAMST_TM_ATM_TRAN_AMT",
        "SAMST_TD_ATM_WDRAW_AMT",
        "SAMST_TD_NON_CON_TFR_AMT",
        "SAMST_TD_CASH_CR_AMT",
        "SAMST_TD_CASH_DB_AMT",
        "SAMST_TD_RT_TXN_AMT",
        "SAMST_TD_AGBR_DB_AMT",
    ]
    balance_columns = ["SAMST_BAL", "SAMST_Y_MAX_BAL"]
    for balance_column in balance_columns:
        output[balance_column] = output[balance_column].replace(0, 1e-10)

    derived_columns = []
    for column in ratio_columns:
        for balance_column in balance_columns:
            new_column = f"ratio_{column}_to_{balance_column}"
            output[new_column] = output[column] / output[balance_column]
            derived_columns.append(new_column)

    stats = ["mean", "std", "max", "min"]
    for column in derived_columns:
        for stat in stats:
            output[f"{column}_{stat}"] = output.groupby("ACC_RANDOM")[column].transform(stat)

    result_columns = [f"{column}_{stat}" for column in derived_columns for stat in stats]
    output = output.drop_duplicates(subset=["ACC_RANDOM"] + result_columns)
    return output[["ACC_RANDOM"] + result_columns]


def prepare_account_data(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for column in ["ACC_RANDOM", "ID_RANDOM"]:
        if column in output.columns:
            output[column] = output[column].astype(str).str.replace(",", "", regex=False)
    return output


def build_account_features_from_df(data: pd.DataFrame, output_path: Optional[Path] = None) -> pd.DataFrame:
    data = prepare_account_data(data)
    base = data[["ACC_RANDOM", "ID_RANDOM"]].drop_duplicates(subset=["ACC_RANDOM"])

    feature_frames = [
        feature_ori(data),
        feature_ever_one(data),
        feature_count_one(data),
        feature_category_company(data),
        feature_statistics(data),
        feature_ratio_to_bal(data),
    ]

    output = base
    for feature_frame in feature_frames:
        output = check_merge(output, feature_frame)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_account_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    return build_account_features_from_df(data, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Taiwan deposit account features.")
    parser.add_argument("--input", type=Path, required=True, help="Input SAMASTER csv path.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature csv path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_account_features(args.input, args.output)
