from pathlib import Path
from typing import Optional

import pandas as pd


def merge_cfaccount(base: pd.DataFrame, cfaccount: pd.DataFrame) -> pd.DataFrame:
    output = base.merge(cfaccount, on="ACC_RANDOM", how="left", suffixes=("", "_cfacc"))
    if "ID_RANDOM_cfacc" in output.columns:
        if "ID_RANDOM" in output.columns:
            output["ID_RANDOM"] = output["ID_RANDOM"].fillna(output["ID_RANDOM_cfacc"])
        else:
            output = output.rename(columns={"ID_RANDOM_cfacc": "ID_RANDOM"})
        output = output.drop(columns=["ID_RANDOM_cfacc"])
    return output


def drop_overlapping_normal_accounts(
    normal_features: pd.DataFrame,
    sar_features: pd.DataFrame,
) -> pd.DataFrame:
    combined_acc_random = pd.concat([normal_features["ACC_RANDOM"], sar_features["ACC_RANDOM"]])
    duplicated_values = combined_acc_random[combined_acc_random.duplicated(keep=False)]
    return normal_features[~normal_features["ACC_RANDOM"].isin(duplicated_values)].copy()


def assemble_training_table_from_frames(
    foreign_trans_0: pd.DataFrame,
    foreign_trans_1: pd.DataFrame,
    foreign_acc_0: pd.DataFrame,
    foreign_acc_1: pd.DataFrame,
    acct_all_0: pd.DataFrame,
    acct_all_1: pd.DataFrame,
    cus_all_0: pd.DataFrame,
    cus_all_1: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    foreign_trans_0 = drop_overlapping_normal_accounts(foreign_trans_0, foreign_trans_1)

    foreign_0 = foreign_trans_0.merge(foreign_acc_0, on="ACC_RANDOM", how="left")
    foreign_1 = foreign_trans_1.merge(foreign_acc_1, on="ACC_RANDOM", how="left")

    foreign_0 = merge_cfaccount(foreign_0, acct_all_0)
    foreign_1 = merge_cfaccount(foreign_1, acct_all_1)

    foreign_0 = foreign_0.merge(cus_all_0, on="ID_RANDOM", how="left")
    foreign_1 = foreign_1.merge(cus_all_1, on="ID_RANDOM", how="left")

    foreign_0["sar"] = 0
    foreign_1["sar"] = 1

    output = pd.concat([foreign_0, foreign_1], ignore_index=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def assemble_training_table(
    feature_tables: dict[str, pd.DataFrame],
    prepared_frames: dict[str, pd.DataFrame],
    feature_root: Optional[Path] = None,
) -> pd.DataFrame:
    return assemble_training_table_from_frames(
        foreign_trans_0=feature_tables["transaction_normal"],
        foreign_trans_1=feature_tables["transaction_sar"],
        foreign_acc_0=feature_tables["account_normal"],
        foreign_acc_1=feature_tables["account_sar"],
        acct_all_0=prepared_frames["cfaccount_normal"],
        acct_all_1=prepared_frames["cfaccount_sar"],
        cus_all_0=prepared_frames["cfmaster_normal"],
        cus_all_1=prepared_frames["cfmaster_sar"],
        output_path=feature_root / "feature_for_cus_acc.csv" if feature_root is not None else None,
    )
