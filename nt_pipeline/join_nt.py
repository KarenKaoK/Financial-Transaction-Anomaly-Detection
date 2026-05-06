import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_id_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = output[column].astype(str).str.replace(",", "", regex=False)
    return output


def build_nt_all_from_frames(
    account_normal: pd.DataFrame,
    transaction_normal: pd.DataFrame,
    account_sar: pd.DataFrame,
    transaction_sar: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    account_normal = normalize_id_columns(account_normal, ["ACC_RANDOM", "ID_RANDOM"])
    transaction_normal = normalize_id_columns(transaction_normal, ["ACC_RANDOM", "ID_RANDOM"])
    account_sar = normalize_id_columns(account_sar, ["ACC_RANDOM", "ID_RANDOM"])
    transaction_sar = normalize_id_columns(transaction_sar, ["ACC_RANDOM", "ID_RANDOM"])

    nt_normal = account_normal.merge(transaction_normal, how="left", on=["ACC_RANDOM", "ID_RANDOM"])
    nt_sar = account_sar.merge(transaction_sar, how="left", on=["ACC_RANDOM", "ID_RANDOM"])
    nt_normal["sar"] = 0
    nt_sar["sar"] = 1

    output = pd.concat([nt_normal, nt_sar], axis=0).reset_index(drop=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_nt_all(
    account_normal_path: Path,
    transaction_normal_path: Path,
    account_sar_path: Path,
    transaction_sar_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    return build_nt_all_from_frames(
        account_normal=pd.read_csv(account_normal_path),
        transaction_normal=pd.read_csv(transaction_normal_path),
        account_sar=pd.read_csv(account_sar_path),
        transaction_sar=pd.read_csv(transaction_sar_path),
        output_path=output_path,
    )


def build_nt_cus_acc_from_frames(
    nt_all: pd.DataFrame,
    customer_normal: pd.DataFrame,
    account_normal: pd.DataFrame,
    customer_sar: pd.DataFrame,
    account_sar: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    nt_all_normal = nt_all[nt_all["sar"] == 0]
    nt_all_sar = nt_all[nt_all["sar"] == 1]

    nt_cus_normal = nt_all_normal.merge(customer_normal, on="ID_RANDOM", how="left")
    nt_cus_acc_normal = nt_cus_normal.merge(account_normal, on=["ID_RANDOM", "ACC_RANDOM"], how="left")

    nt_cus_sar = nt_all_sar.merge(customer_sar, on="ID_RANDOM", how="left")
    nt_cus_acc_sar = nt_cus_sar.merge(account_sar, on=["ID_RANDOM", "ACC_RANDOM"], how="left")

    output = pd.concat([nt_cus_acc_normal, nt_cus_acc_sar], ignore_index=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return output


def build_nt_cus_acc(
    nt_all_path: Path,
    customer_normal_path: Path,
    account_normal_path: Path,
    customer_sar_path: Path,
    account_sar_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    return build_nt_cus_acc_from_frames(
        nt_all=pd.read_csv(nt_all_path),
        customer_normal=pd.read_csv(customer_normal_path),
        account_normal=pd.read_csv(account_normal_path),
        customer_sar=pd.read_csv(customer_sar_path),
        account_sar=pd.read_csv(account_sar_path),
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Taiwan deposit feature tables.")
    parser.add_argument("--account-normal", type=Path, required=True)
    parser.add_argument("--transaction-normal", type=Path, required=True)
    parser.add_argument("--account-sar", type=Path, required=True)
    parser.add_argument("--transaction-sar", type=Path, required=True)
    parser.add_argument("--nt-all-output", type=Path, required=True)
    parser.add_argument("--customer-normal", type=Path, required=True)
    parser.add_argument("--cfaccount-normal", type=Path, required=True)
    parser.add_argument("--customer-sar", type=Path, required=True)
    parser.add_argument("--cfaccount-sar", type=Path, required=True)
    parser.add_argument("--nt-cus-acc-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_nt_all(
        account_normal_path=args.account_normal,
        transaction_normal_path=args.transaction_normal,
        account_sar_path=args.account_sar,
        transaction_sar_path=args.transaction_sar,
        output_path=args.nt_all_output,
    )
    build_nt_cus_acc(
        nt_all_path=args.nt_all_output,
        customer_normal_path=args.customer_normal,
        account_normal_path=args.cfaccount_normal,
        customer_sar_path=args.customer_sar,
        account_sar_path=args.cfaccount_sar,
        output_path=args.nt_cus_acc_output,
    )
