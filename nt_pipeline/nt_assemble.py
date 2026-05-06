from pathlib import Path
from typing import Optional

import pandas as pd

from nt_pipeline.join_nt import build_nt_all_from_frames, build_nt_cus_acc_from_frames


def summarize_nt_overlap(nt_all: pd.DataFrame) -> dict[str, int]:
    normal = nt_all[nt_all["sar"] == 0]
    sar = nt_all[nt_all["sar"] == 1]

    normal_acc = set(normal["ACC_RANDOM"].astype(str))
    sar_acc = set(sar["ACC_RANDOM"].astype(str))
    normal_id = set(normal["ID_RANDOM"].astype(str))
    sar_id = set(sar["ID_RANDOM"].astype(str))

    return {
        "overlap_acc_random": len(normal_acc & sar_acc),
        "overlap_id_random": len(normal_id & sar_id),
    }


def remove_overlap_from_normal(nt_all: pd.DataFrame) -> pd.DataFrame:
    normal = nt_all[nt_all["sar"] == 0].copy()
    sar = nt_all[nt_all["sar"] == 1].copy()

    overlap_acc = set(normal["ACC_RANDOM"].astype(str)) & set(sar["ACC_RANDOM"].astype(str))
    overlap_id = set(normal["ID_RANDOM"].astype(str)) & set(sar["ID_RANDOM"].astype(str))

    filtered_normal = normal[
        ~normal["ACC_RANDOM"].astype(str).isin(overlap_acc) & ~normal["ID_RANDOM"].astype(str).isin(overlap_id)
    ].copy()
    return pd.concat([filtered_normal, sar], ignore_index=True)


def assemble_nt_base_table(
    feature_tables: dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    drop_overlap_from_normal: bool = False,
) -> pd.DataFrame:
    nt_all = build_nt_all_from_frames(
        account_normal=feature_tables["account_normal"],
        transaction_normal=feature_tables["transaction_normal"],
        account_sar=feature_tables["account_sar"],
        transaction_sar=feature_tables["transaction_sar"],
        output_path=None,
    )

    if drop_overlap_from_normal:
        nt_all = remove_overlap_from_normal(nt_all)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nt_all.to_csv(output_path, index=False)
    return nt_all


def assemble_nt_training_table(
    feature_tables: dict[str, pd.DataFrame],
    prepared_frames: dict[str, pd.DataFrame],
    feature_root: Optional[Path] = None,
    drop_overlap_from_normal: bool = False,
) -> pd.DataFrame:
    nt_all = assemble_nt_base_table(
        feature_tables,
        output_path=feature_root / "feature_NT_all.csv" if feature_root is not None else None,
        drop_overlap_from_normal=drop_overlap_from_normal,
    )

    return build_nt_cus_acc_from_frames(
        nt_all=nt_all,
        customer_normal=prepared_frames["cfmaster_normal"],
        account_normal=prepared_frames["cfaccount_normal"],
        customer_sar=prepared_frames["cfmaster_sar"],
        account_sar=prepared_frames["cfaccount_sar"],
        output_path=feature_root / "feature_NT_cus_acc.csv" if feature_root is not None else None,
    )


def assemble_nt_scoring_table(
    feature_tables: dict[str, pd.DataFrame],
    prepared_frames: dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    nt_normal = feature_tables["account_normal"].merge(
        feature_tables["transaction_normal"], how="left", on=["ACC_RANDOM", "ID_RANDOM"]
    )
    scoring_table = nt_normal.merge(prepared_frames["cfmaster_normal"], on="ID_RANDOM", how="left")
    scoring_table = scoring_table.merge(
        prepared_frames["cfaccount_normal"], on=["ID_RANDOM", "ACC_RANDOM"], how="left"
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scoring_table.to_csv(output_path, index=False)
    return scoring_table
