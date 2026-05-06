"""Compatibility wrapper for the relocated foreign pipeline assembly code."""

from pathlib import Path

from foreign_pipeline.assemble import (
    assemble_training_table_from_frames as build_foreign_training_table_from_frames,
    drop_overlapping_normal_accounts,
    merge_cfaccount,
)
from foreign_pipeline.config import raw_paths
from foreign_pipeline.features import build_feature_tables
from foreign_pipeline.io import ensure_inputs_exist, load_csv_tables
from foreign_pipeline.prepare import (
    prepare_account_key_table,
    prepare_customer_key_table,
    prepare_feature_tables,
    prepare_foreign_transaction_feature,
)


def build_foreign_training_table(
    feature_root: Path,
    data_root: Path,
    output_path: Path,
):
    paths = raw_paths(data_root)
    ensure_inputs_exist(paths)
    raw_frames = load_csv_tables(paths, low_memory=False)
    prepared_frames = prepare_feature_tables(raw_frames)
    generated_feature_tables = build_feature_tables(prepared_frames, feature_root=feature_root)
    return build_foreign_training_table_from_frames(
        foreign_trans_0=generated_feature_tables["transaction_normal"],
        foreign_trans_1=generated_feature_tables["transaction_sar"],
        foreign_acc_0=generated_feature_tables["account_normal"],
        foreign_acc_1=generated_feature_tables["account_sar"],
        acct_all_0=prepared_frames["cfaccount_normal"],
        acct_all_1=prepared_frames["cfaccount_sar"],
        cus_all_0=prepared_frames["cfmaster_normal"],
        cus_all_1=prepared_frames["cfmaster_sar"],
        output_path=output_path,
    )

__all__ = [
    "build_foreign_training_table",
    "build_foreign_training_table_from_frames",
    "drop_overlapping_normal_accounts",
    "merge_cfaccount",
    "prepare_account_key_table",
    "prepare_customer_key_table",
    "prepare_foreign_transaction_feature",
]
