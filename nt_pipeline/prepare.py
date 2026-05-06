import pandas as pd

from nt_pipeline.account_features import prepare_account_data
from nt_pipeline.foreign_join import (
    prepare_account_key_table,
    prepare_customer_key_table,
    prepare_foreign_transaction_feature,
)
from nt_pipeline.join_nt import normalize_id_columns
from nt_pipeline.transaction_features import prepare_transaction_data_frame


def prepare_nt_tables(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "samaster_normal": prepare_account_data(raw_frames["samaster_normal"]),
        "samaster_sar": prepare_account_data(raw_frames["samaster_sar"]),
        "satxnrec_normal": prepare_transaction_data_frame(raw_frames["satxnrec_normal"]),
        "satxnrec_sar": prepare_transaction_data_frame(raw_frames["satxnrec_sar"]),
        "cfmaster_normal": normalize_id_columns(raw_frames["cfmaster_normal"], ["ID_RANDOM"]),
        "cfaccount_normal": normalize_id_columns(raw_frames["cfaccount_normal"], ["ACC_RANDOM", "ID_RANDOM"]),
        "cfmaster_sar": normalize_id_columns(raw_frames["cfmaster_sar"], ["ID_RANDOM"]),
        "cfaccount_sar": normalize_id_columns(raw_frames["cfaccount_sar"], ["ACC_RANDOM", "ID_RANDOM"]),
    }


def prepare_foreign_feature_tables(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "fstxn_normal": prepare_foreign_transaction_feature(raw_frames["fstxn_normal"]),
        "fstxn_sar": prepare_foreign_transaction_feature(raw_frames["fstxn_sar"]),
        "fscst_normal": prepare_account_key_table(raw_frames["fscst_normal"]),
        "fscst_sar": prepare_account_key_table(raw_frames["fscst_sar"]),
        "cfmaster_normal": prepare_customer_key_table(raw_frames["cfmaster_normal"]),
        "cfaccount_normal": prepare_account_key_table(raw_frames["cfaccount_normal"]),
        "cfmaster_sar": prepare_customer_key_table(raw_frames["cfmaster_sar"]),
        "cfaccount_sar": prepare_account_key_table(raw_frames["cfaccount_sar"]),
    }


def prepare_foreign_raw_tables(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared = prepare_foreign_feature_tables(raw_frames)
    for name in ("fstxn_normal", "fstxn_sar"):
        if "ID_RANDOM" in raw_frames[name].columns:
            prepared[name]["ID_RANDOM"] = raw_frames[name]["ID_RANDOM"].astype(str).str.replace(",", "", regex=False)
    return prepared
