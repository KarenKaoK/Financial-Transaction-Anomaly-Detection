import pandas as pd

from foreign_pipeline.account_features import prepare_account_data
from foreign_pipeline.transaction_features import prepare_transaction_data_frame


def prepare_foreign_transaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_transaction_data_frame(df)


def prepare_account_key_table(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_account_data(df)


def prepare_customer_key_table(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["ID_RANDOM"] = output["ID_RANDOM"].astype(str).str.replace(",", "", regex=False)
    return output


def prepare_feature_tables(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
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
