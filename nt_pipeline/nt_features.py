from pathlib import Path
from typing import Optional

import pandas as pd

from nt_pipeline.account_features import build_account_features_from_df
from nt_pipeline.transaction_features import build_transaction_features_from_df


def build_nt_feature_tables(
    prepared_frames: dict[str, pd.DataFrame],
    feature_root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    feature_tables = {
        "account_normal": build_account_features_from_df(
            prepared_frames["samaster_normal"],
            feature_root / "feature_NT_account.csv" if feature_root is not None else None,
        ),
        "account_sar": build_account_features_from_df(
            prepared_frames["samaster_sar"],
            feature_root / "feature_NT_account_SAR.csv" if feature_root is not None else None,
        ),
        "transaction_normal": build_transaction_features_from_df(
            prepared_frames["satxnrec_normal"],
            feature_root / "feature_NT_transaction.csv" if feature_root is not None else None,
        ),
        "transaction_sar": build_transaction_features_from_df(
            prepared_frames["satxnrec_sar"],
            feature_root / "feature_NT_transaction_SAR.csv" if feature_root is not None else None,
        ),
    }
    return feature_tables
