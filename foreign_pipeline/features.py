from pathlib import Path
from typing import Optional

import pandas as pd

from foreign_pipeline.account_features import build_account_features_from_df
from foreign_pipeline.transaction_features import build_transaction_features_from_df


def build_feature_tables(
    prepared_frames: dict[str, pd.DataFrame],
    feature_root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    return {
        "account_normal": build_account_features_from_df(
            prepared_frames["fscst_normal"],
            feature_root / "fscst_features.csv" if feature_root is not None else None,
        ),
        "account_sar": build_account_features_from_df(
            prepared_frames["fscst_sar"],
            feature_root / "fscst_sar_features.csv" if feature_root is not None else None,
        ),
        "transaction_normal": build_transaction_features_from_df(
            prepared_frames["fstxn_normal"],
            feature_root / "fstxn_features.csv" if feature_root is not None else None,
        ),
        "transaction_sar": build_transaction_features_from_df(
            prepared_frames["fstxn_sar"],
            feature_root / "fstxn_sar_features.csv" if feature_root is not None else None,
        ),
    }
