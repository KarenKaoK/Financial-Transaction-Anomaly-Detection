from pathlib import Path
from typing import Optional


def raw_paths(data_root: Path) -> dict[str, Path]:
    return {
        "fstxn_normal": data_root / "normal/FSTXN.csv",
        "fstxn_sar": data_root / "abnormal/FSTXN_SAR.csv",
        "fscst_normal": data_root / "normal/FSCST.csv",
        "fscst_sar": data_root / "abnormal/FSCST_SAR.csv",
        "cfmaster_normal": data_root / "normal/CFMASTER.csv",
        "cfaccount_normal": data_root / "normal/CFACCOUNT.csv",
        "cfmaster_sar": data_root / "abnormal/CFMASTER_SAR.csv",
        "cfaccount_sar": data_root / "abnormal/CFACCOUNT_SAR.csv",
    }
