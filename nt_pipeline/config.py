from pathlib import Path


def nt_raw_paths(data_root: Path) -> dict[str, Path]:
    return {
        "samaster_normal": data_root / "normal/SAMASTER.csv",
        "samaster_sar": data_root / "abnormal/SAMASTER_SAR.csv",
        "satxnrec_normal": data_root / "normal/SATXNREC.csv",
        "satxnrec_sar": data_root / "abnormal/SATXNREC_SAR.csv",
        "cfmaster_normal": data_root / "normal/CFMASTER.csv",
        "cfaccount_normal": data_root / "normal/CFACCOUNT.csv",
        "cfmaster_sar": data_root / "abnormal/CFMASTER_SAR.csv",
        "cfaccount_sar": data_root / "abnormal/CFACCOUNT_SAR.csv",
    }


def foreign_raw_paths(data_root: Path) -> dict[str, Path]:
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


def foreign_feature_paths(feature_root: Path, data_root: Path) -> dict[str, Path]:
    return {
        "fstxn_normal": feature_root / "fstxn_features.csv",
        "fstxn_sar": feature_root / "fstxn_sar_features.csv",
        "fscst_normal": feature_root / "fscst_features.csv",
        "fscst_sar": feature_root / "fscst_sar_features.csv",
        "cfmaster_normal": data_root / "normal/CFMASTER.csv",
        "cfaccount_normal": data_root / "normal/CFACCOUNT.csv",
        "cfmaster_sar": data_root / "abnormal/CFMASTER_SAR.csv",
        "cfaccount_sar": data_root / "abnormal/CFACCOUNT_SAR.csv",
    }
