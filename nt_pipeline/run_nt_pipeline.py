import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nt_pipeline.config import nt_raw_paths
from nt_pipeline.io import ensure_inputs_exist
from nt_pipeline.io import load_nt_raw_tables
from nt_pipeline.nt_assemble import assemble_nt_training_table
from nt_pipeline.nt_features import build_nt_feature_tables
from nt_pipeline.nt_model import save_nt_model_artifacts, train_nt_model_from_path
from nt_pipeline.prepare import prepare_nt_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Taiwan deposit end-to-end pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--feature-root", type=Path, default=Path("Data/特徵資料"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/nt_pipeline"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = args.data_root
    feature_root = args.feature_root
    artifact_root = args.artifact_root
    ensure_inputs_exist(nt_raw_paths(data_root))

    raw_tables = load_nt_raw_tables(data_root)
    prepared_tables = prepare_nt_tables(raw_tables)
    feature_tables = build_nt_feature_tables(prepared_tables, feature_root=feature_root)
    assemble_nt_training_table(feature_tables, prepared_tables, feature_root=feature_root)

    bundle, metrics = train_nt_model_from_path(feature_root / "feature_NT_cus_acc.csv")
    save_nt_model_artifacts(bundle, metrics, artifact_root)
