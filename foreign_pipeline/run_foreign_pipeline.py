import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foreign_pipeline.assemble import assemble_training_table
from foreign_pipeline.config import raw_paths
from foreign_pipeline.features import build_feature_tables
from foreign_pipeline.io import ensure_inputs_exist, load_csv_tables
from foreign_pipeline.model import save_foreign_model_artifacts, train_foreign_model_from_path
from foreign_pipeline.prepare import prepare_feature_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run foreign-currency end-to-end pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/foreign_pipeline"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = raw_paths(args.data_root)
    ensure_inputs_exist(paths)

    raw_frames = load_csv_tables(paths, low_memory=False)
    prepared_frames = prepare_feature_tables(raw_frames)
    feature_root = args.artifact_root / "features"
    feature_tables = build_feature_tables(prepared_frames, feature_root=feature_root)
    training_path = args.artifact_root / "features" / "feature_for_cus_acc.csv"
    assemble_training_table(feature_tables, prepared_frames, feature_root=training_path.parent)

    bundle, metrics = train_foreign_model_from_path(training_path)
    save_foreign_model_artifacts(bundle, metrics, args.artifact_root)
