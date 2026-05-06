import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from foreign_pipeline.assemble import assemble_training_table as assemble_foreign_training_table
from foreign_pipeline.config import raw_paths as foreign_raw_paths
from foreign_pipeline.features import build_feature_tables as build_foreign_feature_tables
from foreign_pipeline.io import ensure_inputs_exist as ensure_foreign_inputs_exist
from foreign_pipeline.io import load_csv_tables as load_foreign_csv_tables
from foreign_pipeline.model import save_foreign_model_artifacts, train_foreign_model_from_path
from foreign_pipeline.prepare import prepare_feature_tables as prepare_foreign_feature_tables
from nt_pipeline.account_features import build_account_features_from_df
from nt_pipeline.config import nt_raw_paths
from nt_pipeline.eda import summarize_dataframe, write_eda_report
from nt_pipeline.io import ensure_inputs_exist, load_csv_tables
from nt_pipeline.join_nt import build_nt_all, build_nt_cus_acc
from nt_pipeline.prepare import prepare_nt_tables
from nt_pipeline.transaction_features import build_transaction_features_from_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified abnormal transaction detection pipeline: read, EDA, clean, feature, train, and save pickle."
    )
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/main_pipeline"))
    parser.add_argument("--pipeline", choices=["nt", "foreign", "both"], default="both")
    return parser.parse_args()


def log_stage(message: str) -> None:
    logger.info(f"[main] {message}")


def write_stage_report(name: str, frames: dict[str, pd.DataFrame], output_root: Path) -> None:
    log_stage(f"writing EDA report: {name}")
    report = {frame_name: summarize_dataframe(frame_name, frame) for frame_name, frame in frames.items()}
    write_eda_report(report, output_root / "eda" / f"{name}.json")


def save_cleaned_data(cleaned_frames: dict[str, pd.DataFrame], output_root: Path) -> None:
    log_stage("saving cleaned CSV files")
    clean_root = output_root / "cleaned"
    clean_root.mkdir(parents=True, exist_ok=True)
    for name, frame in cleaned_frames.items():
        frame.to_csv(clean_root / f"{name}.csv", index=False)


def build_features(cleaned_frames: dict[str, pd.DataFrame], output_root: Path) -> Path:
    log_stage("building NT feature tables")
    feature_root = output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)

    build_account_features_from_df(cleaned_frames["samaster_normal"], feature_root / "feature_NT_account.csv")
    build_account_features_from_df(cleaned_frames["samaster_sar"], feature_root / "feature_NT_account_SAR.csv")
    build_transaction_features_from_df(cleaned_frames["satxnrec_normal"], feature_root / "feature_NT_transaction.csv")
    build_transaction_features_from_df(cleaned_frames["satxnrec_sar"], feature_root / "feature_NT_transaction_SAR.csv")

    nt_all_path = feature_root / "feature_NT_all.csv"
    nt_cus_acc_path = feature_root / "feature_NT_cus_acc.csv"

    build_nt_all(
        account_normal_path=feature_root / "feature_NT_account.csv",
        transaction_normal_path=feature_root / "feature_NT_transaction.csv",
        account_sar_path=feature_root / "feature_NT_account_SAR.csv",
        transaction_sar_path=feature_root / "feature_NT_transaction_SAR.csv",
        output_path=nt_all_path,
    )

    customer_normal_path = output_root / "cleaned" / "cfmaster_normal.csv"
    account_normal_path = output_root / "cleaned" / "cfaccount_normal.csv"
    customer_sar_path = output_root / "cleaned" / "cfmaster_sar.csv"
    account_sar_path = output_root / "cleaned" / "cfaccount_sar.csv"

    build_nt_cus_acc(
        nt_all_path=nt_all_path,
        customer_normal_path=customer_normal_path,
        account_normal_path=account_normal_path,
        customer_sar_path=customer_sar_path,
        account_sar_path=account_sar_path,
        output_path=nt_cus_acc_path,
    )
    return nt_cus_acc_path


def train(training_path: Path, output_root: Path) -> dict:
    from nt_pipeline.train_nt import load_training_data, save_training_bundle, train_bundle

    log_stage(f"loading NT training data: {training_path}")
    training_df = load_training_data(training_path)
    log_stage("training NT model")
    bundle, metrics = train_bundle(training_df)

    model_root = output_root / "model"
    log_stage(f"saving NT model bundle: {model_root / 'model_bundle.pkl'}")
    save_training_bundle(bundle, model_root / "model_bundle.pkl")
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_foreign_features(cleaned_frames: dict[str, pd.DataFrame], output_root: Path) -> Path:
    log_stage("building foreign feature tables")
    feature_root = output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_tables = build_foreign_feature_tables(cleaned_frames, feature_root=feature_root)
    assemble_foreign_training_table(feature_tables, cleaned_frames, feature_root=feature_root)
    return feature_root / "feature_for_cus_acc.csv"


def train_foreign(training_path: Path, output_root: Path) -> dict:
    log_stage(f"loading foreign training data: {training_path}")
    log_stage("training foreign model")
    bundle, metrics = train_foreign_model_from_path(training_path)
    log_stage(f"saving foreign model bundle: {output_root / 'model' / 'model_bundle.pkl'}")
    save_foreign_model_artifacts(bundle, metrics, output_root)
    return metrics


def run_nt_pipeline(data_root: Path, output_root: Path) -> dict:
    from nt_pipeline.train_nt import load_training_data

    log_stage(f"starting NT pipeline with data root: {data_root}")
    paths = nt_raw_paths(data_root)
    log_stage("checking NT input files")
    ensure_inputs_exist(paths)

    log_stage("loading NT raw tables")
    raw_frames = load_csv_tables(paths, low_memory=False)
    write_stage_report("raw_eda", raw_frames, output_root)

    log_stage("preparing NT tables")
    cleaned_frames = prepare_nt_tables(raw_frames)
    save_cleaned_data(cleaned_frames, output_root)
    write_stage_report("cleaned_eda", cleaned_frames, output_root)

    training_path = build_features(cleaned_frames, output_root)
    log_stage(f"loading generated NT training features: {training_path}")
    training_df = load_training_data(training_path)
    write_stage_report("training_eda", {"feature_NT_cus_acc": training_df}, output_root)

    metrics = train(training_path, output_root)
    return {
        "training_path": str(training_path),
        "model_bundle": str(output_root / "model" / "model_bundle.pkl"),
        "metrics": metrics,
    }


def run_foreign_pipeline(data_root: Path, output_root: Path) -> dict:
    from foreign_pipeline.train import load_training_data as load_foreign_training_data

    log_stage(f"starting foreign pipeline with data root: {data_root}")
    paths = foreign_raw_paths(data_root)
    log_stage("checking foreign input files")
    ensure_foreign_inputs_exist(paths)

    log_stage("loading foreign raw tables")
    raw_frames = load_foreign_csv_tables(paths, low_memory=False)
    write_stage_report("raw_eda", raw_frames, output_root)

    log_stage("preparing foreign tables")
    cleaned_frames = prepare_foreign_feature_tables(raw_frames)
    save_cleaned_data(cleaned_frames, output_root)
    write_stage_report("cleaned_eda", cleaned_frames, output_root)

    training_path = build_foreign_features(cleaned_frames, output_root)
    log_stage(f"loading generated foreign training features: {training_path}")
    training_df = load_foreign_training_data(training_path)
    write_stage_report("training_eda", {"feature_for_cus_acc": training_df}, output_root)

    metrics = train_foreign(training_path, output_root)
    return {
        "training_path": str(training_path),
        "model_bundle": str(output_root / "model" / "model_bundle.pkl"),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    results = {}

    if args.pipeline in {"nt", "both"}:
        log_stage("running NT pipeline")
        results["nt"] = run_nt_pipeline(args.data_root, args.output_root / "nt")

    if args.pipeline in {"foreign", "both"}:
        log_stage("running foreign pipeline")
        results["foreign"] = run_foreign_pipeline(args.data_root, args.output_root / "foreign")

    log_stage("pipeline finished")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
