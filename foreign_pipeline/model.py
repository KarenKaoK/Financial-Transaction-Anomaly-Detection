import json
import pickle
from pathlib import Path

import pandas as pd

from foreign_pipeline.train import load_training_data, save_training_bundle, train_bundle


def train_foreign_model_from_table(training_df: pd.DataFrame) -> tuple[dict, dict]:
    return train_bundle(training_df)


def train_foreign_model_from_path(training_path: Path) -> tuple[dict, dict]:
    training_df = load_training_data(training_path)
    return train_foreign_model_from_table(training_df)


def save_foreign_model_artifacts(bundle: dict, metrics: dict, output_root: Path) -> None:
    model_root = output_root / "model"
    save_training_bundle(bundle, model_root / "model_bundle.pkl")
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_foreign_model_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as file:
        return pickle.load(file)
