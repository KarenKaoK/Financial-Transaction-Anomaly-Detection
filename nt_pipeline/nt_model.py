import json
import pickle
from pathlib import Path

import pandas as pd

from nt_pipeline.train_nt import load_training_data, save_training_bundle, train_bundle


def train_nt_model_from_table(training_df: pd.DataFrame) -> tuple[dict, dict]:
    return train_bundle(training_df)


def train_nt_model_from_path(training_path: Path) -> tuple[dict, dict]:
    training_df = load_training_data(training_path)
    return train_nt_model_from_table(training_df)


def save_nt_model_artifacts(bundle: dict, metrics: dict, output_root: Path) -> None:
    model_root = output_root / "model"
    save_training_bundle(bundle, model_root / "model_bundle.pkl")
    (model_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_nt_model_bundle(bundle_path: Path) -> dict:
    with open(bundle_path, "rb") as file:
        return pickle.load(file)


def score_nt_table(scoring_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features = scoring_df.copy()
    feature_columns = bundle["feature_columns"]
    drop_columns = bundle["drop_columns"]

    x = features.drop(columns=[col for col in drop_columns if col in features.columns], errors="ignore")
    x = x.reindex(columns=feature_columns)

    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    x_encoded = preprocessor.transform(x)

    output = scoring_df.copy()
    output["pred_sar"] = model.predict(x_encoded)
    if hasattr(model, "predict_proba"):
        output["pred_sar_proba"] = model.predict_proba(x_encoded)[:, 1]
    return output
