import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DROP_COLUMNS = ["sar", "ACC_RANDOM", "ID_RANDOM"]
OPTIONAL_DROP_COLUMNS = [
    "AGE",
    "CFACC_OPEN_DATE",
    "CFACC_CHG_DATE",
    "CFMST_CRT_DATE",
    "CFMST_CHG_DATE",
    "CFACC_BR_CODE",
    "CFACC_BUSI_CODE",
]


def load_training_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, low_memory=False)


def build_split(df: pd.DataFrame):
    df_sar_1 = df[df["sar"] == 1].copy()
    df_sar_0 = df[df["sar"] == 0].copy()

    if len(df_sar_1) < 2 or len(df_sar_0) < 2:
        raise ValueError("Foreign training data requires at least 2 positive and 2 negative samples.")

    test_size = max(1, int(round(len(df_sar_1) * 0.10)))
    test_size = min(test_size, len(df_sar_1) - 1)
    train_sar_1, test_sar_1 = train_test_split(df_sar_1, test_size=test_size, random_state=42)

    train_negative_size = min(2000, max(1, len(df_sar_0) - 1))
    train_sar_0 = df_sar_0.sample(train_negative_size, random_state=333)
    df_sar_0_remaining = df_sar_0.drop(train_sar_0.index)
    if df_sar_0_remaining.empty:
        extra_index = train_sar_0.sample(1, random_state=444).index
        df_sar_0_remaining = train_sar_0.loc[extra_index]
        train_sar_0 = train_sar_0.drop(extra_index)
    test_negative_size = min(300, len(df_sar_0_remaining))
    test_sar_0 = df_sar_0_remaining.sample(test_negative_size, random_state=444)

    train_df = pd.concat([train_sar_1, train_sar_0]).reset_index(drop=True)
    test_df = pd.concat([test_sar_1, test_sar_0]).reset_index(drop=True)

    y_train = train_df["sar"].reset_index(drop=True)
    y_test = test_df["sar"].reset_index(drop=True)
    drop_columns = DROP_COLUMNS + [col for col in OPTIONAL_DROP_COLUMNS if col in train_df.columns]
    x_train = train_df.drop(columns=drop_columns).reset_index(drop=True)
    x_test = test_df.drop(columns=drop_columns).reset_index(drop=True)
    return x_train, x_test, y_train, y_test, drop_columns


def build_encoder(x_train: pd.DataFrame, x_test: pd.DataFrame):
    categorical_features = [col for col in x_train.columns if x_train[col].dtype == "object"]
    x_train = x_train.copy()
    x_test = x_test.copy()
    for column in categorical_features:
        x_train[column] = x_train[column].astype(str)
        x_test[column] = x_test[column].astype(str)

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )
    x_train_encoded = encoder.fit_transform(x_train)
    x_test_encoded = encoder.transform(x_test)
    return encoder, x_train_encoded, x_test_encoded


def model_params() -> dict:
    return {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 3,
        "min_child_weight": 2,
        "gamma": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": 80,
        "reg_alpha": 0.15,
        "reg_lambda": 1.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }


def evaluate_model(model, x_train, y_train, x_test, y_test) -> dict:
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    return {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "train_recall": float(recall_score(y_train, y_pred_train)),
        "train_precision": float(precision_score(y_train, y_pred_train, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_recall": float(recall_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_positive_predictions": int(np.asarray(y_pred_test).sum()),
    }


def train_bundle(df: pd.DataFrame) -> tuple[dict, dict]:
    x_train, x_test, y_train, y_test, drop_columns = build_split(df)
    encoder, x_train_encoded, x_test_encoded = build_encoder(x_train, x_test)

    model = xgb.XGBClassifier(**model_params())
    eval_set = [(x_train_encoded, y_train), (x_test_encoded, y_test)]
    model.fit(x_train_encoded, y_train, eval_set=eval_set, verbose=False)

    metrics = evaluate_model(model, x_train_encoded, y_train, x_test_encoded, y_test)
    bundle = {
        "model": model,
        "encoder": encoder,
        "metrics": metrics,
        "feature_columns": x_train.columns.tolist(),
        "drop_columns": drop_columns,
    }
    return bundle, metrics


def save_training_bundle(bundle: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(bundle, file)


def train_model(input_path: Path, model_output: Path, encoder_output: Path, metrics_output: Path) -> dict:
    df = load_training_data(input_path)
    bundle, metrics = train_bundle(df)
    model = bundle["model"]
    encoder = bundle["encoder"]

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as file:
        pickle.dump(model, file)
    with open(encoder_output, "wb") as file:
        pickle.dump(encoder, file)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train foreign-currency XGBoost model.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--encoder-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        input_path=args.input,
        model_output=args.model_output,
        encoder_output=args.encoder_output,
        metrics_output=args.metrics_output,
    )
