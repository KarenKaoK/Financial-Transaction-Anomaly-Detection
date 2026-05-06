import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DROP_COLUMNS = ["ACC_RANDOM", "ID_RANDOM", "SAMST_BR_CODE"]


def focal_loss(y_true, y_pred, alpha: float = 0.25, gamma: float = 2.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-10, 1 - 1e-10)
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = -alpha_t * np.power(1 - pt, gamma) * np.log(pt)
    return float(np.mean(loss))


def load_training_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, low_memory=False)


def build_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df_sar_1 = df[df["sar"] == 1].copy()
    df_sar_0 = df[df["sar"] == 0].copy()

    train_sar_1, test_sar_1 = train_test_split(df_sar_1, test_size=0.1, random_state=356)
    train_sar_0 = df_sar_0.sample(10000, random_state=222)
    df_sar_0_remaining = df_sar_0.drop(train_sar_0.index)
    test_sar_0 = df_sar_0_remaining.sample(10000, random_state=666)

    train_df = pd.concat([train_sar_1, train_sar_0]).reset_index(drop=True)
    test_df = pd.concat([test_sar_1, test_sar_0]).reset_index(drop=True)

    y_train = train_df.pop("sar").reset_index(drop=True)
    y_test = test_df.pop("sar").reset_index(drop=True)
    x_train = train_df.drop(columns=DROP_COLUMNS).reset_index(drop=True)
    x_test = test_df.drop(columns=DROP_COLUMNS).reset_index(drop=True)
    return x_train, x_test, y_train, y_test


def build_preprocessor(x_train: pd.DataFrame, x_test: pd.DataFrame):
    categorical_features = [col for col in x_train.columns if x_train[col].dtype == "object"]
    x_train = x_train.copy()
    x_test = x_test.copy()
    for column in categorical_features:
        x_train[column] = x_train[column].astype(str)
        x_test[column] = x_test[column].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    return preprocessor, x_train_transformed, x_test_transformed


def model_params() -> dict:
    return {
        "learning_rate": 0.01,
        "n_estimators": 1500,
        "max_depth": 5,
        "min_child_weight": 2,
        "gamma": 0.1,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.2,
        "reg_lambda": 1.1,
        "scale_pos_weight": 100,
        "objective": "binary:logistic",
        "eval_metric": focal_loss,
        "early_stopping_rounds": 50,
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
    x_train, x_test, y_train, y_test = build_split(df)
    preprocessor, x_train_encoded, x_test_encoded = build_preprocessor(x_train, x_test)

    model = xgb.XGBClassifier(**model_params())
    eval_set = [(x_train_encoded, y_train), (x_test_encoded, y_test)]
    model.fit(x_train_encoded, y_train, eval_set=eval_set, verbose=False)

    metrics = evaluate_model(model, x_train_encoded, y_train, x_test_encoded, y_test)
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "feature_columns": x_train.columns.tolist(),
        "drop_columns": DROP_COLUMNS,
    }
    return bundle, metrics


def save_training_bundle(bundle: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(bundle, file)


def train_model(input_path: Path, model_output: Path, preprocessor_output: Path, metrics_output: Path) -> dict:
    df = load_training_data(input_path)
    bundle, metrics = train_bundle(df)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as file:
        pickle.dump(model, file)
    joblib.dump(preprocessor, preprocessor_output)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Taiwan deposit XGBoost model.")
    parser.add_argument("--input", type=Path, required=True, help="Input feature_NT_cus_acc.csv path.")
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--preprocessor-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        input_path=args.input,
        model_output=args.model_output,
        preprocessor_output=args.preprocessor_output,
        metrics_output=args.metrics_output,
    )
