from pathlib import Path

import pandas as pd

from foreign_pipeline.config import raw_paths


def ensure_inputs_exist(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")


def load_csv_tables(paths: dict[str, Path], low_memory: bool = False) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path, low_memory=low_memory) for name, path in paths.items()}


def load_raw_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    paths = raw_paths(data_root)
    ensure_inputs_exist(paths)
    return load_csv_tables(paths, low_memory=False)
