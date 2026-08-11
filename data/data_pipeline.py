"""
data_pipeline.py

Data engineering layer:
  - basic cleaning (clip physically impossible values, forward-fill short gaps)
  - resampling helper (in case raw sensor rate != model rate)
  - noise filtering (rolling median for spikes)
  - LEAKAGE-SAFE split: split by sequence_id, never by row, so no timesteps
    from the same recording session appear in both train and val/test.
  - windowing utility to build fixed-length LSTM training windows with labels.
"""
import numpy as np
import pandas as pd
from typing import Tuple

SENSOR_COLS = ["count_rate", "gain", "battery_voltage", "board_temp_c",
               "signal_quality", "vibration"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["battery_voltage"] = df["battery_voltage"].clip(0, 8.6)
    df["signal_quality"] = df["signal_quality"].clip(0, 1)
    df["count_rate"] = df["count_rate"].clip(0, None)
    df["vibration"] = df["vibration"].clip(0, None)
    # forward-fill isolated NaNs (sensor dropouts), then drop any leftover NaN rows
    df[SENSOR_COLS] = df.groupby("sequence_id")[SENSOR_COLS].ffill(limit=3)
    df = df.dropna(subset=SENSOR_COLS)
    return df


def denoise(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in SENSOR_COLS:
        df[col] = (
            df.groupby("sequence_id")[col]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
        )
    return df


def leakage_safe_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sequence_id so no sequence's timesteps leak across splits."""
    seq_ids = df["sequence_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(seq_ids)

    n = len(seq_ids)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_ids = set(seq_ids[:n_test])
    val_ids = set(seq_ids[n_test:n_test + n_val])
    train_ids = set(seq_ids[n_test + n_val:])

    train_df = df[df["sequence_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["sequence_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["sequence_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def fit_normalizer(train_df: pd.DataFrame):
    """Fit mean/std ONLY on train split -> avoids val/test leakage into scaling."""
    mean = train_df[SENSOR_COLS].mean()
    std = train_df[SENSOR_COLS].std().replace(0, 1.0)
    return mean, std


def apply_normalizer(df: pd.DataFrame, mean, std) -> pd.DataFrame:
    df = df.copy()
    df[SENSOR_COLS] = (df[SENSOR_COLS] - mean) / std
    return df


def make_windows(df: pd.DataFrame, window_len: int = 30, stride: int = 5):
    """
    Build (X, y) windows per sequence for LSTM supervised pretraining.
    X: [n_windows, window_len, n_sensors]
    y: [n_windows]  -> fault label at the LAST timestep of the window
       (so the model predicts "is a fault present/incipient right now")
    """
    X_list, y_list = [], []
    for _, seq in df.groupby("sequence_id"):
        seq = seq.sort_values("t")
        feats = seq[SENSOR_COLS].to_numpy(dtype=np.float32)
        labels = seq["fault_label"].to_numpy()
        for start in range(0, len(seq) - window_len, stride):
            end = start + window_len
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


if __name__ == "__main__":
    raw = pd.read_csv("data/synthetic_sensor_data.csv")
    df = denoise(clean(raw))
    train_df, val_df, test_df = leakage_safe_split(df)
    mean, std = fit_normalizer(train_df)
    train_n = apply_normalizer(train_df, mean, std)
    val_n = apply_normalizer(val_df, mean, std)
    test_n = apply_normalizer(test_df, mean, std)

    Xtr, ytr = make_windows(train_n)
    Xval, yval = make_windows(val_n)
    Xte, yte = make_windows(test_n)
    print("train windows:", Xtr.shape, "val:", Xval.shape, "test:", Xte.shape)
    assert set(train_df.sequence_id) & set(val_df.sequence_id) == set()
    assert set(train_df.sequence_id) & set(test_df.sequence_id) == set()
    print("Leakage check passed: no overlapping sequence_ids across splits.")
