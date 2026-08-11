"""
pretrain_lstm.py

Supervised pretraining of FaultLSTM on synthetic multi-sensor windows,
using the leakage-safe train/val/test split from data_pipeline.py.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.data_pipeline import clean, denoise, leakage_safe_split, fit_normalizer, \
    apply_normalizer, make_windows
from models.lstm_encoder import FaultLSTM


def load_windows(window_len=30, stride=5, seed=42):
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "synthetic_sensor_data.csv"))
    df = denoise(clean(raw))
    train_df, val_df, test_df = leakage_safe_split(df, seed=seed)
    mean, std = fit_normalizer(train_df)  # fit ONLY on train -> no leakage

    train_n = apply_normalizer(train_df, mean, std)
    val_n = apply_normalizer(val_df, mean, std)
    test_n = apply_normalizer(test_df, mean, std)

    Xtr, ytr = make_windows(train_n, window_len, stride)
    Xval, yval = make_windows(val_n, window_len, stride)
    Xte, yte = make_windows(test_n, window_len, stride)
    return (Xtr, ytr), (Xval, yval), (Xte, yte), (mean, std)


def train(epochs=8, batch_size=128, lr=1e-3, device="cpu"):
    (Xtr, ytr), (Xval, yval), (Xte, yte), _ = load_windows()

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(Xval), torch.tensor(yval, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = FaultLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb.to(device))
                pred = logits.argmax(dim=-1).cpu()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_pretrained.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved pretrained LSTM -> {ckpt_path}  (best val_acc={best_val_acc:.4f})")
    return model


if __name__ == "__main__":
    train()
