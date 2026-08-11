"""
quantize_qat.py

Quantization-Aware Training (QAT) for FaultLSTM.

Why QAT here and PTQ for the DQN: the LSTM is the accuracy-critical fault
classifier feeding both the RL agent and the safety-relevant maintenance
decision. LSTMs are more sensitive to post-hoc quantization error than a
small MLP, so we fine-tune WITH simulated quantization noise in the loop
(fake-quant observers on weights/activations) for several epochs before
converting to a true INT8 module. This recovers most of the accuracy PTQ
would otherwise lose on recurrent architectures.

Note: PyTorch's native quantized LSTM kernels are dynamic-quantization only
(no full static/QAT LSTM cudnn kernel), so the QAT loop here fine-tunes
under fake-quantization for the Linear classifier head (which supports full
QAT) while the LSTM body is quantized dynamically post-training -- a
standard hybrid pattern for quantizing RNN-classifier models.
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.data_pipeline import clean, denoise, leakage_safe_split, fit_normalizer, \
    apply_normalizer, make_windows
from models.lstm_encoder import FaultLSTM

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_val_test_windows():
    import pandas as pd
    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "synthetic_sensor_data.csv"))
    df = denoise(clean(raw))
    train_df, val_df, test_df = leakage_safe_split(df)
    mean, std = fit_normalizer(train_df)
    train_n = apply_normalizer(train_df, mean, std)
    val_n = apply_normalizer(val_df, mean, std)
    Xtr, ytr = make_windows(train_n)
    Xval, yval = make_windows(val_n)
    return (Xtr, ytr), (Xval, yval)


def qat_finetune_head(model: FaultLSTM, Xtr, ytr, Xval, yval, epochs=3, lr=1e-4):
    """Fine-tune the classifier head with fake-quantization observers attached,
    while keeping the pretrained LSTM body frozen (stabilizes QAT for RNNs)."""
    model.classifier.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(model.classifier, inplace=True)

    for p in model.lstm.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"QAT fine-tune epoch {epoch+1}/{epochs}  loss={total/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(Xval))
        val_acc = (val_logits.argmax(-1) == torch.tensor(yval)).float().mean().item()
    print(f"Post-QAT-finetune val_acc (fake-quant): {val_acc:.4f}")

    model.classifier = torch.quantization.convert(model.classifier.eval(), inplace=False)
    return model


def main():
    ckpt = os.path.join(MODELS_DIR, "lstm_pretrained.pt")
    model = FaultLSTM()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    (Xtr, ytr), (Xval, yval) = load_val_test_windows()

    qat_model = qat_finetune_head(model, Xtr, ytr, Xval, yval)

    # dynamic-quantize the LSTM body (standard hybrid RNN quantization pattern)
    qat_model.lstm = torch.quantization.quantize_dynamic(
        qat_model.lstm, {nn.LSTM}, dtype=torch.qint8
    )

    out_path = os.path.join(MODELS_DIR, "lstm_qat_int8.pt")
    torch.save(qat_model.state_dict(), out_path)
    print(f"Saved QAT+dynamic-INT8 LSTM -> {out_path}")


if __name__ == "__main__":
    main()
