"""
lstm_encoder.py

LSTM sequence model over the fused multi-sensor stream.

Two use modes:
  1. Supervised pretraining: classify fault type from a sensor window
     (FaultLSTM.forward -> logits over 5 fault classes).
  2. Encoder fine-tuning for RL: FaultLSTM.encode(window) returns the final
     hidden state + a scalar "fault probability" (1 - P(nominal)) that feeds
     directly into DetectorFaultControlEnv's observation / the DQN's input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SENSORS = 6
N_CLASSES = 5  # nominal, gain_drift, power_sag, thermal, signal_noise


class FaultLSTM(nn.Module):
    def __init__(self, n_sensors=N_SENSORS, hidden_size=64, num_layers=2,
                 n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x: [batch, seq_len, n_sensors]
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]                 # [batch, hidden_size]
        logits = self.classifier(last_hidden)  # [batch, n_classes]
        return logits

    @torch.no_grad()
    def encode(self, window: torch.Tensor):
        """
        window: [seq_len, n_sensors] or [batch, seq_len, n_sensors]
        returns: (hidden_state [hidden_size], fault_prob scalar in [0,1])
        Used at inference time to feed the RL agent's observation.
        """
        self.eval()
        if window.dim() == 2:
            window = window.unsqueeze(0)
        out, (h_n, c_n) = self.lstm(window)
        last_hidden = h_n[-1]                       # [batch, hidden_size]
        logits = self.classifier(last_hidden)
        probs = F.softmax(logits, dim=-1)
        fault_prob = 1.0 - probs[:, 0]               # P(not nominal)
        return last_hidden.squeeze(0), fault_prob.squeeze(0).item()


if __name__ == "__main__":
    model = FaultLSTM()
    x = torch.randn(8, 30, N_SENSORS)  # batch=8, seq_len=30
    logits = model(x)
    print("logits shape:", logits.shape)
    h, p = model.encode(x[0])
    print("encoded hidden shape:", h.shape, "fault_prob:", round(p, 3))
    n_params = sum(p.numel() for p in model.parameters())
    print("Param count:", n_params)
