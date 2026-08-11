"""
generate_synthetic_data.py

Generates multi-sensor time-series data for a simulated portable radiation
detector instrument. Streams:
    - count_rate        : detector counts/sec (Poisson-ish, drifts with gain)
    - gain               : internal detector gain (drifts, causes count bias)
    - battery_voltage    : power monitor stream, sags over time / under load
    - board_temp_c       : temperature sensor, rises with duty cycle
    - signal_quality     : SNR-like metric from the front-end electronics
    - vibration           : motion sensor magnitude (mostly noise, spikes on shock)

Fault classes injected (labelled for supervised LSTM pretraining):
    0 = nominal
    1 = gain_drift_fault   (slow detector gain walk -> biased counts)
    2 = power_sag_fault    (battery voltage collapse under load)
    3 = thermal_fault      (overheating -> noisy/unreliable counts)
    4 = signal_fault       (front-end noise spike / cable fault)

This is synthetic-but-physically-motivated data: enough structure for the
LSTM to learn real temporal fault signatures, not just memorize noise.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimConfig:
    n_sequences: int = 400
    seq_len: int = 300          # timesteps per sequence (e.g. 300s @ 1Hz)
    dt: float = 1.0
    seed: int = 42


FAULT_NAMES = {
    0: "nominal",
    1: "gain_drift",
    2: "power_sag",
    3: "thermal",
    4: "signal_noise",
}


def _simulate_one_sequence(rng: np.random.Generator, cfg: SimConfig, fault_type: int):
    T = cfg.seq_len
    t = np.arange(T) * cfg.dt

    # baseline physical processes
    gain = np.full(T, 1.0)
    battery_voltage = np.full(T, 8.4)          # 2S li-ion pack, fully charged
    board_temp = 25.0 + rng.normal(0, 0.3, T).cumsum() * 0.02
    signal_quality = np.full(T, 0.95)
    vibration = np.abs(rng.normal(0, 0.05, T))

    duty_cycle = 0.5 + 0.1 * np.sin(2 * np.pi * t / 120)
    battery_drain_rate = 0.0006 * duty_cycle
    battery_voltage = battery_voltage - np.cumsum(battery_drain_rate)
    board_temp = board_temp + np.cumsum(duty_cycle * 0.01)

    fault_onset = rng.integers(int(T * 0.3), int(T * 0.7))
    label = np.zeros(T, dtype=int)

    if fault_type == 1:  # gain drift
        drift = np.clip((t - fault_onset), 0, None) * rng.uniform(0.001, 0.004)
        gain = gain + drift
        label[fault_onset:] = 1
    elif fault_type == 2:  # power sag
        sag = np.clip((t - fault_onset), 0, None) * rng.uniform(0.01, 0.03)
        battery_voltage = battery_voltage - sag
        label[fault_onset:] = 2
    elif fault_type == 3:  # thermal fault
        heat = np.clip((t - fault_onset), 0, None) * rng.uniform(0.03, 0.08)
        board_temp = board_temp + heat
        signal_quality = signal_quality - np.clip((t - fault_onset), 0, None) * 0.002
        label[fault_onset:] = 3
    elif fault_type == 4:  # signal/cable fault
        noise_burst = np.clip((t - fault_onset), 0, None) * rng.uniform(0.004, 0.01)
        signal_quality = signal_quality - noise_burst
        vibration[fault_onset:] += rng.normal(0, 0.3, T - fault_onset)
        label[fault_onset:] = 4

    # true underlying activity rate (counts/sec), modulated by gain/temp/signal
    base_rate = 120.0
    thermal_bias = 1.0 - 0.01 * np.clip(board_temp - 25, 0, None)
    signal_bias = np.clip(signal_quality, 0.05, 1.0)
    lam = np.clip(base_rate * gain * thermal_bias * signal_bias, 1.0, None)
    count_rate = rng.poisson(lam).astype(float)

    signal_quality = np.clip(signal_quality + rng.normal(0, 0.01, T), 0.0, 1.0)
    battery_voltage = np.clip(battery_voltage + rng.normal(0, 0.01, T), 5.0, 8.6)
    board_temp = board_temp + rng.normal(0, 0.15, T)
    vibration = np.clip(vibration, 0, None)

    df = pd.DataFrame({
        "t": t,
        "count_rate": count_rate,
        "gain": gain,
        "battery_voltage": battery_voltage,
        "board_temp_c": board_temp,
        "signal_quality": signal_quality,
        "vibration": vibration,
        "fault_label": label,
    })
    return df


def generate_dataset(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for seq_id in range(cfg.n_sequences):
        fault_type = rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.175, 0.175, 0.175, 0.175])
        df = _simulate_one_sequence(rng, cfg, fault_type)
        df.insert(0, "sequence_id", seq_id)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


if __name__ == "__main__":
    cfg = SimConfig()
    df = generate_dataset(cfg)
    out_path = "data/synthetic_sensor_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {df['sequence_id'].nunique()} sequences, {len(df)} rows -> {out_path}")
    print(df.groupby("sequence_id")["fault_label"].max().value_counts().rename(index=FAULT_NAMES))
