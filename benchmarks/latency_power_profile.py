"""
latency_power_profile.py

On-device benchmarking: inference latency distribution (mean/p50/p99/max)
for fp32 vs INT8 policy, plus a power-draw proxy.

On a real Jetson, replace `read_power_mw()` with a call to the board's
INA3221 power-monitor sysfs nodes (e.g.
/sys/bus/i2c/drivers/ina3221x/.../in_power0_input on Nano/Xavier, or
`tegrastats` parsing on newer JetPack) to get true instantaneous power in
mW during the benchmark loop. That hook point is marked below.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn import QNetwork

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OBS_DIM, N_ACTIONS = 8, 5


def read_power_mw():
    """
    Proxy power reading. On a real Jetson, replace this with a read from
    the board's power-monitor sysfs node, e.g.:
        with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input') as f:
            return float(f.read().strip())
    Returns None here (no real power rail available on a dev machine) so the
    benchmark still runs and clearly reports that power numbers are N/A off-device.
    """
    return None


def benchmark_model(model, n_runs=3000, batch=1):
    model.eval()
    x = torch.randn(batch, OBS_DIM)
    with torch.no_grad():
        for _ in range(50):  # warmup
            model(x)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return np.array(latencies)


def report(name, latencies):
    print(f"--- {name} ---")
    print(f"  mean:   {latencies.mean():.4f} ms")
    print(f"  p50:    {np.percentile(latencies, 50):.4f} ms")
    print(f"  p99:    {np.percentile(latencies, 99):.4f} ms")
    print(f"  max:    {latencies.max():.4f} ms")
    print(f"  throughput: {1000.0/latencies.mean():.1f} inferences/sec (single-sample)")


def main():
    fp32 = QNetwork(OBS_DIM, N_ACTIONS)
    fp32.load_state_dict(torch.load(os.path.join(MODELS_DIR, "dqn_policy.pt"),
                                     map_location="cpu"))

    int8 = torch.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)

    lat_fp32 = benchmark_model(fp32)
    lat_int8 = benchmark_model(int8)

    report("FP32 policy", lat_fp32)
    report("INT8 (dynamic PTQ) policy", lat_int8)

    power_mw = read_power_mw()
    if power_mw is None:
        print("\nPower draw: N/A on this machine (no Jetson power-monitor sysfs node). "
              "On-device, this script reads in_power0_input during the benchmark loop "
              "to report mW per inference batch alongside latency.")
    else:
        print(f"\nPower draw during benchmark: {power_mw:.1f} mW")


if __name__ == "__main__":
    main()
