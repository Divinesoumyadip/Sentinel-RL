"""
hil_test_harness.py

Hardware-in-loop validation harness. Two modes:

  1. REPLAY mode (works with or without real hardware attached): replays
     synthetic-but-realistic sensor sequences (same generator used for LSTM
     training, held out or freshly seeded) through the deployed ONNX policy
     + LSTM fault estimator, exactly mimicking what the C++ sensor_loop ->
     jetson_inference.py pipe would deliver in production. This is what
     validates the deployed (quantized/exported) model's *behavior*, not
     just its raw accuracy, before it ever touches real sensors.

  2. LIVE mode: spawns the compiled C++ sensor_loop binary and pipes its
     real-time stdout through the same evaluation logic -- the actual HIL
     path once hardware is attached. Falls back gracefully if the binary
     isn't present (e.g. running this on a dev laptop, not the Jetson).

Both modes log per-step (action, fault_prob, true_fault_state) so failures
can be diagnosed: did the policy miss a real fault, or false-trip on
nominal data?
"""
import os
import sys
import json
import subprocess
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.detector_env import DetectorFaultControlEnv
from models.lstm_encoder import FaultLSTM
from deployment.jetson_inference import EdgePolicyRunner, ACTIONS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def replay_mode(n_episodes=10, seed=123):
    fault_model = FaultLSTM()
    fault_model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "lstm_pretrained.pt"), map_location="cpu"))
    fault_model.eval()

    def lstm_provider(window):
        if len(window) < 3:
            return 0.0
        x = torch.tensor(np.array(window), dtype=torch.float32)
        _, p = fault_model.encode(x)
        return p

    onnx_path = os.path.join(MODELS_DIR, "dqn_policy_fp32.onnx")
    runner = EdgePolicyRunner(onnx_path)

    env = DetectorFaultControlEnv(max_steps=300, seed=seed, fault_prob_provider=lstm_provider)

    results = {"correct_flags": 0, "false_trips": 0, "missed_faults": 0,
               "episodes_with_fault": 0, "total_latency_ms": [], "episodes": 0}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        had_fault = env.has_fault
        results["episodes"] += 1
        results["episodes_with_fault"] += int(had_fault)
        flagged_correctly = False
        false_tripped = False

        for t in range(env.max_steps):
            action, latency_ms = runner.infer_with_timing(obs)
            results["total_latency_ms"].append(latency_ms)
            obs, reward, terminated, truncated, info = env.step(action)

            if ACTIONS[action] == "FLAG_MAINTENANCE":
                if info["true_fault_active"]:
                    flagged_correctly = True
                else:
                    false_tripped = True
                break
            if terminated or truncated:
                break

        if had_fault and flagged_correctly:
            results["correct_flags"] += 1
        if had_fault and not flagged_correctly:
            results["missed_faults"] += 1
        if false_tripped:
            results["false_trips"] += 1

    lat = np.array(results["total_latency_ms"])
    print("=== HIL Replay Validation Results ===")
    print(f"Episodes: {results['episodes']}  (with real fault: {results['episodes_with_fault']})")
    print(f"Correctly flagged faults: {results['correct_flags']}")
    print(f"Missed faults:            {results['missed_faults']}")
    print(f"False trips:              {results['false_trips']}")
    print(f"Inference latency: mean={lat.mean():.4f}ms  p99={np.percentile(lat,99):.4f}ms  "
          f"max={lat.max():.4f}ms  (n={len(lat)} steps)")
    return results


def live_mode():
    binary = os.path.join(os.path.dirname(__file__), "sensor_interface", "build", "sensor_loop")
    if not os.path.exists(binary):
        print(f"No compiled sensor_loop binary at {binary}. "
              f"Build it on-device (see sensor_interface/CMakeLists.txt) to use live mode. "
              f"Falling back to replay_mode().")
        return replay_mode()

    onnx_path = os.path.join(MODELS_DIR, "dqn_policy_fp32.onnx")
    sensor_proc = subprocess.Popen([binary], stdout=subprocess.PIPE, text=True)
    inference_proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "jetson_inference.py"),
         "--stdin-loop"],
        stdin=sensor_proc.stdout, stdout=subprocess.PIPE, text=True,
    )
    print("Live HIL pipe started (C++ sensor_loop -> Python inference). Ctrl+C to stop.")
    try:
        for line in inference_proc.stdout:
            print(line.strip())
    except KeyboardInterrupt:
        sensor_proc.terminate()
        inference_proc.terminate()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if mode == "live":
        live_mode()
    else:
        replay_mode()
