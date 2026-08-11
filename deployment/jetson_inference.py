"""
jetson_inference.py

Inference bridge intended to run ON the Jetson board. Loads the exported
ONNX DQN policy via ONNX Runtime (falls back cleanly if TensorRT execution
provider isn't available on the current machine -- e.g. during dev-laptop
testing -- but on an actual Jetson with JetPack installed, TensorRTExecutionProvider
is what you'd select for the real low-latency path).

Sensor readings arrive from the C++ sensor_interface process over a simple
line-based protocol on stdin (JSON per line) -- see sensor_interface/main_inference_loop.cpp
for the C++ side of this bridge. This split (C++ for tight sensor-bus timing,
Python/ONNX Runtime for the model) mirrors how these systems are actually
built in the field: hard real-time I/O in C/C++, inference in a runtime with
mature INT8 kernel support.

CUDA / driver matching note: the ONNX Runtime GPU wheel and TensorRT version
used here must match the JetPack version flashed on the board (e.g. JetPack
5.1.x ships CUDA 11.4 + a specific TensorRT minor version) -- mismatches are
the #1 cause of "works on dev machine, segfaults on device" in this stack.
Pin these three versions together in requirements-jetson.txt, not just
requirements.txt.
"""
import json
import sys
import time
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

ACTIONS = ["NOMINAL_HOLD", "RECALIBRATE", "THROTTLE_SAMPLING",
           "BOOST_POWER_MODE", "FLAG_MAINTENANCE"]


class EdgePolicyRunner:
    def __init__(self, onnx_path: str, prefer_tensorrt=True):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime not installed. On Jetson: "
                                "pip install onnxruntime-gpu matching your JetPack/CUDA version.")
        providers = []
        available = ort.get_available_providers()
        if prefer_tensorrt and "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"Loaded {onnx_path} with providers: {self.session.get_providers()}")

    def infer(self, obs: np.ndarray) -> int:
        obs = obs.astype(np.float32).reshape(1, -1)
        q_values = self.session.run(None, {self.input_name: obs})[0]
        return int(np.argmax(q_values[0]))

    def infer_with_timing(self, obs: np.ndarray):
        t0 = time.perf_counter()
        action = self.infer(obs)
        t1 = time.perf_counter()
        return action, (t1 - t0) * 1000.0  # ms


def run_stdin_loop(onnx_path: str):
    """
    Reads one JSON object per line from stdin (as produced by the C++
    sensor_interface binary), runs inference, writes the chosen action
    (and latency) as JSON to stdout. This is the process boundary between
    the C++ hard-real-time sensor loop and the Python/ONNX inference engine.

    Expected input line:
      {"count_rate":118.0,"gain":1.02,"battery_voltage":8.1,"board_temp_c":31.2,
       "signal_quality":0.93,"vibration":0.08,"lstm_fault_prob":0.12,"steps_since_recal":0.4}
    """
    runner = EdgePolicyRunner(onnx_path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reading = json.loads(line)
            obs = np.array([
                reading["count_rate"], reading["gain"], reading["battery_voltage"],
                reading["board_temp_c"], reading["signal_quality"], reading["vibration"],
                reading["lstm_fault_prob"], reading["steps_since_recal"],
            ], dtype=np.float32)
            action, latency_ms = runner.infer_with_timing(obs)
            out = {"action": ACTIONS[action], "action_id": action, "latency_ms": latency_ms}
        except Exception as e:
            out = {"error": str(e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    import os
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "models",
                              "dqn_policy_fp32.onnx")
    if len(sys.argv) > 1 and sys.argv[1] == "--stdin-loop":
        run_stdin_loop(onnx_path)
    else:
        # standalone smoke test
        runner = EdgePolicyRunner(onnx_path)
        obs = np.array([120.0, 1.0, 8.2, 27.0, 0.94, 0.06, 0.1, 0.2], dtype=np.float32)
        action, latency_ms = runner.infer_with_timing(obs)
        print(f"Action: {ACTIONS[action]}  latency: {latency_ms:.4f} ms")
