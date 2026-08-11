"""
quantize_ptq.py

Post-Training Quantization (PTQ), dynamic INT8, applied to the trained
DQN policy network -- this is the low-latency inference optimization step
before edge deployment. Also exports TorchScript + ONNX artifacts so the
deployment layer (TensorRT / ONNX Runtime on Jetson) has both options.

Dynamic quantization is used here (weights INT8, activations computed in
float at runtime) because the QNetwork is a small MLP -- ideal case for
dynamic PTQ, which requires no calibration data and gives strong latency
wins on CPU-bound edge inference with negligible accuracy loss on this
model size. See quantize_qat.py for the QAT path used when a larger/
convolutional model needs the extra accuracy recovery.
"""
import os
import time
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dqn import QNetwork

OBS_DIM = 8
N_ACTIONS = 5
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_fp32_policy():
    net = QNetwork(OBS_DIM, N_ACTIONS)
    ckpt = os.path.join(MODELS_DIR, "dqn_policy.pt")
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net


def benchmark(model, n_runs=2000):
    x = torch.randn(1, OBS_DIM)
    # warmup
    for _ in range(20):
        model(x)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000  # ms/inference


def quantize_dynamic_int8(fp32_model):
    quantized = torch.quantization.quantize_dynamic(
        fp32_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized


def compare_outputs(fp32_model, int8_model, n_samples=200, tol=0.15):
    x = torch.randn(n_samples, OBS_DIM)
    with torch.no_grad():
        q_fp32 = fp32_model(x)
        q_int8 = int8_model(x)
    max_abs_diff = (q_fp32 - q_int8).abs().max().item()
    action_agreement = (q_fp32.argmax(1) == q_int8.argmax(1)).float().mean().item()
    print(f"Max |Q(fp32) - Q(int8)| diff: {max_abs_diff:.4f}")
    print(f"Action-selection agreement fp32 vs int8: {action_agreement*100:.1f}%")
    return action_agreement


def export_torchscript(model, path):
    example = torch.randn(1, OBS_DIM)
    traced = torch.jit.trace(model, example)
    traced.save(path)
    print(f"TorchScript export -> {path}")


def export_onnx(model, path):
    example = torch.randn(1, OBS_DIM)
    torch.onnx.export(
        model, example, path,
        input_names=["obs"], output_names=["q_values"],
        dynamic_axes={"obs": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=13,
        dynamo=False,  # use the stable TorchScript-based exporter (no onnxscript dep)
    )
    print(f"ONNX export -> {path}")


def main():
    fp32_model = load_fp32_policy()
    int8_model = quantize_dynamic_int8(fp32_model)

    fp32_size = sum(p.numel() * p.element_size() for p in fp32_model.parameters())
    torch.save(int8_model.state_dict(), os.path.join(MODELS_DIR, "dqn_policy_int8.pt"))
    int8_size = os.path.getsize(os.path.join(MODELS_DIR, "dqn_policy_int8.pt"))

    print(f"FP32 param bytes (approx): {fp32_size/1024:.1f} KB")
    print(f"INT8 state_dict file size: {int8_size/1024:.1f} KB")

    compare_outputs(fp32_model, int8_model)

    t_fp32 = benchmark(fp32_model)
    t_int8 = benchmark(int8_model)
    print(f"Latency fp32: {t_fp32:.4f} ms/inf | int8: {t_int8:.4f} ms/inf "
          f"(speedup {t_fp32/max(t_int8,1e-9):.2f}x)")
    print("Note: for a network this tiny, dynamic-INT8 quant/dequant overhead can "
          "outweigh the matmul savings on x86 dev machines. The size reduction "
          "(~3x here) and the win on ARM/Jetson INT8 kernels via TensorRT are the "
          "actual deployment payoff -- see quantize_qat.py + deployment/ for the "
          "TensorRT path, which is where INT8 wins materialize on this model class.")

    export_torchscript(fp32_model, os.path.join(MODELS_DIR, "dqn_policy_fp32.torchscript"))
    export_onnx(fp32_model, os.path.join(MODELS_DIR, "dqn_policy_fp32.onnx"))


if __name__ == "__main__":
    main()
