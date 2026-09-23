"""INT8 ONNX export for the on-lander deployment claim.

Exports the lunar model to ONNX, applies dynamic-range INT8 quantization,
verifies output agreement against PyTorch on real test windows, and measures
size + CPU latency. Writes models/lunar_best.onnx, models/lunar_best_int8.onnx
and results/quantization.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PROJECT_ROOT
from planetseis.model import ARCHS, SeisCNN
from planetseis.preprocessing import normalize_window

N = CFG.window.n_samples
MODELS = PROJECT_ROOT / "models"


def real_windows(k=64):
    xs = []
    for p in sorted((DATA_CACHE / "lunar" / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        tr = z["trace"]
        step = max(1, (len(tr) - N) // 8)
        for s in range(0, len(tr) - N, step):
            xs.append(normalize_window(tr[s:s + N]))
            if len(xs) >= k:
                return np.stack(xs)
    return np.stack(xs)


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ck = torch.load(PROJECT_ROOT / "runs" / "lunar" / "best.pt",
                    map_location="cpu", weights_only=True)
    model = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).eval()
    model.load_state_dict(ck["model"])

    fp32_path = MODELS / "lunar_best.onnx"
    int8_path = MODELS / "lunar_best_int8.onnx"
    dummy = torch.randn(1, 1, N)
    torch.onnx.export(model, dummy, fp32_path, input_names=["window"],
                      output_names=["logit", "offset", "heatmap"],
                      dynamic_axes={"window": {0: "batch"}}, opset_version=17)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    X = real_windows()
    with torch.no_grad():
        t_logit, t_off, _ = model(torch.from_numpy(X).unsqueeze(1))
    t_prob = torch.sigmoid(t_logit).numpy()

    results = {}
    for name, path in [("fp32_onnx", fp32_path), ("int8_onnx", int8_path)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        logit, off, _ = sess.run(None, {"window": X[:, None, :].astype(np.float32)})
        prob = 1 / (1 + np.exp(-logit))
        x1 = X[:1, None, :].astype(np.float32)
        for _ in range(5):
            sess.run(None, {"window": x1})
        t0 = time.perf_counter()
        for _ in range(50):
            sess.run(None, {"window": x1})
        ms = (time.perf_counter() - t0) / 50 * 1000
        results[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 3),
            "cpu_ms_per_window": round(ms, 2),
            "max_prob_dev_vs_torch": round(float(np.abs(prob - t_prob).max()), 5),
            "max_offset_dev_sec": round(float(np.abs(off - t_off.numpy()).max())
                                        * N / 6.625, 2),
        }
        print(name, results[name])

    results["torch_checkpoint_mb"] = round((PROJECT_ROOT / "runs" / "lunar" /
                                            "best.pt").stat().st_size / 1e6, 3)
    (PROJECT_ROOT / "results" / "quantization.json").write_text(
        json.dumps(results, indent=2))
    print("saved -> results/quantization.json")


if __name__ == "__main__":
    main()
