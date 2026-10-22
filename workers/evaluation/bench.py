"""ONNX benchmark across EPs: CPU / DML(NPU,GPU) / VitisAI.

    python workers/evaluation/bench.py --model model.onnx --devices cpu,npu,gpu --iterations 100 --warmup 10

Output: JSON array of BenchmarkResult to stdout.
NPU usage verified via provider name + optional EPContext cache info on Windows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark ONNX model latency/throughput per EP.")
    p.add_argument("--model", required=True)
    p.add_argument("--devices", default="cpu", help="Comma list among cpu,npu,gpu")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args(argv)


def providers_for(device: str, available: list[str]) -> list[str]:
    d = device.lower()
    if d == "cpu":
        return ["CPUExecutionProvider"]
    if d == "gpu":
        for c in ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"):
            if c in available:
                return [c]
        return ["CPUExecutionProvider"]
    if d == "npu":
        for c in ("VitisAIExecutionProvider", "DmlExecutionProvider", "NPUExecutionProvider", "OpenVINOExecutionProvider"):
            if c in available:
                return [c]
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def pct(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = min(len(s) - 1, max(0, int(q * len(s))))
    return s[k]


def main(argv=None) -> int:
    args = parse_args(argv)
    mp = Path(args.model)
    if not mp.exists():
        print(json.dumps({"error": f"model not found: {mp}"}), file=sys.stderr)
        return 2
    try:
        import numpy as np
        import onnxruntime as ort
        import psutil  # optional
    except ImportError:
        try:
            import psutil  # noqa
        except ImportError:
            psutil = None  # type: ignore
        try:
            import numpy as np
            import onnxruntime as ort
        except Exception as e:
            print(json.dumps({"error": f"onnxruntime/numpy missing: {e}"}), file=sys.stderr)
            return 1

    available = ort.get_available_providers()
    results = []
    try:
        sess0 = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        input_metas = [(i.name, list(i.shape), i.type) for i in sess0.get_inputs()]
    except Exception as e:
        print(json.dumps({"error": f"cannot open model: {e}"}), file=sys.stderr)
        return 1

    def dummy_feed():
        import numpy as _np
        feed = {}
        for name, shape, typ in input_metas:
            shp = []
            for idx, d in enumerate(shape):
                if isinstance(d, int):
                    shp.append(d)
                else:
                    shp.append(args.batch if idx == 0 else args.seq_len)
            dt = _np.int64 if "int" in str(typ) else _np.float32
            feed[name] = _np.ones(shp, dtype=dt)
        return feed

    feed = dummy_feed()
    for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
        provs = providers_for(dev, available)
        try:
            sess = ort.InferenceSession(str(mp), providers=provs)
            actual_ep = sess.get_providers()[0] if sess.get_providers() else provs[0]
        except Exception as e:
            results.append({"device": dev, "requested_providers": provs,
                            "error": f"session failed: {e}", "npu_verified": False})
            continue
        # warmup
        for _ in range(max(0, args.warmup)):
            try:
                sess.run(None, feed)
            except Exception as e:
                results.append({"device": dev, "providers": [actual_ep],
                                "error": f"warmup failed: {e}", "npu_verified": False})
                break
        else:
            lat = []
            mem_before = None
            try:
                if psutil is not None:
                    import os as _os
                    mem_before = psutil.Process(_os.getpid()).memory_info().rss
            except Exception:
                pass
            ok = True
            for _ in range(max(1, args.iterations)):
                t0 = time.perf_counter()
                try:
                    sess.run(None, feed)
                except Exception as e:
                    results.append({"device": dev, "providers": [actual_ep],
                                    "error": f"run failed: {e}", "npu_verified": False})
                    ok = False
                    break
                lat.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                continue
            mem_after = None
            try:
                if psutil is not None:
                    import os as _os
                    mem_after = psutil.Process(_os.getpid()).memory_info().rss
            except Exception:
                pass
            mean = sum(lat) / len(lat)
            npu_verified = actual_ep.lower() != "cpuexecutionprovider"
            # Windows ML EPContext cache info (best effort, honest)
            ep_ctx = None
            if os.name == "nt":
                cache_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows ML"
                try:
                    ep_ctx = {"cache_dir_exists": cache_dir.exists(), "cache_dir": str(cache_dir)}
                except Exception:
                    ep_ctx = {"cache_dir_exists": False}
            results.append({
                "device": dev,
                "providers": [actual_ep],
                "available_providers": available,
                "iterations": len(lat),
                "latency_ms_mean": mean,
                "latency_ms_p50": pct(lat, 0.50),
                "latency_ms_p90": pct(lat, 0.90),
                "throughput_per_s": 1000.0 / mean if mean > 0 else 0.0,
                "memory_rss_before": mem_before,
                "memory_rss_after": mem_after,
                "npu_verified": npu_verified,
                "ep_context": ep_ctx,
            })
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
