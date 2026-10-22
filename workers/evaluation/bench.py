"""Benchmark ONNX providers. NPU verification requires exclusive profiled NPU nodes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

NPU_PROVIDERS = {"VitisAIExecutionProvider", "NPUExecutionProvider"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark ONNX model latency/throughput per EP.")
    p.add_argument("--model")
    p.add_argument("--discover-json", action="store_true", help="List installed providers/devices without loading a model")
    p.add_argument("--devices", default="cpu", help="Comma list among cpu,npu,gpu")
    p.add_argument("--ep", default=None,
                   help="Pin vitisai/dml/cuda/cpu or an exact provider name for its matching device")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args(argv)


def check_ort_distributions():
    installed = []
    for name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-windowsml"):
        try:
            metadata.version(name)
            installed.append(name)
        except metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        raise ValueError(f"conflicting ONNX Runtime distributions share one namespace: {installed}; recreate the isolated worker environment")


def device_info(handle) -> dict:
    hardware = getattr(handle, "device", None) or getattr(handle, "hardware_device", None)
    kind = getattr(hardware, "type", None)
    kind = str(getattr(kind, "name", kind)).split(".")[-1].lower()
    details = getattr(hardware, "metadata", {})
    name = getattr(handle, "ep_metadata", {}).get("FULL_DEVICE_NAME") or details.get("Description", "unknown")
    return {"provider": handle.ep_name, "device": kind if kind in ("cpu", "gpu", "npu") else "unknown", "name": str(name)}


@contextmanager
def runtime_providers(ort):
    """Register installed Windows ML 2.0.300 providers; never acquire missing EPs."""
    report = {"providers": [], "devices": [], "errors": [], "windowsml_available": False}
    catalog, registered, handles = None, [], None
    try:
        if sys.platform == "win32":
            try:
                import windowsml
                catalog = windowsml.EpCatalog()
                report["windowsml_available"] = True
                for provider in catalog.find_all_providers():
                    try:
                        state = str(getattr(provider.ready_state, "name", provider.ready_state)).replace("_", "").split(".")[-1].lower()
                        if state == "notpresent":
                            continue
                        if state == "notready":
                            provider.ensure_ready()
                            state = str(getattr(provider.ready_state, "name", provider.ready_state)).replace("_", "").split(".")[-1].lower()
                        if state != "ready" or not provider.library_path:
                            raise ValueError("installed provider is not ready or has no library path")
                        ort.register_execution_provider_library(provider.name, provider.library_path)
                        registered.append(provider.name)
                    except Exception as e:
                        report["errors"].append(f"{provider.name}: {e}")
            except Exception as e:
                report["errors"].append(f"Windows ML catalog unavailable: {e}")
        report["providers"] = list(ort.get_available_providers())
        if hasattr(ort, "get_ep_devices"):
            handles = list(ort.get_ep_devices())
            report["devices"] = [device_info(handle) for handle in handles]
            report["providers"] = sorted(set(report["providers"]) | {handle.ep_name for handle in handles})
        yield report, handles
    finally:
        # Release device/session handles before unloading their registered libraries.
        if handles is not None:
            handles.clear()
        for name in reversed(registered):
            try:
                ort.unregister_execution_provider_library(name)
            except Exception as e:
                print(json.dumps({"warning": f"provider cleanup failed for {name}: {e}"}), file=sys.stderr)
        if catalog is not None:
            catalog.close()


def providers_for(device: str, available: list[str], ep: str | None = None) -> list[str]:
    choices = {
        "cpu": ("CPUExecutionProvider",),
        "gpu": ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"),
        "npu": ("VitisAIExecutionProvider", "NPUExecutionProvider"),
    }
    if device.lower() not in choices:
        raise ValueError(f"unknown device: {device}")
    if ep:
        aliases = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider",
                   "cuda": "CUDAExecutionProvider", "vitisai": "VitisAIExecutionProvider"}
        exact = {provider.lower(): provider for group in choices.values() for provider in group}
        pinned = aliases.get(ep.lower()) or exact.get(ep.lower())
        if pinned is None:
            raise ValueError(f"unknown execution provider: {ep}")
        if pinned in choices[device.lower()]:
            if pinned not in available:
                raise ValueError(f"requested provider {pinned} unavailable; available: {available}")
            return [pinned]
    for provider in choices[device.lower()]:
        if provider in available:
            return [provider]
    raise ValueError(f"requested {device} provider unavailable; available: {available}")


def pct(data: list[float], q: float) -> float:
    return sorted(data)[min(len(data) - 1, int(q * len(data)))]


def placement_from_profile(path: str, selected_provider: str) -> dict:
    """Provider registration alone says nothing about where kernels executed."""
    with open(path, encoding="utf-8") as source:
        events = json.load(source)
    nodes = [event for event in events if event.get("cat") == "Node"
             and event.get("name", "").endswith("_kernel_time")]
    providers = [node.get("args", {}).get("provider") for node in nodes]
    verified = bool(providers) and all(p == selected_provider for p in providers)
    return {"profiled_nodes": len(nodes),
            "node_providers": sorted({p for p in providers if isinstance(p, str)}),
            "provider_verified": verified,
            "npu_verified": verified and selected_provider in NPU_PROVIDERS}


def benchmark_model(args, ort, discovery, handles) -> int:
    devices = [d.strip().lower() for d in args.devices.split(",") if d.strip()]
    if (not devices or args.iterations < 1 or args.warmup < 0 or
            args.batch < 1 or args.seq_len < 1):
        print(json.dumps({"error": "devices and positive iterations/batch/seq-len, nonnegative warmup required"}), file=sys.stderr)
        return 2
    mp = Path(args.model)
    if not mp.is_file():
        print(json.dumps({"error": f"model not found: {mp}"}), file=sys.stderr)
        return 2
    import numpy as np
    try:
        import psutil
    except ImportError:
        psutil = None

    results = []
    available = discovery["providers"]
    for device in devices:
        row = {"device": device, "providers": [], "available_providers": available, "npu_verified": False}
        with tempfile.TemporaryDirectory(prefix="amdnpu-profile-") as profile_dir:
            sess = None
            try:
                requested = providers_for(device, available, args.ep)
                row["requested_providers"] = requested
                matches = None
                if handles is not None:
                    matches = [handle for handle in handles if handle.ep_name == requested[0]
                               and device_info(handle)["device"] == device]
                    if not matches:
                        raise ValueError(f"{requested[0]} exposes no matching {device} device")

                def create_session(profiling):
                    options = ort.SessionOptions()
                    options.enable_profiling = profiling
                    options.profile_file_prefix = str(Path(profile_dir) / "placement")
                    if matches is not None:
                        options.add_provider_for_devices([matches[0]], {})
                        session = ort.InferenceSession(str(mp), sess_options=options)
                    else:
                        session = ort.InferenceSession(str(mp), providers=requested, sess_options=options)
                    actual = session.get_providers()
                    row["providers"] = actual
                    if not actual or actual[0] != requested[0]:
                        raise ValueError(f"provider fallback: requested {requested}, session reports {actual}")
                    return session

                sess = create_session(device in ("npu", "gpu"))
                dtype_names = {"float": "float32", "float16": "float16", "double": "float64",
                               "int64": "int64", "int32": "int32", "int16": "int16",
                               "int8": "int8", "uint8": "uint8", "bool": "bool_"}
                feed = {}
                for item in sess.get_inputs():
                    name = item.type.removeprefix("tensor(").removesuffix(")")
                    if name not in dtype_names:
                        raise ValueError(f"unsupported synthetic benchmark input type: {item.type}")
                    shape = [d if isinstance(d, int) and d > 0 else (args.batch if i == 0 else args.seq_len)
                             for i, d in enumerate(item.shape)]
                    feed[item.name] = np.ones(shape, dtype=getattr(np, dtype_names[name]))
                if device in ("npu", "gpu"):
                    sess.run(None, feed)
                    try:
                        evidence = placement_from_profile(sess.end_profiling(), requested[0])
                        row["placement"] = evidence
                        row["npu_verified"] = evidence["npu_verified"] and "error" not in row
                        if device == "gpu" and not evidence["provider_verified"]:
                            row.setdefault("error", f"GPU node placement is not exclusive to {requested[0]}")
                    except Exception as e:
                        row["placement"] = {"npu_verified": False, "reason": f"profile unavailable: {e}"}
                        if device == "gpu":
                            row.setdefault("error", f"GPU node placement unverified: {e}")
                    # Release the profiled session before measuring latency and memory.
                    sess = None
                    sess = create_session(False)
                for _ in range(args.warmup):
                    sess.run(None, feed)
                process = psutil.Process(os.getpid()) if psutil is not None else None
                before = process.memory_info().rss if process else None
                latencies = []
                for _ in range(args.iterations):
                    start = time.perf_counter()
                    sess.run(None, feed)
                    latencies.append((time.perf_counter() - start) * 1000)
                after = process.memory_info().rss if process else None
                mean = sum(latencies) / len(latencies)
                if not math.isfinite(mean) or mean <= 0:
                    raise ValueError("benchmark timer returned invalid latency")
                row.update(iterations=len(latencies), latency_ms_mean=mean,
                           latency_ms_p50=pct(latencies, .5), latency_ms_p90=pct(latencies, .9),
                           throughput_per_s=1000 / mean,
                           memory_rss_before=before, memory_rss_after=after)
            except Exception as e:
                row["error"] = str(e)
            finally:
                row["npu_verified"] = row["npu_verified"] and "error" not in row
                sess = None
        results.append(row)
    print(json.dumps(results, allow_nan=False))
    return 1 if any("error" in row for row in results) else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.discover_json and not args.model:
        print(json.dumps({"error": "--model is required for benchmarking"}), file=sys.stderr)
        return 2
    try:
        check_ort_distributions()
        import onnxruntime as ort
        with runtime_providers(ort) as (report, handles):
            if args.discover_json:
                print(json.dumps(report, allow_nan=False))
                return 0 if report["providers"] else 1
            return benchmark_model(args, ort, report, handles)
    except Exception as e:
        if args.discover_json:
            print(json.dumps({"providers": [], "devices": [], "errors": [str(e)], "windowsml_available": False}))
        else:
            print(json.dumps({"error": f"benchmark runtime failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
