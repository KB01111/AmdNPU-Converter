"""Quark ONNX quantization worker.

Uses amd-quark (quark.onnx.ModelQuantizer + QConfig) when installed.
NO silent fallback: if quark is missing -> exit 2 with clear message.

Calibration: loads .npy/.npz files from --calibration-dir and feeds ALL
model inputs (name -> np array). --inputs-json optionally maps
input-name -> filename inside the calibration dir.

Post-quant validation: onnx.checker + onnxruntime CPU session; optional
--reference-metrics JSON file {"accuracy": float} to report accuracy-delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROFILES = ["bf16", "a16w8", "a8w8", "int8"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Quantize ONNX with AMD Quark (honest, no fake fallback).")
    p.add_argument("--input", required=True, help="Input .onnx path")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--profile", required=True, choices=PROFILES, help="Quant profile")
    p.add_argument("--calibration-dir", default=None, help="Dir with .npy/.npz calibration samples")
    p.add_argument("--inputs-json", default=None, help='JSON mapping input-name -> file, e.g. {"input_ids":"ids.npy"}')
    p.add_argument("--reference-metrics", default=None, help='JSON file {"accuracy": 0.9} for delta report')
    p.add_argument("--max-samples", type=int, default=64, help="Max calibration samples (default 64)")
    return p.parse_args(argv)


def load_calibration(model_inputs: list[str], calib_dir: Path | None,
                     inputs_json: dict | None, max_samples: int):
    import numpy as np

    if max_samples < 1:
        raise ValueError('max-samples must be positive')
    if calib_dir is None:
        return []
    if not calib_dir.is_dir():
        raise ValueError('calibration directory does not exist')
    mapping = dict(inputs_json or {})
    if mapping and set(mapping) != set(model_inputs):
        raise ValueError('input mapping must contain every model input exactly once')

    def checked(feed):
        if set(feed) != set(model_inputs):
            raise ValueError('sample must contain every model input')
        if any(a.size == 0 or not np.isfinite(a).all() for a in feed.values()):
            raise ValueError('calibration arrays must be nonempty and finite')
        return feed

    if mapping and all(str(name).endswith('.npy') for name in mapping.values()):
        feed = {}
        for name, filename in mapping.items():
            path = (calib_dir / filename).resolve()
            if not path.is_relative_to(calib_dir.resolve()) or not path.is_file():
                raise ValueError('mapped calibration file must exist inside calibration directory')
            feed[name] = np.asarray(np.load(str(path), allow_pickle=False))
        return [checked(feed)]

    feeds = []
    files = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in ('.npy', '.npz'))
    for path in files[:max_samples]:
        if path.suffix.lower() == '.npz':
            with np.load(str(path), allow_pickle=False) as data:
                feed = {name: np.asarray(data[mapping.get(name, name)]) for name in model_inputs}
        elif len(model_inputs) == 1:
            feed = {model_inputs[0]: np.asarray(np.load(str(path), allow_pickle=False))}
        else:
            raise ValueError('multi-input models require complete NPZ samples or an explicit NPY mapping')
        feeds.append(checked(feed))
    return feeds


class NpyCalibrationDataReader:
    """Minimal ORT CalibrationDataReader over preloaded feed dicts."""

    def __init__(self, feeds: list[dict]):
        self._feeds = feeds
        self._it = iter(feeds)

    def get_next(self):
        try:
            return next(self._it)
        except StopIteration:
            return None

    def rewind(self):
        self._it = iter(self._feeds)


def get_model_inputs(onnx_path: str) -> list[str]:
    import onnx
    m = onnx.load(onnx_path)
    initializers = {i.name for i in m.graph.initializer}
    return [i.name for i in m.graph.input if i.name not in initializers]


def profile_to_qconfig(profile: str):
    """Use documented presets, never substitute an unrelated default profile."""
    presets = {"bf16": "BF16", "a16w8": "A16W8", "a8w8": "A8W8", "int8": "XINT8"}
    if profile not in presets:
        raise ValueError(f"unsupported quantization profile: {profile}")
    try:
        from quark.onnx import QConfig  # type: ignore
    except ImportError:
        # Quark 0.8/0.9 use the earlier documented configuration API.
        from quark.onnx.quantization.config import Config, get_default_config
        return Config(global_quant_config=get_default_config(presets[profile]))
    return QConfig.get_default_config(presets[profile])


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        print(json.dumps({"error": f"input not found: {inp}"}), file=sys.stderr)
        return 2
    try:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        inputs_json = json.loads(args.inputs_json) if args.inputs_json else None
        if inputs_json is not None and not isinstance(inputs_json, dict):
            raise ValueError("inputs-json must be a mapping")
        if inp.resolve() == outp.resolve() or outp.exists():
            raise ValueError("output must be a new path distinct from input")
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    try:
        model_inputs = get_model_inputs(str(inp))
    except Exception as e:
        print(json.dumps({"error": f"cannot read ONNX inputs: {e}"}), file=sys.stderr)
        return 1

    calib_dir = Path(args.calibration_dir) if args.calibration_dir else None
    try:
        feeds = load_calibration(model_inputs, calib_dir, inputs_json, args.max_samples)
    except Exception as e:
        print(json.dumps({"error": f"calibration load failed: {e}"}), file=sys.stderr)
        return 1
    if args.profile != "bf16" and not feeds:
        print(json.dumps({"error": "representative calibration data for every model input is required"}), file=sys.stderr)
        return 2

    # --- hard requirement: quark must be installed ---
    try:
        from quark.onnx import ModelQuantizer  # type: ignore
        qconfig = profile_to_qconfig(args.profile)
    except Exception as e:
        print(json.dumps({"error": str(e), "hint": "use workers/.venv-quark (bootstrap.ps1) with requirements-quark.txt"}),
              file=sys.stderr)
        return 2

    data_reader = NpyCalibrationDataReader(feeds) if feeds else None
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
        quantizer = ModelQuantizer(qconfig)
        quantizer.quantize_model(str(inp), str(outp), data_reader)
    except Exception as e:
        print(json.dumps({"error": f"quark quantization failed: {e}"}), file=sys.stderr)
        return 1

    # --- validate output ---
    report: dict = {"input": str(inp), "output": str(outp), "profile": args.profile,
                    "calibration_samples": len(feeds), "inputs": model_inputs}
    try:
        import onnx
        onnx.checker.check_model(str(outp))
        report["onnx_checker"] = "ok"
    except Exception as e:
        report["onnx_checker"] = f"FAIL: {e}"
        print(json.dumps(report), file=sys.stderr)
        return 1
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(outp), providers=["CPUExecutionProvider"])
        report["ort_cpu_load"] = "ok"
        report["ort_inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]
    except Exception as e:
        report["ort_cpu_load"] = f"FAIL: {e}"
        print(json.dumps(report), file=sys.stderr)
        return 1

    if args.reference_metrics:
        try:
            with open(args.reference_metrics, "r", encoding="utf-8") as f:
                ref = json.load(f)
            base = float(ref.get("accuracy", 0.0))
            # Honest: we cannot measure task accuracy without an eval set here;
            # report delta as unevaluated instead of fabricating a number.
            report["reference_accuracy"] = base
            report["accuracy_delta"] = None
            report["accuracy_note"] = "accuracy requires workers/evaluation pipeline; see compare.py"
        except Exception as e:
            report["reference_metrics_error"] = str(e)

    try:
        report["output_bytes"] = outp.stat().st_size
        report["input_bytes"] = inp.stat().st_size
    except Exception:
        pass
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
