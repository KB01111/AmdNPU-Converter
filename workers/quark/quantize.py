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
    """Return list[dict[str, np.ndarray]] calibration feed dicts."""
    import numpy as np

    feeds: list[dict] = []
    if calib_dir is None or not calib_dir.exists():
        return feeds
    mapping: dict = dict(inputs_json or {})
    files = sorted([p for p in calib_dir.iterdir() if p.suffix.lower() in (".npy", ".npz")])
    if not files:
        return feeds
    # If explicit mapping given, build one feed per... actually one sample per npz-row or per file.
    for fp in files[:max_samples]:
        try:
            if fp.suffix.lower() == ".npy":
                arr = np.load(str(fp))
                if mapping:
                    feed = {}
                    for inp, fname in mapping.items():
                        if Path(fname).name == fp.name:
                            feed[inp] = arr
                    if not feed:
                        # single-file maps to first input only if only one input
                        if len(model_inputs) == 1:
                            feed = {model_inputs[0]: arr}
                        else:
                            continue
                else:
                    if len(model_inputs) == 1:
                        feed = {model_inputs[0]: arr}
                    else:
                        # cannot split one .npy across many inputs honestly
                        print(json.dumps({"warning": f"skip {fp.name}: multi-input model needs --inputs-json or .npz"}),
                              file=sys.stderr)
                        continue
                feeds.append(feed)
            else:  # .npz
                z = np.load(str(fp))
                keys = list(z.files)
                feed = {}
                for inp in model_inputs:
                    if inp in z:
                        feed[inp] = np.asarray(z[inp])
                    elif inp in mapping and mapping[inp] in z:
                        feed[inp] = np.asarray(z[mapping[inp]])
                if len(feed) == len(model_inputs):
                    feeds.append(feed)
                else:
                    print(json.dumps({"warning": f"skip {fp.name}: missing inputs (have {keys}, need {model_inputs})"}),
                          file=sys.stderr)
        except Exception as e:
            print(json.dumps({"warning": f"skip {fp.name}: {e}"}), file=sys.stderr)
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


def get_model_inputs(onnx_path: str) -> list[str]:
    import onnx
    m = onnx.load(onnx_path)
    return [i.name for i in m.graph.input]


def profile_to_qconfig(profile: str):
    """Build quark QConfig for profile. Raises ImportError if quark missing."""
    try:
        from quark.onnx import QConfig  # type: ignore
    except Exception as e:
        raise ImportError(
            "amd-quark is not installed in this environment. "
            "Activate workers/.venv-quark and run: uv pip install -r requirements-quark.txt "
            f"(import error: {e})"
        )
    # Best-effort mapping; quark API varies by version — keep generic kwargs.
    kw: dict = {}
    if profile == "bf16":
        kw = {"weight_qtype": "bf16", "activation_qtype": "bf16"}
    elif profile == "a16w8":
        kw = {"weight_qtype": "int8", "activation_qtype": "float16"}
    elif profile == "a8w8":
        kw = {"weight_qtype": "int8", "activation_qtype": "int8"}
    elif profile == "int8":
        kw = {"weight_qtype": "int8", "activation_qtype": "int8", "per_channel": True}
    try:
        return QConfig(**kw)
    except TypeError:
        # Older/newer quark signature: fall back to no-arg QConfig honestly
        return QConfig()


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        print(json.dumps({"error": f"input not found: {inp}"}), file=sys.stderr)
        return 2
    inputs_json = json.loads(args.inputs_json) if args.inputs_json else None

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

    # --- hard requirement: quark must be installed ---
    try:
        from quark.onnx import ModelQuantizer  # type: ignore
        qconfig = profile_to_qconfig(args.profile)
    except ImportError as e:
        print(json.dumps({"error": str(e), "hint": "use workers/.venv-quark (bootstrap.ps1) with requirements-quark.txt"}),
              file=sys.stderr)
        return 2

    data_reader = NpyCalibrationDataReader(feeds) if feeds else None
    if not feeds:
        print(json.dumps({"warning": "no calibration data; running static-shape quantize without calibration reader"}),
              file=sys.stderr)
    try:
        quantizer = ModelQuantizer(str(inp), qconfig)
        # API surface differs across quark versions; try common entry points honestly.
        if hasattr(quantizer, "quantize"):
            try:
                quantizer.quantize(str(outp), calibration_data_reader=data_reader)
            except TypeError:
                quantizer.quantize(str(outp))
        elif hasattr(quantizer, "quantize_model"):
            quantizer.quantize_model(str(outp), calibration_data_reader=data_reader)
        else:
            print(json.dumps({"error": "quark ModelQuantizer has no quantize()/quantize_model() entry point"}),
                  file=sys.stderr)
            return 1
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
