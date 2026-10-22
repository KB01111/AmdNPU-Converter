"""Shared exporter contract + validation helpers."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExportSpec:
    architectures: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)      # name -> example spec str
    outputs: list = field(default_factory=list)     # output names
    dynamic_axes: dict = field(default_factory=dict)
    opset: int = 17


def load_config(model_src: str, token: str | None = None) -> dict:
    """Load HF config.json from repo id or local dir/file. Never fabricates."""
    from huggingface_hub import hf_hub_download

    p = Path(model_src)
    if p.is_file() and p.name == "config.json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    if p.is_dir() and (p / "config.json").exists():
        with open(p / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    # assume HF repo id
    local = hf_hub_download(repo_id=model_src, filename="config.json", token=token)
    with open(local, "r", encoding="utf-8") as f:
        return json.load(f)


def make_example_inputs(tokenizer_or_cfg, seq_len: int = 128, batch: int = 1) -> dict:
    """Build example inputs (honest zeros/ones) for export tracing."""
    import numpy as np

    return {
        "input_ids": np.ones((batch, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
    }


def validate_onnx(path: str, example_inputs: dict, torch_outputs=None,
                   rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    """Run ORT on example_inputs; optionally compare vs torch_outputs with allclose.

    Returns JSON-serialisable report dict. Never raises on numeric mismatch —
    reports max_abs_diff instead.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort

    report: dict = {"model": path}
    try:
        onnx.checker.check_model(path)
        report["onnx_checker"] = "ok"
    except Exception as e:
        report["onnx_checker"] = f"FAIL: {e}"
        return report
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        report["ort_load"] = f"FAIL: {e}"
        return report
    report["ort_load"] = "ok"
    ort_inputs_meta = [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]
    report["ort_inputs"] = ort_inputs_meta
    feed = {}
    for meta in sess.get_inputs():
        if meta.name in example_inputs:
            v = example_inputs[meta.name]
            try:
                import torch as _t
                if isinstance(v, _t.Tensor):
                    v = v.detach().cpu().numpy()
            except Exception:
                pass
            feed[meta.name] = np.asarray(v)
        else:
            # honest: missing input — cannot run
            report["ort_run"] = f"FAIL: missing example input '{meta.name}'"
            return report
    try:
        outs = sess.run(None, feed)
        report["ort_run"] = "ok"
        report["num_outputs"] = len(outs)
        report["output_shapes"] = [list(o.shape) for o in outs]
    except Exception as e:
        report["ort_run"] = f"FAIL: {e}"
        return report
    if torch_outputs is not None:
        try:
            import numpy as _np
            if not isinstance(torch_outputs, (list, tuple)):
                torch_outputs = [torch_outputs]
            diffs = []
            for to, oo in zip(torch_outputs, outs):
                try:
                    import torch as _t
                    if isinstance(to, _t.Tensor):
                        to = to.detach().cpu().numpy()
                except Exception:
                    pass
                to = _np.asarray(to, dtype=_np.float64)
                oo = _np.asarray(oo, dtype=_np.float64)
                if to.shape != oo.shape:
                    diffs.append({"shape_mismatch": [list(to.shape), list(oo.shape)]})
                else:
                    diffs.append({"max_abs_diff": float(_np.max(_np.abs(to - oo))),
                                  "allclose": bool(_np.allclose(to, oo, rtol=rtol, atol=atol))})
            report["numeric_compare"] = diffs
        except Exception as e:
            report["numeric_compare"] = f"compare failed: {e}"
    return report


def cli_main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate an ONNX file (checker + ORT smoke).")
    ap.add_argument("--model", required=True, help="Path to .onnx")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args(argv)
    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    import numpy as np
    feed = {}
    for i in sess.get_inputs():
        # build zero/ones dummy of declared shape (dynamic dims -> seq-len/batch)
        shape = []
        for idx, d in enumerate(i.shape):
            if isinstance(d, int):
                shape.append(d)
            else:
                shape.append(args.batch if idx == 0 else args.seq_len)
        dtype = np.int64 if "int" in i.type else np.float32
        feed[i.name] = np.ones(shape, dtype=dtype)
    rep = validate_onnx(args.model, feed)
    print(json.dumps(rep))
    return 0 if rep.get("ort_run") == "ok" else 1


if __name__ == "__main__":
    sys.exit(cli_main())
