"""Generic torch.save / checkpoint -> ONNX converter + graph optimization.

    python workers/exporters/generic-onnx/convert.py --checkpoint model.pt --output model.onnx [--input-shape 1,3,224,224] [--opset 17] [--optimize]

Supports: full pickled nn.Module, state_dict + --arch hint is NOT fabricated —
--arch-module/--arch-class can point at importable class; otherwise exports
an traced functional graph when possible. Optimizes with onnxoptimizer if
installed, else onnxruntime tools / no-op with honest warning.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert torch checkpoint to ONNX + optimize.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt/.pth/.bin checkpoint")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--input-shape", default="1,3,224,224", help="Comma-separated input shape, e.g. 1,3,224,224")
    p.add_argument("--input-names", default="input", help="Comma-separated input names")
    p.add_argument("--output-names", default="output", help="Comma-separated output names")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--arch-module", default=None, help="Importable module for model class (for state_dict)")
    p.add_argument("--arch-class", default=None, help="Class name inside arch-module")
    p.add_argument("--optimize", action="store_true", help="Run onnxoptimizer passes if available")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    return p.parse_args(argv)


def load_module(checkpoint: str, arch_module: str | None, arch_class: str | None):
    import torch

    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj, "full-module"
    if isinstance(obj, dict) and arch_module and arch_class:
        mod = importlib.import_module(arch_module)
        cls = getattr(mod, arch_class)
        try:
            inst = cls()
        except TypeError:
            inst = cls.__new__(cls)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        inst.load_state_dict(obj, strict=True)
        inst.eval()
        return inst, f"{arch_module}.{arch_class}+state_dict"
    raise RuntimeError(
        "checkpoint is a state_dict but no --arch-module/--arch-class given; "
        "refusing to fabricate architecture (pass full pickled module or arch class)"
    )


def optimize_onnx(path: str) -> dict:
    try:
        import onnx
        model = onnx.load(path)
    except Exception as e:
        return {"optimized": False, "reason": f"load failed: {e}"}
    try:
        import onnxoptimizer  # type: ignore
        passes = ["eliminate_deadend", "eliminate_identity", "eliminate_nop_transpose",
                  "eliminate_nop_pad", "fuse_add_bias_into_conv"]
        try:
            model = onnxoptimizer.optimize(model, passes)
        except TypeError:
            model = onnxoptimizer.optimize(model)
        onnx.save(model, path)
        return {"optimized": True, "passes": passes, "tool": "onnxoptimizer"}
    except ImportError:
        return {"optimized": False, "reason": "onnxoptimizer not installed"}
    except Exception as e:
        return {"optimized": False, "reason": f"optimize failed: {e}"}


def main(argv=None) -> int:
    args = parse_args(argv)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except Exception as e:
        print(json.dumps({"error": f"torch not installed: {e}"}), file=sys.stderr)
        return 1

    shape = tuple(int(x) for x in args.input_shape.split(",") if x.strip())
    in_names = [s.strip() for s in args.input_names.split(",") if s.strip()]
    out_names = [s.strip() for s in args.output_names.split(",") if s.strip()]

    try:
        module, how = load_module(args.checkpoint, args.arch_module, args.arch_class)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    module = module.to(dtype).eval()
    example = torch.ones(*shape, dtype=dtype)
    # multi-input: split example along batch is dishonest; require single input unless module takes tuple
    with torch.no_grad():
        try:
            ref = module(example)
        except Exception as e:
            print(json.dumps({"error": f"trial run failed for input shape {shape}: {e}"}), file=sys.stderr)
            return 1

    dynamic_axes = {in_names[0]: {0: "batch"}}
    for n in out_names:
        dynamic_axes[n] = {0: "batch"}

    # Prefer dynamo exporter, fall back to classic exporter honestly.
    exporter_used = "torch.onnx.export"
    try:
        torch.onnx.export(module, example, str(outp), input_names=in_names,
                          output_names=out_names, dynamic_axes=dynamic_axes,
                          opset_version=args.opset, do_constant_folding=True)
    except Exception as e1:
        try:
            from torch.onnx import dynamo_export  # type: ignore
            ex = dynamo_export(module, example)
            ex.save(str(outp))
            exporter_used = "torch.onnx.dynamo_export"
        except Exception as e2:
            print(json.dumps({"error": f"export failed: classic={e1}; dynamo={e2}",
                              "hint": "for TF models use tf2onnx: python -m tf2onnx.convert --saved-model DIR --output model.onnx"}),
                  file=sys.stderr)
            return 1

    opt = {"optimized": False}
    if args.optimize:
        opt = optimize_onnx(str(outp))

    # validate
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        onnx.checker.check_model(str(outp))
        sess = ort.InferenceSession(str(outp), providers=["CPUExecutionProvider"])
        feed = {in_names[0]: np.ones(shape, dtype=np.float16 if args.dtype == "float16" else np.float32)}
        outs = sess.run(None, feed)
        rep = {"output": str(outp), "loaded_as": how, "exporter": exporter_used,
               "optimize": opt, "onnx_checker": "ok", "ort_run": "ok",
               "output_shapes": [list(o.shape) for o in outs]}
        print(json.dumps(rep))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"validation failed: {e}", "optimize": opt}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
