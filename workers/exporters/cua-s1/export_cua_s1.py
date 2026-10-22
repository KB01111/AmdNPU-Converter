"""Export local CUA Forms/tinyx or Nano text checkpoints using official safe loaders.

Requires the separately installed cua_s1 package from the trusted upstream source.
The official loaders verify safetensors signatures, config, and named tensor keys.
No pickle, remote model code, invented architecture, or shape-based remapping.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validate_onnx


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Local safetensors checkpoint")
    parser.add_argument("--config", help="Matching local checkpoint JSON")
    parser.add_argument("--model-id", help="Source provenance (download using download.py first)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="forms", choices=["forms", "nano-text"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--token", default=None)
    return parser.parse_args(argv)


def load_official_checkpoint(weights: str, config: str | None, variant: str):
    path = Path(weights).resolve()
    config_path = Path(config).resolve() if config else path.with_suffix(".json")
    if path.suffix != ".safetensors" or not path.is_file() or not config_path.is_file():
        raise ValueError("existing local safetensors weights and matching checkpoint JSON required")
    if config_path == path.with_suffix(".json"):
        location = path
    elif path.name == "model.safetensors" and config_path == path.parent / "config.json":
        location = path.parent
    else:
        raise ValueError("checkpoint pair must share a stem, or use model.safetensors plus config.json")
    try:
        if variant == "forms":
            from cua_s1.model import load_checkpoint
        elif variant == "nano-text":
            from cua_s1.nano import load_nano_checkpoint as load_checkpoint
        else:
            raise ValueError(f"unsupported CUA variant: {variant}")
    except ImportError as e:
        raise RuntimeError("install the trusted official cua_s1 source package in the base worker environment") from e
    model, _collator, cfg = load_checkpoint(location, device="cpu")
    return model, cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        if output.exists() or args.opset < 1:
            raise ValueError("a new output path and positive opset required")
        import torch
        model, cfg = load_official_checkpoint(args.weights, args.config, args.variant)
        model.eval()
        lengths = [cfg.get("context_tokens"), cfg.get("option_tokens")]
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths):
            raise ValueError("official checkpoint must declare positive context_tokens/option_tokens")
        context_length, option_length = lengths
        inputs = {
            "context_ids": torch.ones(1, context_length, dtype=torch.long),
            "context_mask": torch.ones(1, context_length, dtype=torch.bool),
            "option_ids": torch.ones(1, 4, option_length, dtype=torch.long),
            "option_token_mask": torch.ones(1, 4, option_length, dtype=torch.bool),
            "option_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        # CUA byte IDs are UTF-8 bytes + 1 (1..256); zero is reserved padding.
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, context_ids, context_mask, option_ids, option_token_mask, option_mask):
                batch = {"context_ids": context_ids, "option_ids": option_ids,
                         "option_token_mask": option_token_mask, "option_mask": option_mask}
                if args.variant == "nano-text":
                    batch.update(context_token_mask=context_mask, modality="text")
                else:
                    batch["context_mask"] = context_mask
                return self.inner(batch)

        wrapped = Wrapper(model).eval()
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            reference = wrapped(*inputs.values())
        axes = {name: {0: "batch"} for name in inputs}
        for name in ("option_ids", "option_token_mask", "option_mask"):
            axes[name][1] = "num_options"
        axes["scores"] = {0: "batch", 1: "num_options"}
        output.parent.mkdir(parents=True, exist_ok=True)
        options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(wrapped, tuple(inputs.values()), str(output), input_names=list(inputs),
                          output_names=["scores"], dynamic_axes=axes, opset_version=args.opset,
                          do_constant_folding=True, **options)
        report = validate_onnx(str(output), inputs, torch_outputs=reference)
        comparison = report.get("numeric_compare")
        passed = (report.get("ort_run") == "ok" and isinstance(comparison, list) and
                  bool(comparison) and all(item.get("allclose") is True for item in comparison))
        report.update(output=str(output), variant=args.variant, validation_passed=passed)
        print(json.dumps(report, allow_nan=False))
        return 0 if passed else 1
    except Exception as e:
        print(json.dumps({"error": f"CUA export failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
