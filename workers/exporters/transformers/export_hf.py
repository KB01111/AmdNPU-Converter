"""Export supported Transformers text and image models using safetensors only."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validate_onnx


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True, help="HF repo id or local safe snapshot")
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--token", default=None)
    p.add_argument("--task", default="auto", choices=["auto", "sequence-classification",
                   "token-classification", "feature-extraction", "image-classification"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outp = Path(args.output)
    if args.batch < 1 or args.seq_len < 1 or args.opset < 1 or outp.exists():
        print(json.dumps({"error": "positive batch/seq-len/opset and a new output path required"}), file=sys.stderr)
        return 2
    try:
        import torch
        import transformers as tf
        source = {"token": args.token, "trust_remote_code": False}
        if Path(args.model_id).is_dir():
            source["local_files_only"] = True
        cfg = tf.AutoConfig.from_pretrained(args.model_id, **source)
        architecture = str((getattr(cfg, "architectures", None) or [""])[0]).lower()
        task = args.task
        if task == "auto":
            if "imageclassification" in architecture:
                task = "image-classification"
            elif "sequenceclassification" in architecture:
                task = "sequence-classification"
            elif "tokenclassification" in architecture:
                task = "token-classification"
            else:
                task = "feature-extraction"
        loaders = {"sequence-classification": tf.AutoModelForSequenceClassification,
                   "token-classification": tf.AutoModelForTokenClassification,
                   "feature-extraction": tf.AutoModel,
                   "image-classification": tf.AutoModelForImageClassification}
        model = loaders[task].from_pretrained(args.model_id, use_safetensors=True, **source).eval()
        if task == "image-classification":
            processor = tf.AutoImageProcessor.from_pretrained(args.model_id, use_fast=False, **source)
            size = getattr(cfg, "image_size", None) or getattr(processor, "crop_size", None) or processor.size
            if isinstance(size, int):
                height = width = size
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                height, width = size
            else:
                height = size.get("height", size.get("shortest_edge", 224))
                width = size.get("width", size.get("shortest_edge", 224))
            image = torch.zeros(getattr(cfg, "num_channels", 3), height, width, dtype=torch.float32)
            encoded = processor(images=[image] * args.batch, return_tensors="pt")
            inputs = {"pixel_values": encoded["pixel_values"]}
            dynamic_axes = {"pixel_values": {0: "batch"}}
        else:
            tokenizer = tf.AutoTokenizer.from_pretrained(args.model_id, **source)
            encoded = tokenizer(["hello world"] * args.batch, padding="max_length", truncation=True,
                                max_length=args.seq_len, return_tensors="pt")
            inputs = {name: value for name, value in encoded.items()
                      if name in ("input_ids", "attention_mask", "token_type_ids")}
            if "input_ids" not in inputs:
                raise ValueError("text tokenizer did not produce input_ids")
            dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in inputs}
        input_names = list(inputs)

        class Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *values):
                result = self.inner(**dict(zip(input_names, values)))
                logits = getattr(result, "logits", None)
                if logits is not None:
                    return logits
                hidden = getattr(result, "last_hidden_state", None)
                return hidden if hidden is not None else result[0]

        wrapped = Wrap(model).eval()
        example = tuple(inputs.values())
        with torch.no_grad():
            reference = wrapped(*example)
        output_name = "output" if task == "feature-extraction" else "logits"
        dynamic_axes[output_name] = {0: "batch"}
        if task in ("feature-extraction", "token-classification"):
            dynamic_axes[output_name][1] = "sequence"
        outp.parent.mkdir(parents=True, exist_ok=True)
        # Explicit legacy exporter retains the tested dynamic_axes contract on newer torch.
        options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(wrapped, example, str(outp), input_names=input_names,
                          output_names=[output_name], dynamic_axes=dynamic_axes,
                          opset_version=args.opset, do_constant_folding=True, **options)
        report = validate_onnx(str(outp), inputs, torch_outputs=reference)
        report.update(model_id=args.model_id, output=str(outp), opset=args.opset, task=task)
        comparison = report.get("numeric_compare")
        passed = (report.get("ort_run") == "ok" and isinstance(comparison, list) and
                  bool(comparison) and all(item.get("allclose") is True for item in comparison))
        report["validation_passed"] = passed
        print(json.dumps(report, allow_nan=False))
        return 0 if passed else 1
    except Exception as e:
        print(json.dumps({"error": f"Transformers export failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
