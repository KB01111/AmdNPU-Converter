"""Generic Hugging Face -> ONNX exporter (opset 17, dynamic batch/sequence).

    python workers/exporters/transformers/export_hf.py --model-id <org/name> --output model.onnx [--seq-len 128] [--opset 17]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import validate_onnx  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export any HF transformers model to ONNX.")
    p.add_argument("--model-id", required=True, help="HF repo id or local dir")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--token", default=None)
    p.add_argument("--task", default="auto", help="auto|sequence-classification|token-classification|feature-extraction")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification,
                                  AutoModelForTokenClassification, AutoTokenizer)
    except Exception as e:
        print(json.dumps({"error": f"transformers/torch not installed: {e}"}), file=sys.stderr)
        return 1

    try:
        cfg = AutoConfig.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
    except Exception as e:
        print(json.dumps({"error": f"load config failed: {e}"}), file=sys.stderr)
        return 1

    task = args.task
    try:
        if task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
        elif task == "token-classification":
            model = AutoModelForTokenClassification.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
        else:
            # auto: prefer classifier heads when arch suggests, else base model
            arch = str((getattr(cfg, "architectures", None) or [""])[0]).lower()
            if "sequenceclassification" in arch:
                model = AutoModelForSequenceClassification.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
            elif "tokenclassification" in arch:
                model = AutoModelForTokenClassification.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
            else:
                model = AutoModel.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
        model.eval()
    except Exception as e:
        print(json.dumps({"error": f"load model failed: {e}"}), file=sys.stderr)
        return 1

    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, token=args.token, trust_remote_code=False)
        enc = tok(["hello world"] * args.batch, padding="max_length", truncation=True,
                  max_length=args.seq_len, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    except Exception:
        input_ids = torch.ones(args.batch, args.seq_len, dtype=torch.long)
        attention_mask = torch.ones(args.batch, args.seq_len, dtype=torch.long)

    dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"}}
    example = (input_ids, attention_mask)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            logits = getattr(out, "logits", None)
            if logits is None:
                hs = getattr(out, "last_hidden_state", None)
                return hs if hs is not None else out[0]
            return logits

    wrapped = Wrap(model)
    wrapped.eval()
    with torch.no_grad():
        ref = wrapped(input_ids, attention_mask)

    # output names
    out_names = ["logits"] if getattr(ref, "dim", lambda: 0)() >= 2 else ["output"]
    dynamic_axes[out_names[0]] = {0: "batch"}

    try:
        torch.onnx.export(
            wrapped, example, str(outp),
            input_names=["input_ids", "attention_mask"],
            output_names=out_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except Exception as e:
        print(json.dumps({"error": f"torch.onnx.export failed: {e}"}), file=sys.stderr)
        return 1

    rep = validate_onnx(str(outp),
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                        torch_outputs=ref)
    rep.update({"model_id": args.model_id, "output": str(outp), "opset": args.opset})
    print(json.dumps(rep))
    ok = rep.get("ort_run") == "ok"
    cmp = rep.get("numeric_compare")
    if isinstance(cmp, list) and cmp and isinstance(cmp[0], dict) and cmp[0].get("allclose") is False:
        print(json.dumps({"warning": "numeric mismatch vs torch (see numeric_compare)"}), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
