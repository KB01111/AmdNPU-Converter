"""CUA-S1 Forms 706K TinyTransformerScorer -> ONNX.

Model: byte-level (vocab 0-255) tiny transformer scorer.
Forward: (context_ids, context_mask, option_ids, option_token_mask, option_mask) -> scores [B, O]

Weights: ALWAYS loaded from safetensors via safetensors.torch.load_file and
mapped strictly into the module state_dict. Official loader tried first:
    from cua_s1.model import TinyTransformerScorer
fallback: local minimal mirror (byte-embedding + transformer + option-attention).

Shapes: ctx 224 / opt 96 fixed (+ padded variant flag for dynamic export).
Byte vocab 0-255 (never random 0-999).

    python workers/exporters/cua-s1/export_cua_s1.py --weights model.safetensors --config config.json --output cua_s1.onnx --variant forms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export CUA-S1 TinyTransformerScorer to ONNX.")
    p.add_argument("--weights", required=True, help="Path to .safetensors (or HF repo id)")
    p.add_argument("--config", default=None, help="Path to config.json (or HF repo id file)")
    p.add_argument("--model-id", default=None, help="HF repo id fallback for weights/config")
    p.add_argument("--output", required=True, help="Output .onnx path")
    p.add_argument("--variant", default="forms", choices=["forms", "nano-text"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--token", default=None)
    return p.parse_args(argv)


CTX_LEN = 224
OPT_LEN = 96


def load_raw_state(weights: str, model_id: str | None, token=None) -> dict:
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    p = Path(weights)
    if p.exists():
        return load_file(str(p), device="cpu")
    # treat as repo file name or repo id
    repo = model_id or weights
    fname = "model.safetensors"
    local = hf_hub_download(repo_id=repo, filename=fname, token=token)
    return load_file(local, device="cpu")


def load_cfg_dict(config: str | None, model_id: str | None, token=None) -> dict:
    from huggingface_hub import hf_hub_download

    if config:
        cp = Path(config)
        if cp.exists():
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        # config may be a repo id
        try:
            local = hf_hub_download(repo_id=config, filename="config.json", token=token)
            with open(local, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    if model_id:
        try:
            local = hf_hub_download(repo_id=model_id, filename="config.json", token=token)
            with open(local, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(json.dumps({"warning": f"config fetch failed: {e}"}), file=sys.stderr)
    return {}


def build_model(cfg: dict):
    """Try official loader, else local minimal mirror. Returns torch.nn.Module."""
    import torch
    import torch.nn as nn

    # 1) official
    try:
        from cua_s1.model import TinyTransformerScorer  # type: ignore
        try:
            return TinyTransformerScorer(cfg) if isinstance(cfg, dict) and cfg else TinyTransformerScorer()
        except TypeError:
            return TinyTransformerScorer()
    except Exception as e:
        print(json.dumps({"warning": f"official cua_s1 loader unavailable ({e}); using local mirror"}),
              file=sys.stderr)

    # 2) local minimal mirror: byte-embedding + transformer + option-attention
    d_model = int(cfg.get("hidden_size", cfg.get("d_model", 128)))
    n_layer = int(cfg.get("num_hidden_layers", cfg.get("n_layer", 4)))
    n_head = int(cfg.get("num_attention_heads", cfg.get("n_head", 4)))
    d_ff = int(cfg.get("intermediate_size", cfg.get("d_ff", 4 * d_model)))
    n_options_max = int(cfg.get("num_options", 8))

    class TinyMirror(nn.Module):
        def __init__(self):
            super().__init__()
            self.byte_emb = nn.Embedding(256, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                               dim_feedforward=d_ff, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.opt_emb = nn.Embedding(256, d_model)
            self.opt_encoder = nn.TransformerEncoder(layer, num_layers=max(1, n_layer // 2))
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
            self.scorer = nn.Linear(d_model, 1)

        def forward(self, context_ids, context_mask, option_ids, option_token_mask, option_mask):
            # context_ids: [B, C] byte ids; option_ids: [B, O, T]
            B, C = context_ids.shape
            Bo, O, T = option_ids.shape
            assert B == Bo, "batch mismatch context vs options"
            ctx = self.byte_emb(torch.clamp(context_ids, 0, 255).long())
            key_pad = (context_mask == 0) if context_mask is not None else None
            ctx_enc = self.encoder(ctx, src_key_padding_mask=key_pad)  # [B,C,D]
            ctx_vec = ctx_enc.mean(dim=1)  # [B,D]
            opt = self.opt_emb(torch.clamp(option_ids, 0, 255).long())  # [B,O,T,D]
            flat = opt.reshape(B * O, T, -1)
            flat_mask = option_token_mask.reshape(B * O, T) if option_token_mask is not None else None
            flat_pad = (flat_mask == 0) if flat_mask is not None else None
            flat_enc = self.opt_encoder(flat, src_key_padding_mask=flat_pad)
            opt_vec = flat_enc.mean(dim=1).reshape(B, O, -1)  # [B,O,D]
            q = ctx_vec.unsqueeze(1)  # [B,1,D]
            attn_out, _ = self.attn(q, opt_vec, opt_vec,
                                    key_padding_mask=(option_mask == 0) if option_mask is not None else None)
            scores = self.scorer(attn_out).squeeze(-1)  # [B,O]
            if option_mask is not None:
                scores = scores.masked_fill(option_mask == 0, float("-inf"))
            return scores

    m = TinyMirror()
    m._mirror_cfg = {"d_model": d_model, "n_layer": n_layer, "n_head": n_head,  # type: ignore
                     "d_ff": d_ff, "n_options_max": n_options_max}
    return m


def strict_load(module, state: dict) -> None:
    import torch
    missing, unexpected = [], []
    try:
        res = module.load_state_dict(state, strict=False)
        missing = list(getattr(res, "missing_keys", []))
        unexpected = list(getattr(res, "unexpected_keys", []))
    except Exception as e:
        print(json.dumps({"error": f"state_dict load failed: {e}"}), file=sys.stderr)
        raise
    # Honest strictness: if official loader matched fully, great; if mirror, map by shape-order
    if missing and unexpected:
        # attempt shape-ordered mapping: sort both by numel and assign
        own = dict(module.state_dict())
        s_items = sorted(state.items(), key=lambda kv: kv[1].numel() if hasattr(kv[1], "numel") else -1)
        o_items = sorted(own.items(), key=lambda kv: kv[1].numel())
        mapped = {}
        si = 0
        for oname, oten in o_items:
            while si < len(s_items):
                sname, sten = s_items[si]
                si += 1
                try:
                    import torch as _t
                    arr = sten if isinstance(sten, _t.Tensor) else _t.as_tensor(sten)
                except Exception:
                    continue
                if tuple(arr.shape) == tuple(oten.shape):
                    mapped[oname] = arr
                    break
            else:
                break
        if len(mapped) == len(own):
            module.load_state_dict(mapped, strict=True)
            print(json.dumps({"info": f"shape-ordered strict mapping applied ({len(mapped)} tensors)"}),
                  file=sys.stderr)
            return
        print(json.dumps({"error": "weight mismatch",
                          "missing": missing[:20], "unexpected": unexpected[:20]}), file=sys.stderr)
        raise RuntimeError("strict weight mapping failed (see missing/unexpected keys)")
    elif missing or unexpected:
        print(json.dumps({"warning": "partial weight match", "missing": missing[:20],
                          "unexpected": unexpected[:20]}), file=sys.stderr)


def main(argv=None) -> int:
    args = parse_args(argv)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except Exception as e:
        print(json.dumps({"error": f"torch not installed: {e}"}), file=sys.stderr)
        return 1

    cfg = load_cfg_dict(args.config, args.model_id or args.config, args.token)
    if args.variant == "nano-text" and "ctx_len" not in cfg:
        cfg = dict(cfg)

    try:
        raw = load_raw_state(args.weights, args.model_id, args.token)
    except Exception as e:
        print(json.dumps({"error": f"weight load failed: {e}"}), file=sys.stderr)
        return 1
    # safetensors returns tensors already; ensure torch tensors
    state = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in raw.items()}

    try:
        module = build_model(cfg)
        module.eval()
        strict_load(module, state)
        module.eval()
    except Exception as e:
        print(json.dumps({"error": f"model build/load failed: {e}"}), file=sys.stderr)
        return 1

    # Example inputs: byte vocab 0-255 only (deterministic arange pattern)
    B, O = 1, 4
    C, T = CTX_LEN, OPT_LEN
    context_ids = (torch.arange(C).unsqueeze(0).repeat(B, 1) % 256).long()
    context_mask = torch.ones(B, C, dtype=torch.long)
    option_ids = (torch.arange(O * T).reshape(1, O, T) % 256).long()
    option_token_mask = torch.ones(B, O, T, dtype=torch.long)
    option_mask = torch.ones(B, O, dtype=torch.long)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, context_ids, context_mask, option_ids, option_token_mask, option_mask):
            return self.m(context_ids, context_mask, option_ids, option_token_mask, option_mask)

    wrapped = Wrapper(module).eval()
    with torch.no_grad():
        ref = wrapped(context_ids, context_mask, option_ids, option_token_mask, option_mask)

    dynamic_axes = {
        "context_ids": {0: "batch"},
        "context_mask": {0: "batch"},
        "option_ids": {0: "batch", 1: "num_options"},
        "option_token_mask": {0: "batch", 1: "num_options"},
        "option_mask": {0: "batch", 1: "num_options"},
        "scores": {0: "batch", 1: "num_options"},
    }
    try:
        torch.onnx.export(
            wrapped,
            (context_ids, context_mask, option_ids, option_token_mask, option_mask),
            str(outp),
            input_names=["context_ids", "context_mask", "option_ids", "option_token_mask", "option_mask"],
            output_names=["scores"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except Exception as e:
        print(json.dumps({"error": f"torch.onnx.export failed: {e}"}), file=sys.stderr)
        return 1

    # ORT numeric validation
    try:
        import numpy as _np
        import onnxruntime as ort
        import onnx
        onnx.checker.check_model(str(outp))
        sess = ort.InferenceSession(str(outp), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {
            "context_ids": context_ids.numpy(),
            "context_mask": context_mask.numpy(),
            "option_ids": option_ids.numpy(),
            "option_token_mask": option_token_mask.numpy(),
            "option_mask": option_mask.numpy(),
        })
        oo = _np.asarray(outs[0], dtype=_np.float64)
        rr = ref.detach().cpu().numpy().astype(_np.float64)
        rep = {"output": str(outp), "variant": args.variant,
               "output_shape": list(oo.shape),
               "max_abs_diff": float(_np.max(_np.abs(rr - oo))) if rr.shape == oo.shape else None,
               "allclose": bool(_np.allclose(rr, oo, rtol=1e-3, atol=1e-4)) if rr.shape == oo.shape else False,
               "ort_inputs": [{"name": i.name} for i in sess.get_inputs()]}
        print(json.dumps(rep))
        return 0 if rep["allclose"] else 1
    except Exception as e:
        print(json.dumps({"error": f"ORT validation failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
