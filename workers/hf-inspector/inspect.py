"""HF model inspector — strict JSON to stdout.

Usage:
    python workers/hf-inspector/inspect.py --model-id <org/name> [--json]
    python workers/hf-inspector/inspect.py --url https://huggingface.co/<org/name>

Exit codes:
    0 success, 1 network/API error, 2 usage error, 3 auth/permission error.
"""
from __future__ import annotations

# NOTE: this file is named inspect.py and lives in its own directory, which
# Python prepends to sys.path when run as a script. That shadows the stdlib
# `inspect` module (argparse -> _colorize -> import inspect). Strip our own
# directory from sys.path BEFORE importing anything that may pull in stdlib
# `inspect`, using only builtins (sys/os) which do not import it.
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path = [p for p in _sys.path if _os.path.abspath(p or ".") != _HERE]

import argparse
import json
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Inspect a Hugging Face model repo (no weight download).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-id", dest="model_id", default=None, help="HF model id, e.g. org/model")
    g.add_argument("--url", dest="url", default=None, help="HF model URL, e.g. https://huggingface.co/org/model")
    p.add_argument("--json", dest="as_json", action="store_true", help="Emit strict JSON (default on).")
    p.add_argument("--revision", default="main", help="Git revision (default: main).")
    p.add_argument("--token", default=None, help="HF token (or HF_TOKEN env).")
    return p.parse_args(argv)


def model_id_from_url(url: str) -> str:
    url = url.strip().rstrip("/")
    # accept https://huggingface.co/org/name[/...] and hf://org/name
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "hf://"):
        if url.startswith(prefix):
            rest = url[len(prefix):]
            parts = [x for x in rest.split("/") if x]
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    raise ValueError(f"Cannot parse model id from url: {url}")


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _param_count_from_safetensors_header(header: dict) -> int:
    total = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and "shape" in v:
            n = 1
            try:
                for d in v["shape"]:
                    n *= int(d)
            except Exception:
                continue
            total += n
    return total


def _estimate_bytes(files_info) -> int:
    total = 0
    for f in files_info or []:
        sz = getattr(f, "size", None)
        if isinstance(sz, int):
            total += sz
    return total


def _pick_export_strategy(arch: str, siblings: list[str]) -> str:
    names = " ".join(siblings).lower()
    a = (arch or "").lower()
    if "cua" in a or "tinytransformer" in a:
        return "cua-s1"
    if any(t in names for t in ("model.safetensors", ".safetensors")):
        return "transformers"
    if names.endswith(".onnx") or ".onnx" in names:
        return "direct-onnx"
    if any(t in names for t in ("pytorch_model.bin", ".pt", ".pth", ".ckpt")):
        return "generic-onnx"
    if "tflite" in names:
        return "tflite-to-onnx"
    return "transformers"


def _inputs_hint(arch: str, config: dict) -> dict:
    arch_l = (arch or "").lower()
    if "cua" in arch_l or "tinytransformer" in arch_l:
        return {
            "context_ids": "[batch, 224] int64 bytes 0-255",
            "context_mask": "[batch, 224] int64",
            "option_ids": "[batch, num_options, 96] int64 bytes 0-255",
            "option_token_mask": "[batch, num_options, 96] int64",
            "option_mask": "[batch, num_options] int64",
        }
    seq_len = config.get("max_position_embeddings") or config.get("n_positions") or 512
    return {
        "input_ids": f"[batch, sequence] int64 (sequence~{seq_len})",
        "attention_mask": "[batch, sequence] int64",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        model_id = args.model_id or model_id_from_url(args.url)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except Exception as e:  # pragma: no cover
        print(json.dumps({"error": f"huggingface_hub not installed: {e}"}), file=sys.stderr)
        return 1

    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id=model_id, revision=args.revision)
    except RepositoryNotFoundError:
        print(json.dumps({"error": f"model not found: {model_id}"}), file=sys.stderr)
        return 1
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "unauthorized" in msg.lower():
            print(json.dumps({"error": f"auth/permission error: {e}"}), file=sys.stderr)
            return 3
        print(json.dumps({"error": f"model_info failed: {e}"}), file=sys.stderr)
        return 1

    try:
        siblings = api.list_repo_files(repo_id=model_id, revision=args.revision)
    except Exception:
        siblings = list(getattr(info, "siblings", []) or [])
        # siblings may be objects with rfilename
        flat = []
        for s in siblings:
            flat.append(getattr(s, "rfilename", str(s)))
        siblings = flat

    license_id = None
    try:
        card = getattr(info, "cardData", None)
        if isinstance(card, dict):
            lic = card.get("license")
            license_id = lic if isinstance(lic, str) else (lic.get("id") if isinstance(lic, dict) else None)
        tags = getattr(info, "tags", []) or []
        if license_id is None:
            for t in tags:
                if isinstance(t, str) and t.startswith("license:"):
                    license_id = t.split(":", 1)[1]
                    break
    except Exception:
        pass

    architecture = None
    parameter_count = 0
    fmt = "unknown"
    config: dict = {}

    # 1) try config.json (cheap, no weights)
    if "config.json" in siblings:
        try:
            cfg_path = hf_hub_download(repo_id=model_id, filename="config.json",
                                       revision=args.revision, token=token)
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            archs = config.get("architectures")
            if isinstance(archs, list) and archs:
                architecture = str(archs[0])
            elif isinstance(config.get("model_type"), str):
                architecture = str(config.get("model_type"))
        except Exception as e:
            print(json.dumps({"warning": f"config.json fetch failed: {e}"}), file=sys.stderr)

    # 2) safetensors index (param count without weights)
    lower = [s.lower() for s in siblings]
    has_safetensors = any(s.endswith(".safetensors") for s in lower)
    has_onnx = any(s.endswith(".onnx") for s in lower)
    has_bin = any(s.endswith(".bin") or s.endswith(".pt") or s.endswith(".pth") for s in lower)
    if has_safetensors:
        fmt = "safetensors"
    elif has_onnx:
        fmt = "onnx"
    elif has_bin:
        fmt = "pytorch"
    else:
        fmt = "unknown"

    index_file = None
    for cand in ("model.safetensors.index.json", "pytorch_model.safetensors.index.json"):
        if cand in siblings:
            index_file = cand
            break
    if index_file:
        try:
            idx_path = hf_hub_download(repo_id=model_id, filename=index_file,
                                       revision=args.revision, token=token)
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            parameter_count = _safe_int((idx.get("metadata") or {}).get("total_size"), 0)
            # total_size in index is bytes, not params; try weight_map shapes? fall back below
            if parameter_count > 10_000_000_000:
                # looks like bytes; keep as bytes hint, reset params to 0 for header pass
                parameter_count = 0
        except Exception as e:
            print(json.dumps({"warning": f"safetensors index fetch failed: {e}"}), file=sys.stderr)

    # 3) single-shard safetensors header (first 8-byte LE length + JSON header)
    if parameter_count == 0 and has_safetensors:
        shard = next((s for s in siblings if s.lower().endswith(".safetensors")), None)
        if shard:
            try:
                import struct
                local = hf_hub_download(repo_id=model_id, filename=shard,
                                        revision=args.revision, token=token)
                with open(local, "rb") as f:
                    raw = f.read(8)
                    if len(raw) == 8:
                        (hlen,) = struct.unpack("<Q", raw)
                        if 0 < hlen < 100 * 1024 * 1024:
                            header = json.loads(f.read(hlen).decode("utf-8"))
                            parameter_count = _param_count_from_safetensors_header(header)
            except Exception as e:
                print(json.dumps({"warning": f"safetensors header read failed: {e}"}), file=sys.stderr)

    # 4) fallback: config-based estimate (num_parameters field or hidden math)
    if parameter_count == 0 and config:
        for key in ("num_parameters", "n_params", "total_params"):
            if isinstance(config.get(key), (int, float)) and config[key] > 0:
                parameter_count = int(config[key])
                break

    if not architecture:
        architecture = config.get("model_type") or getattr(info, "pipeline_tag", None) or "unknown"

    try:
        files_info = getattr(info, "siblings", None)
        estimated = _estimate_bytes(files_info)
    except Exception:
        estimated = 0

    out = {
        "model_id": model_id,
        "architecture": architecture,
        "parameter_count": int(parameter_count),
        "format": fmt,
        "license": license_id,
        "siblings": sorted(siblings),
        "estimated_bytes": int(estimated),
        "inputs_hint": _inputs_hint(str(architecture), config if isinstance(config, dict) else {}),
        "export_strategy": _pick_export_strategy(str(architecture), siblings),
    }
    # strict JSON to stdout (Rust parser reads stdout)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
