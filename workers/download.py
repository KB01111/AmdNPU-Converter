"""Download HF data and safetensors only. Never fetch pickle weights or model code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SAFE_FILES = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
              "sentencepiece.bpe.model", "preprocessor_config.json", "processor_config.json",
              "image_processor_config.json"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    try:
        if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
            raise ValueError("snapshot output must be a new or empty directory")
        from huggingface_hub import snapshot_download
        output.mkdir(parents=True, exist_ok=True)
        pairs = {"cua-ai/cua-s1-forms": ("cua-s1-forms.safetensors", "cua-s1-forms.json"),
                 "cua-ai/cua-s1-nano-0.1": ("text/model.safetensors", "text/config.json")}
        pair = pairs.get(args.model)
        snapshot_download(repo_id=args.model, local_dir=str(output), allow_patterns=list(pair) if pair else SAFE_FILES,
                          ignore_patterns=["*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.py"])
        if pair:
            # Official CUA checkpoints use paired JSON data, not Transformers config names.
            for old, new in zip(pair, ("model.safetensors", "config.json")):
                (output / old).rename(output / new)
        files = [p for p in output.rglob("*") if p.is_file() and ".cache" not in p.relative_to(output).parts]
        if any(not p.resolve().is_relative_to(output.resolve()) for p in files):
            raise ValueError("snapshot contains an external file link")
        if not (output / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("repository requires config.json and safetensors weights; pickle weights are unsupported")
        print(json.dumps({"model_id": args.model, "output": str(output.resolve()),
                          "files": sorted(str(p.relative_to(output)) for p in files)}))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"safe snapshot download failed: {e}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
