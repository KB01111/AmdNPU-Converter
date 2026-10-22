# Conversion recipes

Recipes are versioned JSON data loaded through the conversion settings UI.
The app reads `conversion_request`, source metadata, and calibration/gate
settings. It never evaluates the `commands` strings or accepts remote code.
Choose local calibration/evaluation/output paths before starting a conversion.

| Recipe | Profile | Source |
|---|---|---|
| `amd-xdna/npu-default.json` | auto (BF16) | example ConvNext |
| `amd-xdna/cnn-a8w8.json` | A8W8 | example ConvNext |
| `amd-xdna/transformer-bf16.json` | BF16 | example BERT |
| `cua-s1/forms-a16w8.json` | A16W8 | CUA-S1 Forms; official local loader required |
| `cua-s1/nano-text-bf16.json` | BF16 | CUA-S1 nano-text; official local loader required |
| `generic/hf-default.json` | auto (BF16) | Hugging Face |
| `generic/onnx-import.json` | auto (BF16) | local ONNX |

`amd-xdna/llm-int4.json` is an unsupported reference recipe and is not offered
as a working preset. INT4 needs a validated LLM implementation; adding an enum
value cannot provide it. An arbitrary LLM is not supported merely because it
appears in a recipe.

Custom recipes use `recipe_version: "1.0"`, a nonempty `name`, and a
`conversion_request` matching `src/lib/types.ts`. The minimum shape is:

```json
{
  "recipe_version": "1.0",
  "name": "My model",
  "conversion_request": {
    "modelId": "owner/model",
    "url": "https://huggingface.co/owner/model",
    "sourceKind": "hf",
    "precision": "bf16",
    "device": "npu",
    "powerMode": "automatic",
    "targetEp": "vitisai",
    "evalNpz": "C:/datasets/eval.npz",
    "maxDegradation": 0.02
  }
}
```

For local ONNX set `sourceKind: "onnx"`, `onnxPath` to the source graph, and
an identifying `modelId`; HF URL can be empty. The app also understands shipped
recipes' `source.kind` and `source.onnx_path`. The evaluation NPZ must contain
every input array. Calibration is required for calibrated integer profiles.

Save/load uses native dialogs. Loading validates data and populates controls;
starting performs the additional checks against local files/dependencies.
Recipe saves do not execute a conversion or publish anything.

Historical command examples are informational. The current WinML interface is
`optimize --model INPUT --output OUTPUT` and
`compile --model INPUT --output model.vitisai.onnx --device npu --ep vitisai`.
The EPContext wrapper and sidecars must remain together. The worker's compare
gate measures numeric output drift, not the task accuracy/perplexity suggested
by some historical recipe annotations.
