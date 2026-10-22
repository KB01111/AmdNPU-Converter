# Recipes — Conversion Recipes

Alla filer i denna mapp ar **Conversion Recipes**: versionerade JSON-foreskrifter
vars `conversion_request` mappas 1:1 mot frontend-typen `ConversionRequest`
(`src/lib/types.ts`) och vars `commands`-block enbart anvander verkliga
worker-flaggor (se 建商 `workers/*/... --help`).

## Filoversikt

| Recept | Kategori | Precision | Modell (exempel) |
|---|---|---|---|
| `amd-xdna/npu-default.json` | XDNA default | `auto`, fallback `[bf16, a16w8, a8w8]` | `facebook/convnext-tiny-224` |
| `amd-xdna/cnn-a8w8.json` | CNN | `a8w8` (Quark statisk INT8) | `facebook/convnext-tiny-224` |
| `amd-xdna/transformer-bf16.json` | Transformer | `bf16` | `prajjwal1/bert-tiny` |
| `amd-xdna/llm-int4.json` | LLM | `int4` weight-only — kraver worker-stod (se nedan) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `cua-s1/forms-a16w8.json` | CUA-S1 forms (exporter `cua-s1`) | `a16w8` | `cua-ai/cua-s1-forms` |
| `cua-s1/nano-text-bf16.json` | CUA-S1 nano-text (exporter `cua-s1`) | `bf16` | `cua-ai/cua-s1-nano-text` |
| `generic/hf-default.json` | Generell HF-standard | `auto` | valfri HF-modell (se `tests/fixtures.md`) |
| `generic/onnx-import.json` | ONNX-import (`direct-onnx`) | `auto` | lokal `.onnx` i `downloads/` |

## ConversionRequest-falt (frontend-typ, `src/lib/types.ts`)

```ts
interface ConversionRequest {
  modelId: string;   // t.ex. "facebook/convnext-tiny-224"
  url: string;       // t.ex. "https://huggingface.co/facebook/convnext-tiny-224"
  precision: "auto" | "fp32" | "bf16" | "a16w8" | "a8w8" | "int8";
  device: "auto" | "npu" | "gpu" | "cpu";
  powerMode: "automatic" | "max-performance" | "battery-efficient";
  targetEp: "vitisai" | "qnn" | "cuda" | "cpu";
  outputDir?: string | null;
}
// Store-defaults (src/stores/useAppStore.ts): auto / auto / automatic / vitisai
```

Rust-sidan (`src-tauri/src/jobs/mod.rs`, `commands/mod.rs`) exponerar:
`hf_inspect`, `get_hardware_info`, `start_conversion({ req })`,
`get_job_status`, `list_jobs`, `cancel_job`, `run_benchmark`, `list_models`
samt eventet `job://progress` med stegen
`inspection, download, export, optimize, quantize, compile, validate, benchmark`.
Artefakter per steg: `model.onnx` -> `model.opt.onnx` ->
`model.{precision}.onnx` -> `model.{ep}.bin` (export hardkodar opset 17).

Relevanta worker-flaggor (enda tillatna i `commands`):

- Export HF: `exporters/transformers/export_hf.py --model-id --output --opset 17 --batch --seq-len [--task] [--token]` (trust_remote_code=false ar hardkodat)
- Export CUA-S1: `exporters/cua-s1/export_cua_s1.py --weights --config --model-id --output --variant forms|nano-text --opset 17 [--token]` (CTX 224 / OPT 96 ar konstanter)
- Kvantisering: `quark/quantize.py --input --output --profile bf16|a16w8|a8w8|int8 [--calibration-dir] [--inputs-json] [--max-samples] [--reference-metrics]`
- WinML: `winml/run.py <inspect|export|optimize|quantize|compile|build|perf|eval|config|sys> ...` (vidarebefordrar till `winml.exe`; Rust-bryggan bygger `export --model --out --opset`, `optimize --in --out`, `quantize --in --out --precision`, `compile --in --out --ep`, `build --model --out-dir --config`, `perf --model`, `eval --model --dataset`)
- Utvardering: `evaluation/compare.py --baseline --candidate --eval-npz --max-degradation` (grind, default 0.02), `evaluation/bench.py --model --devices cpu,npu,gpu --iterations --warmup` (NPU-validering: VitisAIExecutionProvider forst)

## Hur recept valjs automatiskt per arkitektur

Motsvarar `workers/hf-inspector/inspect.py::_pick_export_strategy` + precisionstabell:

1. **Lokal `.onnx`** (`source.kind == "onnx"`): `generic/onnx-import.json`
   (`direct-onnx`; torch-checkpoints `.pt/.pth/.bin` gar via
   `exporters/generic-onnx/convert.py --checkpoint --arch-module/--arch-class --optimize`).
2. **CUA-S1** (arch innehaller `cua`/`tinytransformer` eller modell-ID matchar
   `cua-ai/cua-s1-*`): `cua-s1/forms-a16w8.json` (forms) eller
   `cua-s1/nano-text-bf16.json` (nano-text, `--variant nano-text`).
3. **HF-modeller** (via `hf_inspect`-metadata):
   - CNN-familjer (`convnext`, `resnet`, `efficientnet`, ...) -> `amd-xdna/cnn-a8w8.json`
   - Transformer-encoders (`bert`, `roberta`, ...) -> `amd-xdna/transformer-bf16.json`
   - Dekoder-LLM:er (`llama`, `tinyllama`, ...) -> `amd-xdna/llm-int4.json` (se INT4-notis)
   - Okant/ovrigt (`.safetensors` -> `transformers`) -> `generic/hf-default.json`
4. Anvandaren kan alltid byta recept manuellt fore `start_conversion`.

## Cache-nyckel (enda sanningen: `src-tauri/src/winml/bridge.rs`)

```text
compute_cache_key(model_sha, config, quant, hw, ep, driver)
  = sha256_hex(model_sha + NUL + config + NUL + quant + NUL + hw + NUL + ep + NUL + driver)
```

- Aterskapa aldrig cache over ny Vitis AI EP- eller NPU-drivrutinsversion.
- EPContext (`ep.context_enable=1`, `ep.context_embed_mode=1`) ar deploymentsformatet;
  appens compile-steg emitterar `model.<ep>.bin` via `winml compile --ep vitisai`.

## INT4-notis (llm-int4.json)

AMD-rekommendationen for LLM pa NPU ar INT4 weight-only, men varken
`Precision`-unionen (`src/lib/types.ts`) eller `PROFILES` i
`workers/quark/quantize.py` stoder `int4` annu. Receptet anger `precision: "int4"`
och ar korbart sa fort tva enrads-tillagg gjorts (lagg till `"int4"` i unionen
respektive `PROFILES`). Tills dess: kor LLM-receptet med `a16w8` eller `int8`.
Gapet ar dokumenterat i receptet istallet for att mockas.

## Spara egna Conversion Recipes fran appen

1. Kor ett floode i appen (HF-standard, ONNX-import eller CUA-S1).
2. Justera `precision`/`device`/`powerMode`/`targetEp` i konverteringsvyn.
3. Klicka **"Save as Recipe"** (skapar `recipes/custom/<namn>.json` med
   `recipe_version: "1.0"` och ifylld `conversion_request` + bevarade `commands`).
4. Verifiera: ladda receptet via **"Load Recipe"** och kor checklistan i
   `tests/e2e-conversion.md` (framgang = exporterad + validerad +
   kompilerad + verifierad NPU-korning).
5. Committa receptfilen — den ar korbar direkt via sitt `commands`-block.
