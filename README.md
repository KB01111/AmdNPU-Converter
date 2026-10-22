[![CI](https://github.com/KB01111/AmdNPU-Converter/actions/workflows/ci.yml/badge.svg)](https://github.com/KB01111/AmdNPU-Converter/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Tauri v2](https://img.shields.io/badge/Tauri-v2-blue.svg) ![Platform: Windows](https://img.shields.io/badge/platform-Windows-blue.svg)

# AMD NPU Model Converter

Svenska / English — desktop app (Tauri) som konverterar HF- och ONNX-modeller
till AMD XDNA NPU (Ryzen AI) via WinML/ONNX Runtime + Vitis AI Execution Provider.

## Vad den gor / What it does

Tre huvudfloden (alla slutar med exporterad + validerad + kompilerad +
verifierad NPU-korning):

1. **ONNX-import -> compile** — befintlig `.onnx` (`direct-onnx`) -> `winml
   optimize` -> Quark-kvantisering -> `winml compile --ep vitisai` ->
   `compare.py`-grind -> `bench.py --devices npu`.
2. **HF-standard -> build** — `start_conversion(ConversionRequest)` kor
   pipelinen `inspection, download, export, optimize, quantize, compile,
   validate, benchmark` (opset 17). Recept valjs per arkitektur
   (CNN -> `a8w8`, transformer -> `bf16`, LLM -> `int4` nar worker-stod finns,
   okand -> `auto`).
3. **CUA-S1 Forms -> custom export** — `cua-ai/cua-s1-forms` via
   `exporters/cua-s1/export_cua_s1.py --variant forms` (5 inputs, ctx 224 /
   opt 96, byte-vocab 0-255, endast safetensors) -> `a16w8` -> kompilering ->
   verifiering.

## Arkitektur (textdiagram)

```text
+------------------+  start_conversion(req)   +------------------+
|  Tauri UI (web)  | -----------------------> | Rust (src-tauri) |
|  receptval,      |  ConversionRequest       |  jobs::run_pipeline|
|  Save/Load       |  {modelId,url,precision,  |  STEPS: inspection |
|  Recipe          |   device,powerMode,      |  download export   |
|                  |   targetEp,outputDir}    |  optimize quantize |
+------------------+ <----------------------- |  compile validate |
                   |  job://progress          |  benchmark         |
                   |  get_job_status          +---------+----------+
                   |                          | winml::bridge      |
                   |                          | compute_cache_key  |
                   +------------+-------------+----------+---------+
                                |                        |
                    run_python_worker          run_winml -> winml.exe
                    workers/*.py               inspect|export|optimize|
                                |              quantize|compile|build|
              hf-inspector/     |              perf|eval|config|sys
              exporters/{transformers|         |
                cua-s1|generic-onnx|common}    |
              quark/quantize.py |              |
              evaluation/{compare,bench}.py     |
```

Recept: `recipes/` (versionerad `ConversionRequest`-JSON + verkliga
`commands`). Tester: `tests/`. Workers: `workers/`.

## Prerequisites

- Windows 11 + Ryzen AI-maskin med NPU + installerad NPU-drivrutin
  (Ryzen AI Software, inkl. Vitis AI Execution Provider / WinML CLI)
- Python 3.11 + [uv](https://docs.astral.sh/uv/)
- Node.js + npm, Rust (via rustup) for Tauri
- HF-atkomst for testmodeller (se `tests/fixtures.md`); `cua-ai/*` kan krava token

## Quickstart

```powershell
# 1. Miljo + beroenden (skapar workers/.venv, .venv-winml, .venv-quark)
powershell -ExecutionPolicy Bypass -File workers/bootstrap.ps1

# 2. Systemkontroll: NPU + Vitis AI EP synliga?
workers/.venv-winml/Scripts/python.exe workers/winml/run.py sys --output-json

# 3. Appen
npm install
npm run tauri dev      # utveckling
npm run tauri build    # produktion
```

Kor sedan ett flode i appen (`start_conversion`) eller direkt via ett recepts
`commands`-block, t.ex. `recipes/amd-xdna/cnn-a8w8.json`.

## Cache-nyckel (enda sanningen: `src-tauri/src/winml/bridge.rs`)

```text
compute_cache_key(model_sha, config, quant, hw, ep, driver)
  = sha256_hex(model_sha + NUL + config + NUL + quant + NUL + hw + NUL + ep + NUL + driver)
```

- Atteranvand aldrig cache over ny Vitis AI EP- eller NPU-drivrutinsversion.
- EPContext (`ep.context_enable=1`, `ep.context_embed_mode=1`) ar deploymentsformatet;
  appens compile-steg emitterar `model.<ep>.bin` via `winml compile --ep vitisai`.
- Opset 17 (export-steget hardkodar `--opset 17`; AMD rekommenderar opset 17).

## Sakerhet / Security

- `trust_remote_code=false` alltid (hardkodat i `export_hf.py`; cuba-s1-loadern
  anvander aldrig fjarrkod, endast safetensors via `safetensors.torch.load_file`).
- Endast safetensors-vikter; `generic-onnx/convert.py` vagar fabricera
  arkitektur (kraver `--arch-module/--arch-class` eller full picklad modul).
- `quantize.py` har ingen tyst fallback: saknas `amd-quark` blir det exit 2 med
  install-hint; `accuracy_delta` fejkas aldrig (kraver evaluation-pipelinen).
- `downloads/`, `artifacts/` och stora kalibreringsfiler ar git-ignorerade.

## Felsokning / Troubleshooting

| Problem | Atgard |
|---|---|
| `LayerNormalization`-fel / axis-mismatch efter export | **Apply Fix & Retry**: atgarda LayerNorm-axlarna (export med korrigerad graph), kor om export-steget, validera med onnx.checker + ORT-load, forsok kompileringen igen |
| Dynamiska axlar (batch/sequence) kvar efter export | Vantas fran exporterna; `optimize`/`compile` specialiserar fasta shapes (batch=1) fore NPU |
| Fel opset | Export hardkodar 17; aldre `.onnx` konverteras med ONNX Version Converter fore optimize |
| `quantize.py` exit 2 (quark saknas) | `bootstrap.ps1 -QuarkOnly` (miljon `workers/.venv-quark` + `requirements-quark.txt`) |
| `compare.py` over `--max-degradation` | Bygget stoppas (exit != 0): utoka representativ kalibrering eller ga upp ett steg i `bf16 -> a16w8 -> a8w8` |
| Langsam forsta korning | Vantas: NPU-kompilering sker vid forsta sessionen (minuter; LLM tiotals min, >= 32 GB RAM). Senare korningar ateranvander EPContext/cache |
| NPU hittas inte | `run.py sys --output-json`; uppdatera NPU-drivrutin/Ryzen AI Software; `bench.py --devices npu` far aldrig tyst falla tillbaka vid verifierade floden |

Se `recipes/README.md` (receptval, INT4-notis, spara egna recept),
`tests/e2e-conversion.md` (manuell checklista) och `tests/fixtures.md` (riktiga modeller).
