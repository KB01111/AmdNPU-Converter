# Python Worker Engine — AMD NPU Model Converter

Real, runnable Python 3.11 workers (`uv`). No mocks: every script does real
I/O and honest exit codes. JSON goes to **stdout** (Rust parser); warnings to stderr.

## Environments

| env | file | use |
|---|---|---|
| base | `pyproject.toml` | `uv venv --python 3.11 .venv` + `uv pip install -e .` |
| winml (locked) | `requirements-winml.txt` | `uv pip install --python .venv-winml/Scripts/python.exe -r requirements-winml.txt` |
| quark (locked) | `requirements-quark.txt` | `uv pip install --python .venv-quark/Scripts/python.exe -r requirements-quark.txt` |

Bootstrap all: `powershell -ExecutionPolicy Bypass -File workers/bootstrap.ps1`

`winml.exe` lookup: `$env:WINML_BIN` → `workers/.venv-winml/Scripts/winml.exe` →
`workers/.venv/Scripts/winml.exe` → `PATH`.

## HF inspect

```powershell
.venv/Scripts/python.exe workers/hf-inspector/inspect.py --model-id <org/name> --json
.venv/Scripts/python.exe workers/hf-inspector/inspect.py --url https://huggingface.co/<org/name>
```

## WinML (pass-through)

```powershell
.venv-winml/Scripts/python.exe workers/winml/run.py sys --output-json
.venv-winml/Scripts/python.exe workers/winml/run.py inspect --model model.onnx
.venv-winml/Scripts/python.exe workers/winml/run.py build --model model.onnx --out out/ --job-id abc123
```

## Quark quantization profiles

| model category | profile | note |
|---|---|---|
| CNN (ResNet/MobileNet/EfficientNet) | `a8w8` | static quant, needs calibration `.npy/.npz` |
| Transformer (BERT/mini-LM) | `bf16` or `a16w8` | accuracy-safe default `bf16` |
| LLM (7B+) | `bf16` or `int4*` (`int8` here) | prefer BF16; INT4 via Quark LLM path |

```powershell
.venv-quark/Scripts/python.exe workers/quark/quantize.py --input model.onnx --output model-a8w8.onnx --profile a8w8 --calibration-dir calib/
```

Missing quark → exit 2 (no fake ORT fallback).

## Export / Eval

```powershell
.venv/Scripts/python.exe workers/exporters/transformers/export_hf.py --model-id <org/name> --output model.onnx
.venv/Scripts/python.exe workers/exporters/cua-s1/export_cua_s1.py --weights m.safetensors --config config.json --output cua.onnx --variant forms
.venv/Scripts/python.exe workers/exporters/generic-onnx/convert.py --checkpoint m.pt --output m.onnx --input-shape 1,3,224,224 --optimize
.venv/Scripts/python.exe workers/evaluation/bench.py --model model.onnx --devices cpu,npu --iterations 100 --warmup 10
.venv/Scripts/python.exe workers/evaluation/compare.py --baseline base.onnx --candidate quant.onnx --eval-npz eval.npz --max-degradation 0.02
```
