# Python worker engine

Workers emit JSON to stdout, diagnostics to stderr, and nonzero status on required
failures. Use Python 3.11: the pinned WinML CLI 0.4.0 requires `>=3.11,<3.12`.

## Environments

Run `powershell -ExecutionPolicy Bypass -File workers/bootstrap.ps1` to create
the base `.venv`, `.venv-winml`, and `.venv-quark` directories under `workers/`.
Installation failures stop bootstrap. The base project installs dependencies;
worker source is bundled separately by the desktop app.

Each environment has exactly one ONNX Runtime distribution. The base uses
DirectML on Windows and CPU ORT elsewhere; the WinML environment lets the pinned
CLI select matching `windowsml` and `onnxruntime-windowsml` versions. Recreate old
environments containing multiple ORT wheels; additive installs cannot reliably
repair their shared `onnxruntime` namespace. Benchmark/discovery reject conflicts.

The CUA exporters additionally require the official `cua_s1` source package from
[trycua/cua](https://github.com/trycua/cua/tree/main/libs/cua-s1/python) installed
from a trusted checkout into the base environment. That research package is not
published to PyPI. Export fails clearly when it is unavailable. No model code is
fetched or executed by a conversion.

`winml.exe` lookup: `WINML_BIN`, `.venv-winml/Scripts/winml.exe`,
`.venv/Scripts/winml.exe`, then `PATH`.

## Import and export

From the repository root:

```powershell
workers/.venv/Scripts/python.exe workers/hf-inspector/inspect.py --model-id org/model --json
workers/.venv/Scripts/python.exe workers/download.py --model org/model --out job/source
workers/.venv/Scripts/python.exe workers/exporters/transformers/export_hf.py --model-id job/source --output job/model.onnx
workers/.venv/Scripts/python.exe workers/prepare_onnx.py --input original.onnx --output job/model.onnx --opset 17
```

Inspection fetches only checkpoint JSON metadata. Download accepts a new or empty
directory and safetensors/config/tokenizer/image-processor data only. Pickle
weights and Python files are excluded. HF repositories containing only ONNX can
be used through local ONNX import instead. Preparation preserves the original,
validates local external tensor references, and converts opsets when supported.

The Transformers exporter handles text feature/classification models and image
classifiers including ConvNext. It uses `trust_remote_code=False`,
`use_safetensors=True`, local-only loading for downloaded snapshots, and requires
successful checker/runtime/numeric validation. Other architectures fail with
their loader or input error rather than substituting random model inputs.

Official `cua-ai/cua-s1-forms` and `cua-ai/cua-s1-nano-0.1` downloads normalize the
Forms pair or Nano **text** pair to `model.safetensors` and `config.json`:

```powershell
workers/.venv/Scripts/python.exe workers/exporters/cua-s1/export_cua_s1.py --weights job/source/model.safetensors --config job/source/config.json --output job/model.onnx --variant forms
```

Use `--variant nano-text` for Nano. Official loaders verify tensor signatures and
named state; there is no invented model or shape-based weight substitution.
Inputs use UTF-8 byte values plus one, with zero padding and boolean masks.
Export smoke inputs establish graph equivalence only, not task accuracy.

## Quantization and evaluation

```powershell
workers/.venv-quark/Scripts/python.exe workers/quark/quantize.py --input job/model.onnx --output job/quant.onnx --profile a8w8 --calibration-dir calib
workers/.venv/Scripts/python.exe workers/evaluation/compare.py --baseline job/model.onnx --candidate job/quant.onnx --eval-npz eval.npz --max-degradation 0.02
workers/.venv-winml/Scripts/python.exe workers/winml/run.py compile --model job/quant.onnx --output job/compiled.onnx
workers/.venv-winml/Scripts/python.exe workers/evaluation/bench.py --model job/compiled.onnx --devices cpu,npu --ep vitisai --iterations 100 --warmup 10
```

Quark profiles map to the official `BF16`, `A16W8`, `A8W8`, and `XINT8` presets.
Unavailable profiles/APIs fail. Integer profiles require representative
calibration samples for every model input; NPZ files contain arrays named for
each input. Explicit NPY mappings form one complete sample. Calibration never
uses pickle loading or silently drops inputs.

Comparison rejects missing evaluation datasets, missing/mismatched output names,
counts or shapes, nonfinite values, and invalid thresholds. It measures relative
numeric drift on the supplied input arrays, not a labeled task accuracy score.
Without evaluation data it can print a synthetic smoke report but cannot pass.

Benchmark returns a JSON array retaining successful and failed device rows. Any
missing provider, actual provider fallback, or inference error yields a nonzero
exit. `npu_verified` requires nonempty ONNX Runtime kernel profiling evidence
attributing **every recorded node** to VitisAI/NPU providers. CPU/GPU nodes,
unknown placement, missing profiles, and provider names alone cannot verify an
NPU run. DirectML/OpenVINO availability does not identify an NPU. This evidence
still requires validation on the installed provider/driver and real hardware.

Use `--ep cuda`, `dml`, `vitisai`, `cpu`, or an exact provider name to pin the
matching device's provider; other comparison devices retain their defaults.
A missing explicitly selected provider fails rather than selecting another.
GPU rows also require nonempty profiling with every recorded kernel on the
selected GPU provider; CPU/mixed/unknown placement produces an error row.

On Windows, benchmarking and hardware discovery run in `.venv-winml`, register
only installed providers through `windowsml.EpCatalog` (the pinned 2.0.300 API),
and merge dynamic `ort.get_ep_devices()` entries with built-in providers.
Sessions bind the matching provider **and hardware device type** through
`SessionOptions.add_provider_for_devices`. A GPU handle exposed by the same
provider cannot satisfy an NPU request. `NotPresent` providers are skipped without
calling readiness/download methods; `NotReady` installed providers may be added
to the process dependency graph. Discovery never downloads providers or models.

`bench.py --discover-json` needs no model or NumPy and returns
`providers: string[]`, `devices: {provider, device, name}[]`, `errors: string[]`,
and `windowsml_available: boolean`. Unknown hardware types remain `unknown`.
Missing catalog support is reported alongside any usable built-in providers.

## Checks without ML dependencies

```powershell
python -m unittest discover -s workers/tests -v
python -m compileall -q workers
```

Unit tests replace external SDK execution boundaries with explicit fakes. They
do not download weights or establish WinML, Quark, model accuracy, or native NPU
acceptance.
