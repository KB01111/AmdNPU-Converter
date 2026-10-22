[![CI](https://github.com/KB01111/AmdNPU-Converter/actions/workflows/ci.yml/badge.svg)](https://github.com/KB01111/AmdNPU-Converter/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# AMD NPU Model Converter

A Windows desktop app for importing Hugging Face or local ONNX models,
configuring export/quantization, compiling with WinML, checking numeric drift,
and measuring inference performance. Built with React, Tauri 2, Rust, SQLite,
and Python workers.

The app orchestrates external ML tools. It does not bundle Python environments,
model weights, WinML, Quark, or NPU drivers. Supported model/operator/precision
combinations depend on the installed tools and target hardware; failures are
reported as failures rather than replaced with simulated results.

## Desktop setup

- Windows 11; AMD Ryzen AI hardware and matching driver/provider stack for NPU.
- Python 3.11 and [uv](https://docs.astral.sh/uv/).
- Node.js 22+, Rust stable, and the Windows WebView2/MSVC prerequisites for
  [Tauri development](https://v2.tauri.app/start/prerequisites/).
- Hugging Face access for the selected model. Use the local Hugging Face
  authentication setup; the app does not store tokens in recipes or logs.
- Representative calibration samples and an evaluation `.npz` with a named
  array for every model input. Model weights and datasets are user-supplied.

```powershell
npm ci
# Installs the actual ML dependencies; can require substantial disk space.
powershell -ExecutionPolicy Bypass -File workers/bootstrap.ps1
npm run tauri dev
```

Tauri starts Vite automatically. `npm run tauri build` builds the frontend and
native application and includes the worker scripts and recipes. A plain
`npm run dev` is a browser preview: native inspection, file dialogs, conversion,
hardware, and benchmark operations require the desktop app.

For an installed app, prepare worker environments in a writable copy of the
`workers` directory and set this before launching:

```powershell
$env:NPU_CONVERTER_WORKERS_DIR = 'C:\path\to\workers'
# Optional explicit WinML executable:
$env:WINML_BIN = 'C:\path\to\winml.exe'
```

Worker source scripts are bundled separately from the environments. See
[workers/README.md](workers/README.md) for dependency/setup details. Installation
does not modify drivers or install ML packages automatically.

## Workflows

**Import** accepts a Hugging Face URL or `owner/repo` ID, local ONNX files, and
CUA-S1 presets. Inspect HF metadata before converting; changing the source
invalidates the previous inspection. Model files are downloaded only after a
conversion starts. HF export uses safetensors and disables remote code.

**Settings and recipes** choose output location, concrete precision/provider,
calibration directory, evaluation NPZ, numeric-drift threshold, sample limit,
and benchmark iterations/warmup. Shipped recipes can populate the controls;
custom recipes can be saved and loaded through native dialogs. Recipes are
data, and their historical `commands` strings are never executed.

`auto` resolves to the supported BF16 profile. Available Quark profiles are
BF16, A16W8, A8W8, and INT8; calibration is required for calibrated integer
profiles. Unsupported precision/device/provider combinations fail validation.
Power management remains automatic; the app does not claim to change Windows
power policy. INT4 is not implemented by the ONNX worker.

**Pipeline** shows the active job, stage status, logs, errors, and artifacts.
HF flows download and export before optimization/quantization/compilation;
local ONNX skips HF download/export. Local imports validate/convert to opset 17.
Each job uses a separate output location. Cancel stops execution before the
job becomes cancelled; retry creates a new job from its saved settings.
Jobs left running when the app exits are marked interrupted on restart.

**Hardware** reports detected CPU/GPU/NPU, WinML availability, providers, and
driver information. Detection is a readiness hint, not evidence that a model
executes on the NPU.

**Benchmark** runs a selected ONNX/EPContext model on requested devices, with
latency percentiles, throughput in inferences per second, available memory
measurements, provider names, and per-device errors. Reports can be exported.
An unavailable accelerator does not silently become a successful CPU run.
NPU verification requires nonempty execution-profile node evidence and rejects
fallback placement; provider availability alone is insufficient. Benchmark
timing inputs are synthetic and do not establish accuracy.

**Library** shows persisted job history and converted artifacts, opens jobs in
Pipeline, opens artifact folders, and selects models for benchmarking. Runtime
state continues to update while switching pages.

## Evaluation and export limits

The conversion acceptance gate requires a real evaluation NPZ. It measures
relative output drift against the optimized baseline, not task accuracy such
as ImageNet top-1. Missing inputs, missing datasets, incompatible outputs,
nonfinite values, or excessive drift fail the gate. Representative input data
and an appropriate threshold remain the model owner's responsibility.

Transformers export covers supported text/image architectures through installed
Transformers implementations. Custom remote-code models and arbitrary pickled
checkpoints are not automatically trusted. CUA-S1 requires its official model
implementation installed locally and a strict matching safetensors state;
the app does not invent a substitute architecture when that loader is absent.

There is no unvalidated automatic “Apply Fix & Retry” graph rewrite. Inspect the
actual tool error, correct the model/settings/dependencies, and retry. Successful
builds and fixture tests do not establish live model accuracy or NPU support.

Compilation writes an ONNX EPContext wrapper and any provider-owned sidecars;
keep those artifacts together for deployment. Optimization/compilation use the
current WinML [`--model`/`--output` interface](https://microsoft.github.io/winml-cli/latest/commands/compile/).

## Development checks

```powershell
npm test
npm run build
npx playwright test
cargo test --manifest-path src-tauri/Cargo.toml
python -m unittest discover -s workers/tests -v
python -m compileall -q workers
```

UI tests use installed Edge locally (serial) and Playwright Chromium in CI.
Their native bridge is explicitly a test fixture. Worker tests isolate external
ML dependencies and do not download model weights. Actual hardware acceptance
is a separate [manual checklist](tests/e2e-conversion.md).

Application data is stored under `%APPDATA%\npu-converter`, including SQLite
history and job logs. Outputs use per-job directories; local input files are
not overwritten. Cache identity and model artifacts must not be assumed
portable across provider or driver versions.

## Project layout

- `src/`: typed frontend, conversion controls, dashboards and library.
- `src-tauri/src/`: commands, validation, jobs, process execution, storage.
- `workers/`: download, ONNX preparation, exporters, quantization, evaluation.
- `recipes/`: versioned JSON presets; user-supplied data paths need adjustment.
- `tests/ui/`: browser workflow tests with explicit Tauri fixtures.
