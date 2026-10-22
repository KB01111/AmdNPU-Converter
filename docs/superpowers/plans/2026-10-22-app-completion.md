# App Completion Implementation Plan

> **For agentic workers:** Use executing-plans for integration and dispatching-parallel-agents for independent file domains. Steps use checkbox syntax for tracking.

**Goal:** Make the documented conversion workflows usable through the native UI and publish a tested PR.

**Architecture:** Preserve React/Tauri/Rust/SQLite/Python. The contracts below are shared across independently owned frontend, Rust, and worker tasks. The controller owns packaging, CI, documentation, integration, and review.

**Tech Stack:** React 18, TypeScript, Tauri 2, Rust, Python 3.11+, ONNX Runtime, AMD Quark, WinML.

## Global constraints

- All shell commands use `rtk`; use `rtk proxy` for unsupported commands.
- Never execute recipe command strings or remote model code.
- Fail clearly on missing prerequisites, failed validation, and device fallback.
- Synthetic inputs do not establish accuracy; provider names do not establish NPU node execution.
- Tests must show failures before fixing the behavior. Use standard-library Python unittest, Rust unit tests, and frontend tests appropriate to the behavior.
- Do not download model weights or install drivers/large ML environments during verification.

## Shared contracts

Rust serde fields are camelCase unless describing Python worker output. Existing Rust metadata/hardware/model-row shapes remain authoritative.

```ts
interface ConversionRequest {
  modelId: string; url: string; precision: string; device: string;
  powerMode: string; targetEp: string; outputDir?: string | null;
  sourceKind?: "hf" | "onnx"; onnxPath?: string | null;
  calibrationDir?: string | null; evalNpz?: string | null;
  maxDegradation?: number; maxSamples?: number;
  iterations?: number; warmup?: number;
}
interface JobStatus {
  jobId: string; modelId: string; status: string; currentStep: string;
  progress: number; logs: {ts:string;stream:string;line:string}[];
  artifacts: string[]; error?: string | null; request?: ConversionRequest | null;
}
interface BenchmarkRequest {
  modelPath:string; devices:string[]; iterations?:number; warmup?:number; jobId?:string | null; targetEp?:string | null;
}
interface BenchmarkResult {
  device:string; providers:string[]; avgLatencyMs?:number | null;
  p50Ms?:number | null; p90Ms?:number | null; throughputPerS?:number | null;
  npuVerified:boolean; error?:string | null; memoryMb?:number | null;
}
```

Commands: `start_conversion({request})->JobStatus`, `get_job_status({jobId})->JobStatus`,
`cancel_job({jobId})->void`, `retry_job({jobId})->JobStatus`, `list_jobs({limit?})->JobStatus[]`,
`run_benchmark({request})->BenchmarkResult[]`, `list_models()->ModelRow[]`,
`open_model_folder({modelId})->void`, `list_recipes()->Recipe[]`,
`load_recipe({path})->Recipe`, `save_recipe({path,recipe})->void`,
`export_report({path,contents})->void`. HF inspect and hardware commands retain current signatures.

`Recipe` is the shipped JSON structure: `recipe_version`, `name`, `conversion_request`, optional
`source.onnx_path`, optional `quantization.calibration.dir/max_samples` and
`quantization.accuracy_gate.max_degradation`. Load those data fields; ignore `commands`.
Progress events contain `jobId` plus any of `status,currentStep,progress,error,stream,line`.

Worker CLI: add `download.py --model ID --out DIR` safe snapshot;
`prepare_onnx.py --input PATH --output PATH --opset 17` validates/converts local ONNX;
exporters keep existing CLIs. `evaluation/bench.py` keeps `--devices`, `--iterations`,
`--warmup`, returns existing snake_case metric JSON; Rust normalizes it without fake zeros.
`npu_verified` is false absent node placement evidence. Missing requested accelerator is an
error row and nonzero exit, with successful rows retained for standalone benchmarks.
`evaluation/compare.py` must reject missing supplied datasets, missing/mismatched outputs,
nonfinite drift, and invalid thresholds.

Integration decisions: ModelRecord additionally persists optional `jobId`/`targetEp`
for producing-artifact context. Optional benchmark `targetEp` maps to worker `--ep`;
only its matching device is pinned. Memory is an RSS snapshot, not a claimed peak.
Windows benchmark/discovery use the isolated WinML runtime, with installed-only
dynamic provider registration. Local ONNX uses stage `prepare`; validation precedes
compilation. Windows Job Objects own subprocess trees, including probes and
standalone benchmarks, so process exit/future cancellation cannot orphan workers.

### Task 1: Python worker integrity

Files: workers/download.py, workers/prepare_onnx.py, workers/evaluation/*.py,
workers/tests/test_workers.py. Owner: worker agent.

- [x] Write unittest regressions: `providers_for('npu',['CPUExecutionProvider'])` must reject fallback; missing supplied eval file returns nonzero; allclose cannot pass missing/nonfinite outputs; safe download excludes pickle weights.
- [x] Run `rtk proxy python -m unittest discover -s workers/tests -v` and observe expected failures.
- [x] Implement safe download, ONNX validation/preparation, strict benchmark provider/error reporting and evaluation inputs. Unit-test without installing ML dependencies, injecting fake modules only at external execution boundaries.
- [x] Run worker tests and Python compilation, report exact commands/results and limitations.

### Task 2: Native orchestration

Files: src-tauri/src/**. Owner: Rust agent.

- [x] Write failing Rust tests for request validation, request/response serde contracts, worker argument routing, benchmark JSON normalization, recipe validation, SQLite restart recovery and terminal state handling.
- [x] Run focused `rtk cargo test --manifest-path src-tauri/Cargo.toml` and confirm failures.
- [x] Align command signatures with shared contracts; validate paths, precision/provider/device combinations, calibration/evaluation and numeric bounds. Implement source routing, safe subprocess cancellation and cleanup, persistent failure/artifact/request data, per-job output isolation and retry.
- [x] Run workers in the correct venv, use packaged resources independent of cwd, persist artifacts and benchmark output, add data-only recipe/file-report commands. Resolve resources through `app.path().resource_dir()` and configured workers/recipes mapping. Development source fallback uses `CARGO_MANIFEST_DIR` parent.
- [x] Test database/serde/routing and error paths; run cargo test and cargo check. Do not claim native NPU success.

### Task 3: Complete frontend workflows

Files: src/**, frontend tests and test configuration, package.json/package-lock.json. Owner: frontend agent.

- [x] Write failing regression tests for invoke payloads, mixed job-event merging, recipe settings, stale source metadata, and navigation/actions.
- [x] Correct all DTOs to shared contracts and existing Rust metadata/hardware/model rows; complete HF/ONNX/CUA-S1 import, file pickers, settings/recipes, event subscription above pages, progress/logs/errors/cancel/retry, hardware readiness, benchmark details/export, model artifact and navigation actions.
- [x] Use `@tauri-apps/plugin-dialog` open/save (directory true for folders); text paths remain editable. New report and recipe writes go through Rust commands. Include loading/errors, accessible labels/native selects, reduced motion, responsive navigation and honest browser runtime guidance.
- [x] Keep current design, lazy-load heavy pages, retain form/benchmark state across navigation. Avoid optimistic completion/verification claims.
- [x] Run targeted tests then `rtk npm run build`; report exact test outcomes. Vitest/jsdom/testing-library may be added only for meaningful frontend regressions.

### Task 4: Integrate, verify and publish

Files: src-tauri/tauri.conf.json, .github/workflows/ci.yml, README.md, CHANGELOG.md,
tests/e2e-conversion.md, docs/superpowers/**. Owner: controller.

- [x] Wire frontend dev/build lifecycle and bundle worker/recipe source resources while excluding venvs/cache. Test packaged lookup and compile resource schema.
- [x] Run integrated frontend, Rust and Python checks; fix cross-task incompatibilities and review all trust/cancellation/accuracy boundaries.
- [x] Exercise UI in browser with explicitly injected native fixtures and native app where available. Capture UI evidence; browser fixtures do not establish hardware acceptance.
- [x] Update docs/changelog and CI to match supported behavior and actual verification limits; remove misleading automatic-fix claims.
- [x] Review complete diff, commit, push `codex/complete-app-workflows`, create/attach PR, inspect final-head hosted CI and fix valid failures.

Completion evidence: see docs/verification/2026-10-22-app-completion.md. Hosted CI is tracked on the PR; real model/hardware acceptance remains separately unverified.
