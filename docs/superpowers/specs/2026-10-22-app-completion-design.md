# App completion design

Status: full scope approved by the user on 2026-10-22.

## Goal and scope

Complete the workflows already promised by README.md, recipes/, and
tests/e2e-conversion.md. Keep React, Tauri v2, Rust-owned job orchestration,
SQLite history, and Python exporters/evaluation. Deliver one implementation PR
with tests and an honest account of native and hardware verification.

This means completing Hugging Face import, local ONNX import, CUA-S1 export,
conversion configuration and recipes, job monitoring and recovery, hardware
readiness, benchmark results, and model artifact actions. It does not mean
inventing new model architectures or claiming unsupported precision/provider
combinations work.

## Findings at origin/main e8ca526

1. Frontend/backend contracts disagree across all primary workflows.
   `start_conversion` expects `request`, `hfUrl`, and `ep`; the UI sends `req`,
   `url`, and `targetEp`. Job commands expect `jobId`, while wrappers send `id`.
   Rust emits `jobId`, `currentStep`, structured log lines, and several event
   shapes; the UI expects `id`, `stage`, string logs, and a single event shape.
   Hardware, model metadata, model rows, and benchmark results also disagree.
2. The pipeline invokes a nonexistent `workers/download.py`. Export,
   quantization, validation, and benchmarking bypass the supplied workers or
   omit their required inputs. `device` and `powerMode` are accepted without
   affecting execution, and output-directory selection is absent.
3. ONNX import, recipe selection/save/load, calibration/evaluation inputs, and
   dedicated CUA-S1 routing are documented but unavailable in the UI.
4. The benchmark command passes `--ep` and `--json` to a worker that accepts
   `--devices` and returns an array with different metric names. The worker
   treats any non-CPU provider as NPU verified and silently substitutes CPU.
5. Cancellation marks state before the running subprocess stops and is only
   observed between stages. Registry cleanup, restart recovery, persistent
   failure details, and artifact reporting need completion.
6. Library Open changes a job ID without navigating to Pipeline. Model rows
   lack usable artifact/benchmark actions. Failed cancellations are not shown.
7. Settings use a custom select without normal select keyboard semantics;
   navigation and several inputs lack accessible labeling/state. Reduced
   motion, loading states, runtime guidance, and user-facing descriptions need
   attention. The frontend eagerly bundles every page.
8. Worker lookup relies on the current working directory; Tauri does not bundle
   worker/recipe resources or start/build the frontend from its lifecycle hooks.
9. CI currently builds/typechecks and compiles Python syntax; it does not catch
   the cross-language contract or worker behavior failures above.

## Approach

Recommended: repair the contracts first, then complete each documented workflow
through the existing layers. A UI-only pass would retain broken backend paths.
A rewrite would increase risk without solving an architectural limitation.

### Contracts and state

Make the Rust command/serde definitions and Python JSON output authoritative.
Update TypeScript types and wrappers consistently, with regression coverage for
argument names, payloads, progress/log/error events, and metric units. Avoid
fabricated response fields and zero-valued metrics for malformed output.

Keep shared navigation, conversion inputs, active jobs, and benchmark selection
in the existing store. Subscribe to job events above individual pages so progress
survives navigation. Merge partial event shapes safely, cap displayed logs, and
poll active jobs as recovery. Show terminal states and errors persistently.

### Import, settings, and recipes

Offer Hugging Face and local ONNX input modes, plus the existing CUA-S1 presets.
Accept canonical HF IDs and URLs; clear stale inspection results when sources
change and ignore stale asynchronous inspection responses.

Expose output directory, supported precision, target provider, calibration
directory, evaluation NPZ, maximum degradation, and benchmark settings. Choose
files/directories with native dialogs, retaining text paths for clarity. Validate
the same constraints again in Rust before creating jobs.

Load the shipped versioned recipes as data and support saving/loading user
recipes. Parse and validate schemas; never execute their `commands` strings.
Preserve settings required for reproducibility. Unsupported INT4/provider or
power-policy options must be explicitly unavailable or advisory rather than
pretending to take effect.

### Pipeline and workers

Implement the missing safe HF download stage with safetensors-only weights and
no remote code execution. Route Transformers and CUA-S1 to their existing
exporters. Local ONNX skips HF inspection/download/export and validates ONNX and
opset before optimization. Keep local source files intact.

Use the correct worker environment and arguments for each stage. Resolve auto
precision to a supported concrete profile; pass calibration data into Quark and
require the configured representative evaluation data for the acceptance gate.
Validate baseline/candidate outputs before completing the job. Compile with the
selected supported provider and persist verified artifact paths and reports.

Use isolated per-job output locations to avoid concurrent jobs corrupting each
other. Treat every required worker failure and missing artifact as failure.
Persist actionable errors and complete registry cleanup on every exit. Cancel
the active process and stop further stages before reporting cancelled; avoid
unsafe in-process NPU teardown. On restart, mark orphaned jobs interrupted.

Provide retry from the saved request with a new job/output location. The README's
Apply Fix & Retry claim must be reconciled: do not ship a cosmetic repair action
without a validated graph transformation; retain clear diagnostics and ordinary
retry where automatic repair cannot be demonstrated safely.

### Hardware, benchmark, and library

Show actual CPU/GPU/NPU/WinML/provider and driver fields, with setup guidance and
readiness/errors. Hardware refresh must not overwrite user conversion choices.

Benchmark the selected ONNX/EPContext artifact with the worker's real CLI, expose
iteration/warmup controls, per-device failures, provider details, latency
percentiles, throughput in inferences/s, and available memory information.
Missing or fallback providers cannot count as successful NPU verification.
Provider availability alone is insufficient proof of node placement; use
execution evidence where supported and report unverified placement honestly.

Persist results with job/model context. Library rows navigate to jobs, open
actual artifact folders, and populate the benchmark page. Refresh completed
models/jobs, provide empty/loading/error states, and avoid promising nonexistent
size/metric fields. Add copy/export actions for logs and benchmark reports where
they support diagnosis and reproducibility.

### UI and packaging

Keep the current five-page layout and visual language. Use accessible labels,
native keyboard-capable selection, visible focus, status announcements, readable
paths/logs, responsive navigation, and reduced-motion preferences. Replace
implementation commentary in product text with instructions about the task.
Lazy-load pages where it reduces the initial bundle without complicating state.

Connect Tauri's frontend development/build lifecycle, bundle workers and recipes,
resolve installed resources independently of cwd, and show missing external
Python/WinML/Quark dependencies clearly. Do not install drivers, download model
weights, or run expensive native conversions automatically during validation.

## Verification and acceptance

- Regression tests exercise real command payloads/serde shapes, event merging,
  stale inspection handling, settings/recipe validation, cancellation/recovery,
  worker routing, evaluation gates, and benchmark fallback/error behavior.
- Frontend typecheck/build, Rust tests/checks, and Python behavior tests run in CI.
- Exercise import/settings/recipes, job progress/history, hardware errors,
  benchmark results, keyboard selection, and artifact actions in UI tests with
  explicit test fixtures. Fixture success is not native hardware acceptance.
- Build and launch the native app if this machine's prerequisites allow it;
  confirm packaged resource resolution separately from browser rendering.
- Run the documented three conversion flows only when real model access,
  representative calibration/evaluation datasets, WinML/Quark, and Ryzen AI NPU
  are available. Record any unavailable acceptance rather than reporting it as
  passed. No synthetic input may establish model accuracy.
- Review the final diff, update docs/changelog to match actual behavior, push a
  `codex/` branch, create/attach a PR, and inspect its final-head hosted checks.

## Baseline evidence

- Working tree was clean; HEAD matched fetched origin/main at e8ca526.
- `npm run build` passed, with a 549 kB JavaScript chunk warning.
- Existing main CI run 39797704198 passed its build-only checks.
- Rust baseline passed compilation, with zero tests in three suites.
- Worker virtual environments are absent from workers/ on this checkout.

## Implementation order after approval

1. Contracts and regression harness.
2. Native request validation, job lifecycle, and worker execution.
3. Imports, configuration/recipes, progress/retry, artifact and benchmark UI.
4. Accessibility, packaging, documentation, end-to-end verification, and PR.
