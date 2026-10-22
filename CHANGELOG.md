# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Local ONNX import, CUA-S1 presets, calibration/evaluation settings, native
  file selection, and versioned recipe load/save controls.
- Job cancellation/retry/restart recovery, persistent diagnostics and artifacts,
  benchmark reports, and library navigation/artifact actions.
- Regression coverage for UI/native contracts, worker gates, and job lifecycle;
  CI executes behavior tests in addition to compilation.

### Fixed

- Frontend/Rust command arguments and metadata/hardware/job/model/benchmark
  response contracts, including partial progress/log/error events.
- Missing safe HF download and ONNX preparation workers; real exporter,
  quantization, evaluation, and WinML CLI routing.
- Silent accelerator fallback, false NPU verification, unsafe CUA-S1 weight
  matching, and acceptance of missing/nonfinite evaluation data.
- Worker/recipe resource packaging, frontend build/dev hooks, keyboard controls,
  reduced-motion behavior, and desktop/runtime guidance.

## [0.1.0] - 2026-10-22

### Added

- Initial release.
- Tauri v2 desktop app with React frontend.
- Rust core (`src-tauri`): job pipeline (`inspection, download, export,
  optimize, quantize, compile, validate, benchmark`) and WinML bridge
  with cache-key computation.
- Python workers (`workers/`): HF inspection, transformers/CUA-S1/generic-ONNX
  exporters, Quark quantization, WinML runner, evaluation (`compare.py`,
  `bench.py`).
- Versioned conversion recipes (`recipes/`) per architecture
  (`cnn-a8w8`, transformer-`bf16`, LLM-`int4` where worker-supported).
- Docs: quickstart, cache-key contract, security notes, troubleshooting,
  manual e2e checklist and test fixtures.
