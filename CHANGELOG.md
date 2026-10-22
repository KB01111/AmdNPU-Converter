# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
