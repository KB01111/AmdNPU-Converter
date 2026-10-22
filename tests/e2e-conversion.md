# Native conversion acceptance checklist

These checks require real model access, representative data, the external ML
environments, and matching NPU drivers/providers. Browser fixtures, successful
compilation, and provider enumeration are separate evidence. Leave unavailable
checks unchecked and record the missing prerequisite.

## Setup

- [ ] Python 3.11 worker bootstrap finishes without failures.
- [ ] WinML CLI is discoverable (`WINML_BIN`, environment, or PATH).
- [ ] AMD Quark and an actual VitisAI/NPU ORT provider are available.
- [ ] Native app starts through `npm run tauri dev` without a separately started Vite.
- [ ] Installed build resolves bundled workers/recipes from a different cwd.
- [ ] Hardware page shows the actual accelerator and drivers, without treating CPU names as NPUs.
- [ ] Evaluation NPZ includes all model inputs and representative examples.

## A: Local ONNX

- [ ] Select local ONNX import and an existing source graph.
- [ ] Choose supported precision/provider, evaluation file, calibration if needed, and output root.
- [ ] Start: HF download/export are skipped; source graph is not overwritten.
- [ ] ONNX checks/opset preparation, optimization, quantization, compile, and numeric gate succeed.
- [ ] EPContext wrapper and sidecars are present in the per-job output directory.
- [ ] Benchmark records actual NPU execution-profile evidence without fallback nodes.
- [ ] Library opens the correct output folder and populates the benchmark path.

## B: Hugging Face standard

- [ ] Inspect `google/bert_uncased_L-2_H-128_A-2` (safetensors verified at `30b0a37ccaaa32f332884b96992754e246e48c5f`).
- [ ] Inspect `WinKawaks/vit-tiny-patch16-224` (safetensors verified at `77d1485af66b34d4ed0fe95dbb0c60c7496f950b`).
- [ ] Confirm the default revision still matches the [verified checkpoint files](fixtures.md) and record the actual revision/hash used.
- [ ] Metadata shows supported export strategy; source edits invalidate old metadata.
- [ ] Configure representative calibration/evaluation inputs for the chosen model.
- [ ] Safe download, correct text/image exporter, optimize, quantize and compile succeed.
- [ ] Numeric gate passes on supplied data, with actual threshold/output drift retained.
- [ ] NPU benchmark has execution-profile evidence; GPU or CPU execution cannot pass as NPU.

## C: CUA-S1

- [ ] Official `cua_s1` model implementation is installed locally.
- [ ] Forms/nano-text model safetensors match the official model strictly.
- [ ] Choose preset and representative model-specific calibration/evaluation data.
- [ ] Correct variant exports; Forms exposes its five expected inputs and scores output.
- [ ] Quantize/compile/evaluate succeed; NPU benchmark has placement evidence.
- [ ] Missing official implementation or mismatched weights fail clearly without substitute graphs.

## Lifecycle and error paths

- [ ] Save/load a recipe preserves source/configuration and does not execute its command strings.
- [ ] Unsupported INT4, invalid paths, invalid numeric bounds and provider/device conflicts are rejected.
- [ ] Missing worker dependencies and missing accelerator produce actionable errors.
- [ ] Cancel during an active worker: child execution stops before terminal cancelled is shown.
- [ ] Retry creates a new job/output directory from persisted settings.
- [ ] Restart interrupts orphaned jobs; completed jobs and errors remain intact.
- [ ] Missing evaluation NPZ, missing input keys, output mismatches/NaN/Inf and excess drift fail.
- [ ] Switching pages retains inputs/results; events and terminal status remain correct.
- [ ] Keyboard navigation, native selects, labels, focus, reduced motion and 860px layout work.
- [ ] Exported logs/benchmark reports contain actual results and no credentials.

Record model revision, model/data hashes, provider/driver/tool versions, precision,
device, threshold, artifact paths, and test time alongside native acceptance.
