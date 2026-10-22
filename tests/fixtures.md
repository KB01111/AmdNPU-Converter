# Test fixtures and real-model acceptance

Automated UI and worker tests use explicit native/SDK fixtures. They do not
download weights, run real Quark/WinML compilation, measure model accuracy, or
establish NPU acceptance. Real-model checks are the separate manual procedure in
[e2e-conversion.md](e2e-conversion.md).

Potential real-model inputs, subject to repository access and installed exporter
support:

| Model | Purpose | Input preparation |
|---|---|---|
| `WinKawaks/vit-tiny-patch16-224` ([verified revision](https://huggingface.co/WinKawaks/vit-tiny-patch16-224/tree/77d1485af66b34d4ed0fe95dbb0c60c7496f950b)) | image classification export | processor-generated pixel_values; representative images for evaluation/calibration |
| `google/bert_uncased_L-2_H-128_A-2` ([verified revision](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f)) | small text model export | tokenizer arrays, including every exported input |
| `cua-ai/cua-s1-forms` | official Forms checkpoint | official collator, checkpoint-defined context/option lengths, boolean masks |
| `cua-ai/cua-s1-nano-0.1` | official Nano text checkpoint | text checkpoint and official collator; exported wrapper uses the five context/option arrays |

The standard text/image checkpoints above were checked on 2026-10-22: both
referenced revisions provide `model.safetensors` and `config.json`, with BERT
`vocab.txt` or ViT `preprocessor_config.json` respectively. These are file-access
checks, not completed conversion or hardware acceptance. The app downloads the
repository's default revision; confirm it still matches the linked revision and
record the actual revision/hash used for acceptance.

CUA requires the separately installed trusted upstream `cua_s1` implementation.
Forms source files are `cua-s1-forms.safetensors` plus its matching JSON;
Nano text uses `text/model.safetensors` and `text/config.json`. The download
worker normalizes these to a local `model.safetensors`/`config.json` pair.
Bytes are encoded as UTF-8 byte values plus one, zero is padding, and masks are
boolean. Derive lengths from the actual checkpoint configuration.

Supply real `.npz` evaluation data with one array per ONNX input. Integer
quantization also needs representative complete calibration samples. Synthetic
export/benchmark inputs are only tracing/timing inputs. Numeric output drift is
not an accuracy or perplexity score.

Never commit private weights, model tokens, or large calibration datasets. The
download/export paths disable remote code and reject pickle-only model loading.
Access failures and missing tool/model support must be recorded as unavailable
acceptance, rather than replaced with fake weights or substitute architectures.

The INT4 recipe is an unsupported reference; TinyLlama/other LLMs are not accepted
fixtures for the current ONNX pipeline. Real NPU evidence requires actual
profiled NPU node execution without fallback, together with model/data hashes
and provider/driver/tool versions.
