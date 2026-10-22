# Test-fixtures — riktiga modeller (INGA fejkade vikter)

Alla E2E-tester kor mot verkliga vikter. `trust_remote_code=false` galler alltid
(hardkodat i `export_hf.py`; `export_cua_s1.py` laster endast safetensors).
Ladda ner via appen eller `huggingface_hub`; committa aldrig
`.onnx`/`.safetensors`/`.bin` (se `.gitignore`: `downloads/`, `artifacts/`,
`calibration/`-radata hor inte hemma i git om de innehaller stora filer).

| # | Modell (HF-ID) | Typ | Anvands i | Inputs / shapes | Kalibrering |
|---|---|---|---|---|---|
| 1 | `facebook/convnext-tiny-224` | CNN, image-classification (ImageNet-1k, 224x224) | Flode A, B; recept `cnn-a8w8`, `npu-default`, `hf-default`, `onnx-import` | batch 1 (`--batch 1`); NPU-mal 1x3x224x224 via optimize/compile | imagenet-1k-val-subset, 300 samples (CNN) / 100 (default) som `.npz` i `./calibration/<recept>/` |
| 2 | `prajjwal1/bert-tiny` | Liten BERT (L=2, H=128) | Recept `transformer-bf16`; alternativ fixture for flode B | `--seq-len 128 --batch 1` -> `input_ids`/`attention_mask` 1x128 | Ingen (bf16, ingen kvantisering) |
| 3 | `cua-ai/cua-s1-forms` | CUA-S1 Forms (exporter `cua-s1`, byte-vocab 0-255) | Flode C; recept `forms-a16w8` | `context_ids` 1x224, `context_mask` 1x224, `option_ids` 1x4x96, `option_token_mask` 1x4x96, `option_mask` 1x4 -> `scores` 1x4 | cua-s1-forms calibration split (verkliga formular), 200 samples som `.npz` med en nyckel per inputnamn |
| 4 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Liten LLM (~1.1B) | Recept `llm-int4` | `--seq-len 2048 --batch 1` | wikitext-2-raw-v1 val, 512 samples (int4 weight-only, group 128 — nar worker-stod finns; tills dess int8/a16w8) |
| 5 | `cua-ai/cua-s1-nano-text` | CUA-S1 nano-text (exporter `cua-s1 --variant nano-text`) | Recept `nano-text-bf16` | `input_ids` 1x128, `attention_mask` 1x128 | Ingen (bf16) |

Anmarkningar:
- `cua-ai/*` ar projektets egna modeller — atkomst kan krava HF-token
  (`--token` / `HF_TOKEN`). Utan atkomst kan flode C inte koras: dokumentera
  det i testresultatet istallet for att fejka vikter.
- For ONNX-import (flode A): exportera forst fixture 1 med
  `exporters/transformers/export_hf.py --opset 17` och placera resultatet i
  `downloads/` (ignorerad av git).
- Kalibreringsdata: `.npy` (en-inputsmodeller) eller `.npz` (en array per
  inputnamn); multi-input `.npy` avvisas av `quantize.py` med varning.
  `--reference-metrics '{"accuracy": ...}'` + `compare.py --max-degradation`
  ar accuracy-grinden (quantize-rapporten satter aldrig sjalv `accuracy_delta`).
- NPU-bevis: `bench.py --devices npu` maste visa NPU-provider
  (VitisAIExecutionProvider forstahandsval); CPU-fallback = underkant test.
