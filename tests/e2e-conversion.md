# E2E-conversion — manuell checklista

Kor dessa 3 floden manuellt pa en Ryzen AI-maskin (Win11 + NPU-drivrutin).
**Definition of done per flode:** modellen ar **exporterad + validerad +
kompilerad + verifierad pa NPU** (alla fyra, i ordning). Bocka av varje steg.

Forberedelser (gemensamt):
- [ ] `powershell -ExecutionPolicy Bypass -File workers/bootstrap.ps1` gar utan fel
      (skapar `workers/.venv`, `.venv-winml`, `.venv-quark`; Python 3.11 + uv kravs)
- [ ] `workers/.venv-winml/Scripts/python.exe workers/winml/run.py sys --output-json`
      rapporterar NPU + Vitis AI EP
- [ ] `npm install` + `npm run tauri dev` startar appen; `get_hardware_info` visar NPU

Pipeline-referens (Rust `STEPS`): `inspection, download, export, optimize,
quantize, compile, validate, benchmark` — event `job://progress`.

---

## Flode A — ONNX-import -> compile (recept `recipes/generic/onnx-import.json`)

1. [ ] Lagg test-ONNX i `downloads/` (t.ex. exporterad `convnext-tiny-224`, opset 17)
2. [ ] Appen: valj **ONNX-import**, peka pa filen (inspector-strategi `direct-onnx`)
3. [ ] **Exporterad/fardig import:** `winml/run.py optimize -- --in ./downloads/model.onnx --out ./artifacts/onnx-import/model.opt.onnx`
4. [ ] **Kvantiserad:** `quark/quantize.py --input model.opt.onnx --output model.<profil>.onnx --profile <bf16|a16w8|a8w8|int8> --calibration-dir ./calibration/onnx-import --max-samples 100` (onnx.checker + ORT CPU-load OK i rapporten)
5. [ ] **Kompilerad:** `winml/run.py compile -- --in model.<profil>.onnx --out model.vitisai.bin --ep vitisai`
6. [ ] **Validerad:** `evaluation/compare.py --baseline model.opt.onnx --candidate model.<profil>.onnx --eval-npz ./calibration/onnx-import/eval.npz --max-degradation <troskel>` exit 0
7. [ ] **Verifierad NPU-korning:** `evaluation/bench.py --model model.vitisai.bin --devices npu ...` visar NPU-provider (VitisAIExecutionProvider), ingen tyst CPU-fallback

## Flode B — HF-standard -> build (recept `recipes/generic/hf-default.json`)

1. [ ] Appen: valj **HF-standard**, ange `facebook/convnext-tiny-224` (alternativ `prajjwal1/bert-tiny`, se `tests/fixtures.md`); `hf_inspect` visar metadata + strategi `transformers`
2. [ ] **Exporterad:** `exporters/transformers/export_hf.py --model-id <id> --output ./artifacts/hf-default/model.onnx --opset 17 --batch 1 --seq-len 128` (ort_run=ok; trust_remote_code=false hardkodat)
3. [ ] **Kvantiserad + kompilerad:** optimize -> `quantize.py --profile <profil>` -> `winml/run.py compile -- --ep vitisai` ger `model.vitisai.bin`
4. [ ] **Validerad:** `compare.py --max-degradation <troskel>` exit 0
5. [ ] **Verifierad NPU-korning:** `bench.py --devices npu` OK; folj jobbet via `get_job_status`/`job://progress`
6. [ ] (Valfritt) **Save as Recipe** -> `recipes/custom/<namn>.json`, ladda om, kor om

## Flode C — CUA-S1 Forms -> custom export (recept `recipes/cua-s1/forms-a16w8.json`)

1. [ ] Appen: valj **CUA-S1 Forms**, ange `cua-ai/cua-s1-forms` (HF-token vid behov)
2. [ ] Recept `cua-s1/forms-a16w8` autovaljs (strategi `cua-s1`); kontrollera **5 inputs** i exportrapporten: `context_ids [1,224]`, `context_mask [1,224]`, `option_ids [1,4,96]`, `option_token_mask [1,4,96]`, `option_mask [1,4]` -> `scores [1,4]`
3. [ ] **Exporterad:** `exporters/cua-s1/export_cua_s1.py --weights <model.safetensors> --config <config.json> --model-id cua-ai/cua-s1-forms --output ./artifacts/forms-a16w8/model.onnx --variant forms --opset 17` (allclose=true; vikter endast via safetensors)
4. [ ] **Kvantiserad (a16w8):** `quantize.py --input model.onnx --output model.a16w8.onnx --profile a16w8 --calibration-dir ./calibration/forms-a16w8 --max-samples 200` (representativa formular-`.npz`, en nyckel per inputnamn)
5. [ ] **Kompilerad:** `winml/run.py compile -- --in model.a16w8.onnx --out model.vitisai.bin --ep vitisai`
6. [ ] **Validerad + verifierad:** `compare.py --max-degradation 0.01` exit 0, `bench.py --devices npu` visar NPU-korning

---

## Felvagar att provoka (minst en per flode)

- [ ] Annat opset an 17 -> konvertera med ONNX Version Converter fore optimize, eller fel tydligt
- [ ] Kalibrerings-`.npz` saknar inputnycklar -> `quantize.py`/`compare.py` felar tydligt (exit != 0), ingen tyst fallback
- [ ] `compare.py` over `--max-degradation` -> exit != 0, bygget stoppas fore distribution
- [ ] `quark` saknas i `.venv-quark` -> exit 2 med install-hint, ingen fejkad kvantisering
- [ ] NPU saknas/drivrutin gammal -> `run.py sys` + `bench.py --devices npu` visar CPU-fallback oppent; verifierade floden godkanns EJ
