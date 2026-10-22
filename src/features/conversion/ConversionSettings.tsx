import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { PathInput } from "@/components/ui/path-input";
import { useAppStore } from "@/stores/useAppStore";
import { choosePath, chooseSavePath, errorMessage, isTauriRuntime, listRecipes, loadRecipe, saveRecipe } from "@/lib/tauri";
import type { ConversionRequest, Recipe } from "@/lib/types";
import { validateHfUrl } from "@/features/huggingface/HuggingFaceImport";

export default function ConversionSettings() {
  const s = useAppStore();
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [recipeName, setRecipeName] = useState("");
  const [recipePath, setRecipePath] = useState("");
  const recipes = useQuery({ queryKey: ["recipes"], queryFn: listRecipes, enabled: isTauriRuntime(), retry: false, staleTime: Infinity });
  async function recipeAction(action: "load" | "save") {
    setBusy(true); setError(null); setNotice(null);
    try {
      const path = recipePath.trim() || (action === "load" ? await choosePath(false, "json") : await chooseSavePath());
      if (!path) return;
      if (action === "load") { const recipe = await loadRecipe(path); s.applyRecipe(recipe); setRecipeName(recipe.name); setNotice(`Loaded ${recipe.name}. Check model, calibration and evaluation paths before converting.`); }
      else {
        const state = useAppStore.getState();
        const hf = validateHfUrl(state.huggingfaceUrl);
        if (state.sourceKind === "hf" && !hf) throw new Error("Enter a valid Hugging Face model ID or URL before saving a recipe.");
        const request: ConversionRequest = { modelId: state.sourceKind === "onnx" ? state.localModelId : `${hf!.owner}/${hf!.repo}`, url: state.sourceKind === "hf" ? state.huggingfaceUrl.trim() : "", sourceKind: state.sourceKind, onnxPath: state.sourceKind === "onnx" ? state.onnxPath : null, precision: state.precision, device: state.device, powerMode: state.powerMode, targetEp: state.targetEp, outputDir: state.outputDir.trim() || null, calibrationDir: state.calibrationDir.trim() || null, evalNpz: state.evalNpz.trim() || null, maxDegradation: state.maxDegradation, maxSamples: state.maxSamples, iterations: state.iterations, warmup: state.warmup };
        const recipe: Recipe = { recipe_version: "1.0", name: recipeName.trim() || "My conversion", conversion_request: request, source: { kind: state.sourceKind, ...(state.sourceKind === "onnx" ? { onnx_path: state.onnxPath } : {}) }, quantization: { calibration: { ...(state.calibrationDir.trim() ? { dir: state.calibrationDir.trim() } : {}), max_samples: state.maxSamples }, accuracy_gate: { max_degradation: state.maxDegradation } } };
        await saveRecipe(path, recipe); setNotice(`Recipe saved to ${path}`);
      }
      setRecipePath(path);
    } catch (error) { setError(errorMessage(error)); } finally { setBusy(false); }
  }
  function applyShipped(name: string) {
    const recipe = recipes.data?.find(item => item.name === name);
    if (!recipe) return;
    try { s.applyRecipe(recipe); setRecipeName(recipe.name); setError(null); setNotice(`Loaded ${recipe.name}. Check source and dataset paths.`); }
    catch (error) { setError(errorMessage(error)); }
  }
  return <Card>
    <CardHeader><CardTitle>Conversion settings</CardTitle><CardDescription>Choose a supported precision and provider, then supply representative validation data.</CardDescription></CardHeader>
    <CardContent className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-1.5"><label htmlFor="precision" className="text-sm font-medium">Precision</label><Select id="precision" value={s.precision} onValueChange={precision => s.updateSettings({ precision: precision as ConversionRequest["precision"] })} options={[{ value: "auto", label: "Auto (BF16)" }, { value: "bf16", label: "BF16" }, { value: "a16w8", label: "A16W8 · calibration required" }, { value: "a8w8", label: "A8W8 · calibration required" }, { value: "int8", label: "INT8 · calibration required" }, { value: "fp32", label: "FP32 · unavailable", disabled: true }, { value: "int4", label: "INT4 · unavailable", disabled: true }]} /></div>
        <div className="space-y-1.5"><label htmlFor="device" className="text-sm font-medium">Device</label><Select id="device" value={s.device} onValueChange={device => s.updateSettings({ device: device as ConversionRequest["device"] })} options={[{ value: "auto", label: "Auto (provider target)" }, { value: "npu", label: "NPU" }, { value: "gpu", label: "GPU" }, { value: "cpu", label: "CPU" }]} /></div>
        <div className="space-y-1.5"><label htmlFor="target-ep" className="text-sm font-medium">Target execution provider</label><Select id="target-ep" value={s.targetEp} onValueChange={targetEp => s.updateSettings({ targetEp: targetEp as ConversionRequest["targetEp"] })} options={[{ value: "vitisai", label: "VitisAI · NPU", disabled: !["auto", "npu"].includes(s.device) }, { value: "qnn", label: "QNN · unavailable", disabled: true }, { value: "cuda", label: "CUDA · GPU", disabled: !["auto", "gpu"].includes(s.device) }, { value: "dml", label: "DirectML · GPU", disabled: !["auto", "gpu"].includes(s.device) }, { value: "cpu", label: "CPU", disabled: !["auto", "cpu"].includes(s.device) }]} /></div>
        <div className="space-y-1.5"><label htmlFor="power-mode" className="text-sm font-medium">Power mode</label><Select id="power-mode" value={s.powerMode} onValueChange={powerMode => s.updateSettings({ powerMode: powerMode as ConversionRequest["powerMode"] })} options={[{ value: "automatic", label: "Automatic" }, { value: "max-performance", label: "Max performance · unavailable", disabled: true }, { value: "battery-efficient", label: "Battery efficient · unavailable", disabled: true }]} /><p className="text-xs text-muted-foreground">Uses the system power policy. The app does not change Windows power settings.</p></div>
        <PathInput label="Output directory" value={s.outputDir} onChange={outputDir => s.updateSettings({ outputDir })} directory onError={setError} placeholder="Default app artifacts directory" />
        <PathInput label="Calibration directory" value={s.calibrationDir} onChange={calibrationDir => s.updateSettings({ calibrationDir })} directory onError={setError} />
        <PathInput label="Evaluation NPZ (required)" value={s.evalNpz} onChange={evalNpz => s.updateSettings({ evalNpz })} extension="npz" onError={setError} />
        <div className="space-y-1.5"><label htmlFor="max-degradation" className="text-sm font-medium">Maximum degradation (0–1)</label><Input id="max-degradation" type="number" min="0" max="1" step="0.001" value={s.maxDegradation} onChange={e => s.updateSettings({ maxDegradation: e.target.valueAsNumber })} /></div>
        <div className="space-y-1.5"><label htmlFor="max-samples" className="text-sm font-medium">Maximum calibration samples</label><Input id="max-samples" type="number" min="1" max="100000" value={s.maxSamples} onChange={e => s.updateSettings({ maxSamples: e.target.valueAsNumber })} /></div>
        <div className="grid grid-cols-2 gap-3"><div className="space-y-1.5"><label htmlFor="conversion-iterations" className="text-sm font-medium">Benchmark iterations</label><Input id="conversion-iterations" type="number" min="1" max="100000" value={s.iterations} onChange={e => s.updateSettings({ iterations: e.target.valueAsNumber })} /></div><div className="space-y-1.5"><label htmlFor="conversion-warmup" className="text-sm font-medium">Warmup runs</label><Input id="conversion-warmup" type="number" min="0" max="10000" value={s.warmup} onChange={e => s.updateSettings({ warmup: e.target.valueAsNumber })} /></div></div>
      </div>
      <p className="text-xs text-muted-foreground">Use real calibration and evaluation inputs matching the model. Synthetic inputs cannot establish accuracy. Each conversion writes to its own job directory.</p>
      <div className="space-y-3 border-t pt-4">
        <h3 className="text-sm font-semibold">Recipes</h3>
        <div className="grid gap-3 md:grid-cols-2"><div className="space-y-1.5"><label htmlFor="shipped-recipe" className="text-sm font-medium">Shipped recipe</label><Select id="shipped-recipe" value="" onValueChange={applyShipped} disabled={!recipes.data?.length || busy} options={[{ value: "", label: recipes.isFetching ? "Loading recipes…" : "Choose a recipe…" }, ...(recipes.data ?? []).map(recipe => ({ value: recipe.name, label: recipe.name }))]} /></div>
          <div className="space-y-1.5"><label htmlFor="recipe-name" className="text-sm font-medium">Recipe name</label><Input id="recipe-name" value={recipeName} onChange={e => setRecipeName(e.target.value)} placeholder="My conversion" /></div>
        </div>
        <div className="space-y-1.5"><label htmlFor="recipe-path" className="text-sm font-medium">Recipe JSON path</label><Input id="recipe-path" value={recipePath} onChange={e => setRecipePath(e.target.value)} placeholder="Leave empty to choose a file" /></div>
        <div className="flex flex-wrap gap-2"><Button variant="outline" disabled={busy || !isTauriRuntime()} onClick={() => void recipeAction("load")}>Load recipe</Button><Button variant="outline" disabled={busy || !isTauriRuntime()} onClick={() => void recipeAction("save")}>Save recipe</Button></div>
        {recipes.isError && <p role="alert" className="text-sm text-destructive">Could not load shipped recipes: {errorMessage(recipes.error)}</p>}
      </div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}{notice && <p role="status" className="text-sm break-all">{notice}</p>}
    </CardContent>
  </Card>;
}
