import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { PathInput } from "@/components/ui/path-input";
import { inspectHfModel, isTauriRuntime, errorMessage } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";

export function validateHfUrl(input: string): { owner: string; repo: string } | null {
  const path = input.trim().replace(/^\/+|\/+$/g, "").replace(/^(?:https?:\/\/)?huggingface\.co\//, "");
  const segments = path.split("/");
  if (segments.length < 2 || segments.some(part => !part || part === "." || part === "..") ||
      (segments.length > 2 && !["tree", "blob", "resolve"].includes(segments[2]))) return null;
  const [owner, repo] = segments;
  return [owner, repo].every(part => /^[A-Za-z0-9._-]{1,96}$/.test(part)) ? { owner, repo } : null;
}
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** i).toFixed(2)} ${units[i]}`;
}

export default function HuggingFaceImport() {
  const s = useAppStore();
  const [pendingRevision, setPendingRevision] = useState<number | null>(null);
  const [failure, setFailure] = useState<{ revision: number; message: string } | null>(null);
  const pending = pendingRevision === s.sourceRevision;
  const error = failure?.revision === s.sourceRevision ? failure.message : null;
  async function inspect() {
    const current = useAppStore.getState();
    const parsed = validateHfUrl(current.huggingfaceUrl);
    if (!parsed) { setFailure({ revision: current.sourceRevision, message: "Enter owner/model or https://huggingface.co/owner/model." }); return; }
    const revision = current.sourceRevision;
    setPendingRevision(revision); setFailure(null);
    try { const metadata = await inspectHfModel(`${parsed.owner}/${parsed.repo}`); current.acceptInspection(revision, metadata); }
    catch (error) { setFailure({ revision, message: errorMessage(error) }); }
    finally { setPendingRevision(previous => previous === revision ? null : previous); }
  }
  function preset(modelId: string) {
    s.setSourceKind("hf"); s.setHuggingfaceUrl(modelId);
    s.updateSettings({ precision: modelId.endsWith("forms") ? "a16w8" : "bf16", device: "npu", targetEp: "vitisai", powerMode: "automatic" });
  }
  return <Card>
    <CardHeader><CardTitle>Import model</CardTitle><CardDescription>Inspect a Hugging Face model or convert an existing local ONNX file.</CardDescription></CardHeader>
    <CardContent className="space-y-4">
      <div className="max-w-sm space-y-1.5"><label htmlFor="source-kind" className="text-sm font-medium">Model source</label>
        <Select id="source-kind" value={s.sourceKind} onValueChange={value => s.setSourceKind(value as "hf" | "onnx")} options={[{ value: "hf", label: "Hugging Face" }, { value: "onnx", label: "Local ONNX" }]} />
      </div>
      {s.sourceKind === "hf" ? <>
        <div className="space-y-1.5"><label htmlFor="hf-source" className="text-sm font-medium">Hugging Face model ID or URL</label>
          <div className="flex flex-col gap-2 sm:flex-row"><Input id="hf-source" value={s.huggingfaceUrl} onChange={e => s.setHuggingfaceUrl(e.target.value)} placeholder="owner/model" onKeyDown={e => { if (e.key === "Enter" && !pending) void inspect(); }} />
            <Button onClick={() => void inspect()} disabled={!isTauriRuntime() || pending || !s.huggingfaceUrl.trim()}>{pending ? "Inspecting…" : "Inspect"}</Button>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2"><span className="text-sm text-muted-foreground">CUA-S1 presets</span>
          <Button variant="outline" size="sm" onClick={() => preset("cua-ai/cua-s1-nano-0.1")}>Nano text · BF16</Button>
          <Button variant="outline" size="sm" onClick={() => preset("cua-ai/cua-s1-forms")}>Forms · A16W8</Button>
        </div>
        <p className="text-xs text-muted-foreground">CUA-S1 uses its dedicated exporter. Model access and representative evaluation inputs are required.</p>
      </> : <>
        <PathInput label="Local ONNX path" value={s.onnxPath} onChange={s.setOnnxPath} extension="onnx" onError={message => setFailure({ revision: s.sourceRevision, message })} />
        <div className="space-y-1.5"><label htmlFor="local-model-id" className="text-sm font-medium">Local model name</label><Input id="local-model-id" value={s.localModelId} onChange={e => s.setLocalModelId(e.target.value)} /></div>
        <p className="text-xs text-muted-foreground">The pipeline validates ONNX and prepares opset 17 before optimization. The original file is retained.</p>
      </>}
      {pending && <p role="status" className="text-sm text-muted-foreground">Inspecting model metadata…</p>}
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
      {s.modelMetadata && <div className="space-y-3 rounded-md border p-3 text-sm">
        <h3 className="font-semibold break-all">{s.modelMetadata.modelId}</h3>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[["Architecture", s.modelMetadata.arch], ["Parameters", s.modelMetadata.params], ["Format", s.modelMetadata.format], ["Estimated size", formatBytes(s.modelMetadata.estimatedBytes)], ["License", s.modelMetadata.license], ["Export", s.modelMetadata.compat.exportStrategy], ["Opset", s.modelMetadata.compat.opset], ["Inputs", s.modelMetadata.inputTypes.join(", ")]].map(([label, value]) => <div key={label}><dt className="text-muted-foreground">{label}</dt><dd className="break-words">{value || "Unknown"}</dd></div>)}</dl>
        <details><summary className="cursor-pointer">Model files ({s.modelMetadata.files.length})</summary><ul className="mt-2 max-h-48 overflow-auto font-mono text-xs">{s.modelMetadata.files.map(file => <li key={file.rfilename} className="break-all py-1">{file.rfilename}{file.size != null ? ` · ${formatBytes(file.size)}` : ""}</li>)}</ul></details>
      </div>}
    </CardContent>
  </Card>;
}
