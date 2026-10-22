import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { startConversion, errorMessage, isTauriRuntime } from "@/lib/tauri";
import { conversionRequest, useAppStore } from "@/stores/useAppStore";

export default function ModelInspector() {
  const s = useAppStore();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  let validation = "";
  try { conversionRequest(s); } catch (error) { validation = errorMessage(error); }
  async function convert() {
    setPending(true); setError(null);
    try {
      const job = await startConversion(conversionRequest(useAppStore.getState()));
      s.upsertJob(job); s.openJob(job.jobId);
    } catch (error) { setError(errorMessage(error)); } finally { setPending(false); }
  }
  return <Card>
    <CardHeader><CardTitle>Start conversion</CardTitle><CardDescription>{s.sourceKind === "onnx" ? s.onnxPath || "Select a local ONNX file." : s.modelMetadata?.modelId || "Inspect the current Hugging Face model first."}</CardDescription></CardHeader>
    <CardContent className="space-y-3">
      <div className="flex flex-wrap gap-2"><Badge variant="outline">{s.precision} · {s.device} · {s.targetEp}</Badge><Badge variant="secondary">NPU placement requires execution evidence</Badge></div>
      {s.hardwareInfo && <p className="text-sm text-muted-foreground">NPU {s.hardwareInfo.npu.present ? "detected" : "not detected"} · WinML {s.hardwareInfo.winml.available ? "available" : "unavailable"}. Check Hardware for prerequisites.</p>}
      {validation && <p className="text-sm text-muted-foreground">{validation}</p>}
      {error && <p role="alert" className="text-sm text-destructive">Conversion could not start: {error}</p>}
      <Button onClick={() => void convert()} disabled={pending || !!validation || !isTauriRuntime()}>{pending ? "Starting…" : "Start conversion"}</Button>
    </CardContent>
  </Card>;
}
