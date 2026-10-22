import { useMemo, useState } from "react";
import { ReactFlow, Background, Controls, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { cancelJob, chooseSavePath, errorMessage, exportReport, getJobStatus, isTauriRuntime, retryJob } from "@/lib/tauri";
import { isTerminal, useAppStore } from "@/stores/useAppStore";

export default function ConversionPipeline() {
  const s = useAppStore();
  const job = s.jobs.find(job => job.jobId === s.activeJobId) ?? null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const stages = job?.request?.sourceKind === "onnx" ? ["inspection", "prepare", "optimize", "quantize", "validate", "compile", "benchmark"] : ["inspection", "download", "export", "optimize", "quantize", "validate", "compile", "benchmark"];
  const activeIndex = job?.currentStep === "done" ? stages.length : stages.indexOf(job?.currentStep ?? "");
  const nodes: Node[] = stages.map((stage, index) => {
    const completed = job?.status === "completed" || index < activeIndex;
    const active = index === activeIndex;
    const label = `${stage} · ${completed ? "done" : active ? job?.status ?? "pending" : "pending"}`;
    return { id: stage, position: { x: (index % 4) * 225, y: Math.floor(index / 4) * 110 }, data: { label }, style: { width: 200, borderColor: active && job?.status === "failed" ? "hsl(var(--destructive))" : completed || active ? "hsl(var(--primary))" : "hsl(var(--border))", background: "hsl(var(--card))", color: "hsl(var(--foreground))" } };
  });
  const edges: Edge[] = useMemo(() => stages.slice(1).map((stage, i) => ({ id: `${stages[i]}-${stage}`, source: stages[i], target: stage })), [stages.join(",")]);
  const logs = job?.logs.map(log => `[${log.ts}] ${log.stream}: ${log.line}`).join("\n") ?? "";
  async function action(kind: "cancel" | "retry" | "copy" | "export") {
    if (!job) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      if (kind === "cancel") { await cancelJob(job.jobId); s.upsertJob(await getJobStatus(job.jobId)); setNotice("Cancellation requested. The job stops after its worker exits."); }
      if (kind === "retry") { const next = await retryJob(job.jobId); s.upsertJob(next); s.openJob(next.jobId); }
      if (kind === "copy") { await navigator.clipboard.writeText(logs); setNotice("Logs copied."); }
      if (kind === "export") { const path = await chooseSavePath(); if (path) { await exportReport(path, JSON.stringify(job, null, 2)); setNotice(`Job report saved to ${path}`); } }
    } catch (error) { setError(errorMessage(error)); } finally { setBusy(false); }
  }
  return <div className="space-y-4">
    <Card><CardHeader><CardTitle>Conversion pipeline</CardTitle><CardDescription>Monitor stages, worker diagnostics and the artifacts produced by each job.</CardDescription></CardHeader><CardContent className="space-y-4">
      <div className="max-w-xl space-y-1.5"><label htmlFor="pipeline-job" className="text-sm font-medium">Conversion job</label><Select id="pipeline-job" value={s.activeJobId ?? ""} onValueChange={s.openJob} options={[{ value: "", label: "Choose a job…" }, ...s.jobs.map(item => ({ value: item.jobId, label: `${item.modelId} · ${item.status} · ${item.jobId.slice(0, 8)}` }))]} /></div>
      {!job ? <p className="text-sm text-muted-foreground">Start a conversion from Import or open an existing job from Library.</p> : <>
        <div className="space-y-2"><h3 className="break-all font-semibold">{job.modelId}</h3><div role="status" className="flex flex-wrap items-center gap-3 text-sm"><Badge variant={job.status === "failed" || job.status === "interrupted" ? "destructive" : job.status === "completed" ? "success" : "secondary"}>{job.status}</Badge><span>{job.currentStep} · {Math.round(job.progress)}%</span></div><Progress aria-label="Conversion progress" value={job.progress} /></div>
        <div className="flex flex-wrap gap-2">{!isTerminal(job.status) && <Button variant="destructive" disabled={busy || !isTauriRuntime()} onClick={() => void action("cancel")}>Cancel job</Button>}{isTerminal(job.status) && job.request && <Button disabled={busy || !isTauriRuntime()} onClick={() => void action("retry")}>Retry conversion</Button>}<Button variant="outline" disabled={busy || !logs} onClick={() => void action("copy")}>Copy logs</Button><Button variant="outline" disabled={busy || !isTauriRuntime()} onClick={() => void action("export")}>Export job report</Button></div>
        {job.error && <div role="alert" className="rounded-md border border-destructive p-3 text-sm"><h3 className="font-semibold">Job diagnostics</h3><p className="mt-1 whitespace-pre-wrap break-words">{job.error}</p></div>}
        <details><summary className="cursor-pointer text-sm font-medium">Stage details</summary><ol className="mt-2 grid gap-2 text-sm sm:grid-cols-2">{nodes.map(node => <li key={node.id}>{String(node.data.label)}</li>)}</ol></details>
      </>}
      <div className="h-[300px] w-full rounded-lg border" aria-label="Conversion stage graph"><ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false}><Background /><Controls showInteractive={false} /></ReactFlow></div>
      {error && <p role="alert" className="text-sm text-destructive">{error}</p>}{notice && <p role="status" className="text-sm break-all">{notice}</p>}
    </CardContent></Card>
    {job && <Card><CardHeader><CardTitle>Worker logs</CardTitle><CardDescription>The most recent 500 structured log lines.</CardDescription></CardHeader><CardContent><pre aria-label="Worker log output" className="max-h-80 overflow-auto whitespace-pre-wrap break-all rounded bg-muted p-3 font-mono text-xs">{logs || "No worker logs yet."}</pre></CardContent></Card>}
    {job && <Card><CardHeader><CardTitle>Artifacts</CardTitle></CardHeader><CardContent className="space-y-2">{job.artifacts.length ? job.artifacts.map(path => <div key={path} className="flex flex-wrap items-center justify-between gap-2 rounded border p-2"><code className="min-w-0 break-all text-xs">{path}</code>{/\.(onnx|bin)$/i.test(path) && <Button variant="outline" size="sm" onClick={() => s.benchmarkArtifact(path, job.jobId, job.request?.targetEp)}>Benchmark artifact</Button>}</div>) : <p className="text-sm text-muted-foreground">No artifacts reported for this job.</p>}</CardContent></Card>}
  </div>;
}
