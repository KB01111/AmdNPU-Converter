import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cancelJob, errorMessage, getJobStatus, isTauriRuntime, listJobs, listModels, openModelFolder, retryJob } from "@/lib/tauri";
import { isTerminal, useAppStore } from "@/stores/useAppStore";

export default function ModelLibrary() {
  const s = useAppStore();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const models = useQuery({ queryKey: ["models"], queryFn: listModels, enabled: isTauriRuntime(), retry: false, staleTime: 10000 });
  async function refresh() {
    setLoadingJobs(true); setError(null);
    try { const jobs = await listJobs(); jobs.forEach(s.upsertJob); await models.refetch(); }
    catch (error) { setError(errorMessage(error)); } finally { setLoadingJobs(false); }
  }
  async function jobAction(id: string, kind: "cancel" | "retry" | "folder") {
    setBusy(id); setError(null);
    try {
      if (kind === "folder") await openModelFolder(id);
      else if (kind === "retry") { const job = await retryJob(id); s.upsertJob(job); s.openJob(job.jobId); }
      else { await cancelJob(id); s.upsertJob(await getJobStatus(id)); }
    } catch (error) { setError(errorMessage(error)); } finally { setBusy(null); }
  }
  return <div className="space-y-4">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-lg font-semibold">Model library &amp; job history</h2><Button variant="outline" disabled={!isTauriRuntime() || loadingJobs || models.isFetching} onClick={() => void refresh()}>Refresh</Button></div>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {models.isError && <p role="alert" className="text-sm text-destructive">Could not load model library: {errorMessage(models.error)}</p>}
    <Card><CardHeader><CardTitle>Conversion jobs ({s.jobs.length})</CardTitle><CardDescription>Open a job to inspect its stages, logs and diagnostics in Pipeline.</CardDescription></CardHeader><CardContent>
      {loadingJobs && <p role="status" className="text-sm">Refreshing jobs…</p>}
      {!s.jobs.length ? <p className="text-sm text-muted-foreground">No conversion jobs yet. Start a conversion from Import.</p> : <div className="overflow-x-auto"><table aria-label="Conversion jobs" className="w-full text-left text-sm"><thead><tr className="border-b"><th className="p-2">Model</th><th className="p-2">Status</th><th className="p-2">Stage</th><th className="p-2">Progress</th><th className="p-2">Actions</th></tr></thead><tbody>{s.jobs.map(job => <tr key={job.jobId} className="border-b"><th scope="row" className="p-2 font-medium"><span className="block">{job.modelId}</span><span className="font-mono text-xs text-muted-foreground">{job.jobId.slice(0, 8)}</span></th><td className="p-2"><Badge variant={job.status === "failed" || job.status === "interrupted" ? "destructive" : job.status === "completed" ? "success" : "secondary"}>{job.status}</Badge></td><td className="p-2">{job.currentStep}</td><td className="p-2">{Math.round(job.progress)}%</td><td className="p-2"><div className="flex gap-2"><Button variant="outline" size="sm" onClick={() => s.openJob(job.jobId)}>Open</Button>{!isTerminal(job.status) && <Button variant="ghost" size="sm" disabled={busy === job.jobId} onClick={() => void jobAction(job.jobId, "cancel")}>Cancel</Button>}{isTerminal(job.status) && job.request && <Button variant="ghost" size="sm" disabled={busy === job.jobId} onClick={() => void jobAction(job.jobId, "retry")}>Retry</Button>}</div></td></tr>)}</tbody></table></div>}
    </CardContent></Card>
    <Card><CardHeader><CardTitle>Converted models ({models.data?.length ?? 0})</CardTitle><CardDescription>Open artifact folders or select an artifact for benchmarking.</CardDescription></CardHeader><CardContent className="space-y-3">
      {models.isFetching && !models.data ? <p role="status" className="text-sm">Loading converted models…</p> : !models.data?.length ? <p className="text-sm text-muted-foreground">No converted models reported yet.</p> : models.data.map(model => {
        const producer = s.jobs.find(job => job.jobId === model.jobId)
          ?? s.jobs.find(job => job.artifacts.some(path => path === model.onnxPath || path === model.compiledPath));
        const jobId = model.jobId ?? producer?.jobId ?? null;
        const targetEp = model.targetEp ?? producer?.request?.targetEp ?? null;
        return <div key={model.modelId} className="space-y-3 rounded-lg border p-3 text-sm">
          <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="break-all font-semibold">{model.hfId || model.modelId}</h3><Badge variant="outline">{model.format}</Badge></div>
          <p className="text-muted-foreground">{model.arch} · {model.params} parameters · {model.createdAt}</p>
          <dl className="space-y-1 text-xs">{[["Folder", model.localPath], ["ONNX", model.onnxPath], ["Compiled", model.compiledPath]].map(([label, path]) => <div key={label}><dt className="font-medium">{label}</dt><dd className="break-all font-mono text-muted-foreground">{path || "Not reported"}</dd></div>)}</dl>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" disabled={busy === model.modelId || !isTauriRuntime()} onClick={() => void jobAction(model.modelId, "folder")}>Open artifact folder</Button>
            {model.onnxPath && <Button size="sm" variant="outline" onClick={() => s.benchmarkArtifact(model.onnxPath, jobId, targetEp)}>Benchmark ONNX</Button>}
            {model.compiledPath && <Button size="sm" variant="outline" onClick={() => s.benchmarkArtifact(model.compiledPath, jobId, targetEp)}>Benchmark compiled</Button>}
            {jobId && <Button size="sm" variant="ghost" onClick={() => s.openJob(jobId)}>Open conversion</Button>}
          </div>
        </div>;
      })}
    </CardContent></Card>
  </div>;
}
