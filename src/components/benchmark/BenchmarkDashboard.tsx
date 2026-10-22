import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PathInput } from "@/components/ui/path-input";
import { Select } from "@/components/ui/select";
import { chooseSavePath, errorMessage, exportReport, isTauriRuntime, runBenchmark } from "@/lib/tauri";
import { benchmarkDeviceForEp, useAppStore } from "@/stores/useAppStore";
import type { BenchmarkRequest, TargetEp } from "@/lib/types";

const metric = (value?: number | null) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
export default function BenchmarkDashboard() {
  const s = useAppStore();
  const [notice, setNotice] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const pinnedDevice = benchmarkDeviceForEp(s.benchmarkTargetEp);
  async function run() {
    const current = useAppStore.getState();
    if (!current.benchmarkPath.trim() || !current.benchmarkDevices.length) return;
    if (!Number.isInteger(current.benchmarkIterations) || current.benchmarkIterations < 1 || current.benchmarkIterations > 100000 || !Number.isInteger(current.benchmarkWarmup) || current.benchmarkWarmup < 0 || current.benchmarkWarmup > 10000) {
      useAppStore.setState({ benchmarkError: "Use 1–100000 iterations and 0–10000 warmup runs." }); return;
    }
    const request: BenchmarkRequest = { modelPath: current.benchmarkPath.trim(), devices: [...current.benchmarkDevices], iterations: current.benchmarkIterations, warmup: current.benchmarkWarmup, jobId: current.benchmarkJobId, ...(current.benchmarkTargetEp ? { targetEp: current.benchmarkTargetEp } : {}) };
    useAppStore.setState({ benchmarkRunning: true, benchmarkError: null, benchmarkResults: [], benchmarkRequest: request });
    try { useAppStore.setState({ benchmarkResults: await runBenchmark(request) }); }
    catch (error) { useAppStore.setState({ benchmarkError: errorMessage(error) }); }
    finally { useAppStore.setState({ benchmarkRunning: false }); }
  }
  async function exportResults() {
    setExporting(true); setNotice(null);
    try { const path = await chooseSavePath(); if (path) { await exportReport(path, JSON.stringify({ request: s.benchmarkRequest, results: s.benchmarkResults, error: s.benchmarkError }, null, 2)); setNotice(`Benchmark report saved to ${path}`); } }
    catch (error) { setNotice(`Failed to export benchmark report: ${errorMessage(error)}`); }
    finally { setExporting(false); }
  }
  return <div className="space-y-4">
    <Card><CardHeader><CardTitle>Benchmark</CardTitle><CardDescription>Compare actual execution providers on an ONNX or EPContext artifact. Unsupported devices report their own errors.</CardDescription></CardHeader><CardContent className="space-y-4">
      <PathInput label="Model artifact path" value={s.benchmarkPath} onChange={benchmarkPath => useAppStore.setState({ benchmarkPath, benchmarkJobId: null, benchmarkTargetEp: null })} disabled={s.benchmarkRunning} onError={benchmarkError => useAppStore.setState({ benchmarkError })} />
      <div className="max-w-lg space-y-1.5"><label htmlFor="benchmark-provider" className="text-sm font-medium">Execution provider</label><Select id="benchmark-provider" disabled={s.benchmarkRunning} value={s.benchmarkTargetEp ?? "auto"} onValueChange={value => {
        const benchmarkTargetEp = value === "auto" ? null : value as TargetEp;
        const device = benchmarkDeviceForEp(benchmarkTargetEp);
        useAppStore.setState({ benchmarkTargetEp, benchmarkDevices: device && !s.benchmarkDevices.includes(device) ? [...s.benchmarkDevices, device] : s.benchmarkDevices });
      }} options={[{ value: "auto", label: "Auto (device defaults)" }, { value: "vitisai", label: "VitisAI · NPU" }, { value: "dml", label: "DirectML · GPU" }, { value: "cuda", label: "CUDA · GPU" }, { value: "cpu", label: "CPU" }]} /><p className="text-xs text-muted-foreground">The selected provider pins its matching device. Other comparison devices use their default providers.</p></div>
      <fieldset disabled={s.benchmarkRunning} className="space-y-2"><legend className="text-sm font-medium">Benchmark devices</legend><div className="flex flex-wrap gap-4">{["cpu", "gpu", "npu"].map(device => <label key={device} className="flex items-center gap-2 text-sm"><input type="checkbox" disabled={device === pinnedDevice} checked={s.benchmarkDevices.includes(device)} onChange={e => useAppStore.setState({ benchmarkDevices: e.target.checked ? [...s.benchmarkDevices, device] : s.benchmarkDevices.filter(d => d !== device) })} />{device.toUpperCase()}</label>)}</div></fieldset>
      <div className="grid max-w-lg grid-cols-2 gap-3"><div className="space-y-1.5"><label htmlFor="benchmark-iterations" className="text-sm font-medium">Iterations</label><Input id="benchmark-iterations" type="number" min="1" max="100000" disabled={s.benchmarkRunning} value={s.benchmarkIterations} onChange={e => useAppStore.setState({ benchmarkIterations: e.target.valueAsNumber })} /></div><div className="space-y-1.5"><label htmlFor="benchmark-warmup" className="text-sm font-medium">Warmup</label><Input id="benchmark-warmup" type="number" min="0" max="10000" disabled={s.benchmarkRunning} value={s.benchmarkWarmup} onChange={e => useAppStore.setState({ benchmarkWarmup: e.target.valueAsNumber })} /></div></div>
      <Button disabled={!isTauriRuntime() || s.benchmarkRunning || !s.benchmarkPath.trim() || !s.benchmarkDevices.length} onClick={() => void run()}>{s.benchmarkRunning ? "Benchmark running…" : "Run benchmark"}</Button>
      {s.benchmarkRunning && <p role="status" className="text-sm text-muted-foreground">Benchmarking selected devices. You can switch pages while it runs.</p>}
      {s.benchmarkError && <p role="alert" className="whitespace-pre-wrap text-sm text-destructive">Benchmark failed: {s.benchmarkError}</p>}
    </CardContent></Card>
    {!!s.benchmarkResults.length && <Card><CardHeader><CardTitle>Results</CardTitle><CardDescription className="break-all">{s.benchmarkRequest?.modelPath} · {s.benchmarkRequest?.iterations} iterations · {s.benchmarkRequest?.warmup} warmup runs</CardDescription></CardHeader><CardContent className="space-y-4">
      <div className="overflow-x-auto"><table className="w-full text-left text-sm"><caption className="sr-only">Benchmark results by device</caption><thead><tr className="border-b"><th className="p-2">Device</th><th className="p-2">Mean (ms)</th><th className="p-2">p50 (ms)</th><th className="p-2">p90 (ms)</th><th className="p-2">Throughput (inferences/s)</th><th className="p-2">Process memory (MB)</th></tr></thead><tbody>{s.benchmarkResults.map(result => <tr key={result.device} className="border-b"><th scope="row" className="p-2">{result.device.toUpperCase()}</th><td className="p-2">{metric(result.avgLatencyMs)}</td><td className="p-2">{metric(result.p50Ms)}</td><td className="p-2">{metric(result.p90Ms)}</td><td className="p-2">{metric(result.throughputPerS)}</td><td className="p-2">{metric(result.memoryMb)}</td></tr>)}</tbody></table></div>
      {s.benchmarkResults.map(result => <div key={result.device} className="space-y-1 rounded border p-3 text-sm"><h3 className="font-semibold">{result.device.toUpperCase()}</h3><p className="break-all">Providers: {result.providers.length ? result.providers.join(", ") : "None reported"}</p>{result.error && <p role="alert" className="whitespace-pre-wrap text-destructive">{result.error}</p>}{result.device === "npu" && <p>{result.npuVerified ? "NPU node execution verified by runtime evidence." : "NPU placement unverified. Provider availability does not establish NPU node execution."}</p>}</div>)}
      <Button variant="outline" disabled={exporting || !isTauriRuntime()} onClick={() => void exportResults()}>Export benchmark report</Button>
      {notice && <p role="status" className="text-sm break-all">{notice}</p>}
    </CardContent></Card>}
  </div>;
}
