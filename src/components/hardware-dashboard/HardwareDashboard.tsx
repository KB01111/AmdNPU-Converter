import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { errorMessage, getHardware, isTauriRuntime } from "@/lib/tauri";
import { useAppStore } from "@/stores/useAppStore";

export default function HardwareDashboard() {
  const hardware = useAppStore(s => s.hardwareInfo);
  const query = useQuery({ queryKey: ["hardware"], enabled: isTauriRuntime(), retry: false, staleTime: 30000,
    queryFn: async () => { const info = await getHardware(); useAppStore.getState().setHardwareInfo(info); return info; } });
  const info = query.data ?? hardware;
  return <div className="space-y-4">
    <div className="flex items-center justify-between gap-3"><h2 className="text-lg font-semibold">Hardware dashboard</h2><Button variant="outline" onClick={() => void query.refetch()} disabled={!isTauriRuntime() || query.isFetching}>{query.isFetching ? "Refreshing…" : "Refresh"}</Button></div>
    {query.isFetching && !info && <p role="status" className="text-sm">Reading hardware and provider information…</p>}
    {query.isError && <p role="alert" className="text-sm text-destructive">Failed to read hardware: {errorMessage(query.error)}</p>}
    {!info && !query.isFetching && <p className="text-sm text-muted-foreground">No hardware data yet. Open the desktop app and refresh to check local prerequisites.</p>}
    {info && <>
      {!!info.providerErrors?.length && <div role="alert" className="rounded-md border p-3 text-sm"><h3 className="font-semibold">Provider setup needs attention</h3>{info.providerErrors.map((message, index) => <p key={index} className="mt-1 whitespace-pre-wrap break-words">{message}</p>)}</div>}
      <div className="grid gap-4 sm:grid-cols-2">
        <Card><CardHeader><CardTitle>CPU</CardTitle></CardHeader><CardContent className="space-y-1 text-sm"><p className="font-medium">{info.cpu.brand}</p><p>{info.cpu.physicalCores} physical / {info.cpu.logicalCores} logical cores · {info.cpu.frequencyMhz} MHz</p><p className="text-muted-foreground">{info.cpu.vendor} · {info.totalMemoryGb.toFixed(1)} GB memory</p><p>{info.osName} {info.osVersion}</p></CardContent></Card>
        <Card><CardHeader><CardTitle>GPU</CardTitle></CardHeader><CardContent className="space-y-3 text-sm">{info.gpus.length ? info.gpus.map((gpu, index) => <div key={`${gpu.name}-${index}`}><p className="font-medium">{gpu.name}</p><p className="text-muted-foreground">{gpu.vendor} · driver {gpu.driverVersion || "unknown"}</p></div>) : <p>No GPU reported.</p>}</CardContent></Card>
        <Card><CardHeader><CardTitle>NPU</CardTitle></CardHeader><CardContent className="space-y-2 text-sm"><p>{info.npu.name || "No NPU reported"}</p><Badge variant={info.npu.present ? "secondary" : "warning"}>{info.npu.present ? "Detected" : "Not detected"}</Badge><p>Driver: {info.npu.driverVersion || "Unknown"}</p><p className="text-muted-foreground">Detection alone does not establish NPU execution or model compatibility.</p></CardContent></Card>
        <Card><CardHeader><CardTitle>WinML</CardTitle></CardHeader><CardContent className="space-y-2 text-sm"><Badge variant={info.winml.available ? "secondary" : "warning"}>{info.winml.available ? "Available" : "Unavailable"}</Badge><p>Version: {info.winml.version || "Unknown"}</p><p className="break-all font-mono text-xs">{info.winml.binaryPath || "No WinML executable found"}</p></CardContent></Card>
      </div>
      <Card><CardHeader><CardTitle>Execution providers and drivers</CardTitle><CardDescription>Providers available to the local runtime. Availability does not prove node placement.</CardDescription></CardHeader><CardContent className="space-y-4"><div className="flex flex-wrap gap-2">{info.executionProviders.length ? info.executionProviders.map(provider => <Badge key={provider} variant="outline">{provider}</Badge>) : <p className="text-sm">No execution providers reported. Check the Python worker environment.</p>}</div><dl className="grid gap-2 text-sm sm:grid-cols-2">{Object.entries(info.driverVersions).map(([name, version]) => <div key={name}><dt className="font-medium">{name}</dt><dd className="break-all text-muted-foreground">{version || "Unknown"}</dd></div>)}</dl></CardContent></Card>
      <Card><CardHeader><CardTitle>Setup readiness</CardTitle></CardHeader><CardContent className="space-y-2 text-sm"><p>{info.npu.present && info.winml.available ? "Hardware and WinML detected. Conversion still checks the selected provider, Python worker environments and representative datasets." : "Complete the missing prerequisites before attempting an NPU conversion."}</p>{!info.npu.present && <p>Install the supported Ryzen AI NPU driver and verify that Windows detects the device.</p>}{!info.winml.available && <p>Install WinML and configure its executable using the project setup instructions.</p>}<p>Follow README setup to bootstrap the separate exporter, Quark and WinML worker environments. Refresh after installing prerequisites.</p></CardContent></Card>
    </>}
  </div>;
}
