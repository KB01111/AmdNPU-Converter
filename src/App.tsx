import { lazy, Suspense, useEffect } from "react";
import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { Cpu, Download, GitBranch, FlaskConical, Library } from "lucide-react";
import HuggingFaceImport from "@/features/huggingface/HuggingFaceImport";
import ConversionSettings from "@/features/conversion/ConversionSettings";
import ModelInspector from "@/components/model-inspector/ModelInspector";
import { errorMessage, getJobStatus, isTauriRuntime, listJobs, onJobProgress } from "@/lib/tauri";
import { isTerminal, useAppStore } from "@/stores/useAppStore";
import type { AppTab, JobStatus } from "@/lib/types";

const ConversionPipeline = lazy(() => import("@/features/conversion/ConversionPipeline"));
const HardwareDashboard = lazy(() => import("@/components/hardware-dashboard/HardwareDashboard"));
const BenchmarkDashboard = lazy(() => import("@/components/benchmark/BenchmarkDashboard"));
const ModelLibrary = lazy(() => import("@/components/model-library/ModelLibrary"));
const queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } });
const TABS = [
  { id: "import", label: "Import", icon: Download }, { id: "pipeline", label: "Pipeline", icon: GitBranch },
  { id: "hardware", label: "Hardware", icon: Cpu }, { id: "benchmark", label: "Benchmark", icon: FlaskConical },
  { id: "library", label: "Library", icon: Library },
] as const;

/** Keep events and recovery polling alive across page navigation. */
function JobBridge() {
  const client = useQueryClient();
  useEffect(() => {
    if (!isTauriRuntime()) return;
    let disposed = false;
    let unlisten: (() => void) | undefined;
    let polling = false;
    function accept(job: JobStatus) {
      if (disposed) return;
      const previous = useAppStore.getState().jobs.find(j => j.jobId === job.jobId);
      useAppStore.getState().upsertJob(job);
      if (job.status === "completed" && previous?.status !== "completed") void client.invalidateQueries({ queryKey: ["models"] });
    }
    void onJobProgress(event => {
      if (disposed) return;
      const known = useAppStore.getState().jobs.some(job => job.jobId === event.jobId && job.modelId !== "Loading model…");
      useAppStore.getState().mergeJobEvent(event);
      if (!known || (event.status && isTerminal(event.status))) void getJobStatus(event.jobId).then(accept).catch(error => {
        if (!disposed) useAppStore.setState({ jobError: errorMessage(error) });
      });
      if (event.status === "completed") void client.invalidateQueries({ queryKey: ["models"] });
    }).then(stop => { if (disposed) stop(); else unlisten = stop; }).catch(error => {
      if (!disposed) useAppStore.setState({ jobError: `Live job events unavailable; polling for updates. ${errorMessage(error)}` });
    });
    void listJobs().then(jobs => { if (!disposed) jobs.forEach(accept); }).catch(error => {
      if (!disposed) useAppStore.setState({ jobError: `Could not load job history: ${errorMessage(error)}` });
    });
    async function poll() {
      if (polling || disposed) return;
      polling = true;
      try {
        const state = useAppStore.getState();
        const ids = state.jobs.filter(job => !isTerminal(job.status)).map(job => job.jobId);
        if (state.activeJobId && !state.jobs.some(job => job.jobId === state.activeJobId)) ids.push(state.activeJobId);
        const results = await Promise.allSettled(ids.map(async id => accept(await getJobStatus(id))));
        const failure = results.find(result => result.status === "rejected");
        if (!disposed && failure?.status === "rejected") useAppStore.setState({ jobError: `Job refresh failed: ${errorMessage(failure.reason)}` });
      } finally { polling = false; }
    }
    const timer = window.setInterval(() => void poll(), 1500);
    return () => { disposed = true; unlisten?.(); window.clearInterval(timer); };
  }, [client]);
  return null;
}

function Workspace() {
  const tab = useAppStore(s => s.tab);
  const setTab = useAppStore(s => s.setTab);
  const jobError = useAppStore(s => s.jobError);
  return <div className="min-h-screen bg-background text-foreground">
    <JobBridge />
    <header className="border-b"><div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-2"><span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary font-bold text-white" aria-hidden="true">N</span><div><h1 className="text-base font-semibold leading-tight">AMD NPU Model Converter</h1><p className="text-xs text-muted-foreground">Import, convert and evaluate local model artifacts</p></div></div>
      <nav aria-label="Main navigation" className="flex max-w-full gap-1 overflow-x-auto rounded-lg bg-muted p-1">{TABS.map(item => <button key={item.id} type="button" aria-current={tab === item.id ? "page" : undefined} onClick={() => setTab(item.id as AppTab)} className={`flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${tab === item.id ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"}`}><item.icon className="h-4 w-4" aria-hidden="true" />{item.label}</button>)}</nav>
    </div></header>
    <main className="mx-auto max-w-6xl space-y-4 px-4 py-6">
      {!isTauriRuntime() && <div role="status" className="rounded-lg border bg-muted p-4 text-sm"><h2 className="font-semibold">Desktop runtime required</h2><p className="mt-1">This browser preview lets you explore the interface. Open the desktop app to inspect models, choose local files, read hardware, run conversions and benchmarks, or save recipes and reports.</p><p className="mt-1 text-muted-foreground">For local development, follow the README setup, bootstrap the Python workers and launch the Tauri app.</p></div>}
      {jobError && <div role="alert" className="flex items-start justify-between gap-3 rounded-md border p-3 text-sm"><p>{jobError}</p><button type="button" className="underline" onClick={() => useAppStore.setState({ jobError: null })}>Dismiss</button></div>}
      <Suspense fallback={<p role="status" className="text-sm text-muted-foreground">Loading page…</p>}>
        {tab === "import" && <div className="space-y-4"><HuggingFaceImport /><ConversionSettings /><ModelInspector /></div>}
        {tab === "pipeline" && <ConversionPipeline />}
        {tab === "hardware" && <HardwareDashboard />}
        {tab === "benchmark" && <BenchmarkDashboard />}
        {tab === "library" && <ModelLibrary />}
      </Suspense>
    </main>
  </div>;
}
export default function App() { return <QueryClientProvider client={queryClient}><Workspace /></QueryClientProvider>; }
