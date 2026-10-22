import { create } from "zustand";
import type { AppTab, BenchmarkRequest, BenchmarkResult, ConversionRequest, HardwareInfo, JobProgressEvent, JobStatus, ModelMetadata, Recipe, TargetEp } from "@/lib/types";

export const isTerminal = (status: string) => ["completed", "failed", "cancelled", "interrupted"].includes(status);
export const benchmarkDeviceForEp = (ep: TargetEp | null): string | null => ep === "vitisai" ? "npu" : ep === "dml" || ep === "cuda" ? "gpu" : ep === "cpu" ? "cpu" : null;

export interface Settings {
  precision: ConversionRequest["precision"]; device: ConversionRequest["device"];
  powerMode: ConversionRequest["powerMode"]; targetEp: ConversionRequest["targetEp"];
  outputDir: string; calibrationDir: string; evalNpz: string;
  maxDegradation: number; maxSamples: number; iterations: number; warmup: number;
}
const defaults: Settings = { precision: "auto", device: "npu", powerMode: "automatic", targetEp: "vitisai", outputDir: "", calibrationDir: "", evalNpz: "", maxDegradation: 0.02, maxSamples: 64, iterations: 100, warmup: 10 };
interface AppState extends Settings {
  tab: AppTab; sourceKind: "hf" | "onnx"; huggingfaceUrl: string; onnxPath: string; localModelId: string;
  sourceRevision: number; modelMetadata: ModelMetadata | null; hardwareInfo: HardwareInfo | null;
  jobs: JobStatus[]; activeJobId: string | null; jobError: string | null;
  benchmarkPath: string; benchmarkJobId: string | null; benchmarkTargetEp: TargetEp | null; benchmarkDevices: string[];
  benchmarkIterations: number; benchmarkWarmup: number; benchmarkResults: BenchmarkResult[];
  benchmarkRequest: BenchmarkRequest | null; benchmarkRunning: boolean; benchmarkError: string | null;
  setTab: (tab: AppTab) => void;
  setSourceKind: (kind: "hf" | "onnx") => void;
  setHuggingfaceUrl: (url: string) => void; setOnnxPath: (path: string) => void;
  setLocalModelId: (id: string) => void;
  acceptInspection: (revision: number, data: ModelMetadata) => void;
  setHardwareInfo: (hardware: HardwareInfo) => void;
  updateSettings: (settings: Partial<Settings>) => void;
  setJobs: (jobs: JobStatus[]) => void; upsertJob: (job: JobStatus) => void;
  mergeJobEvent: (event: JobProgressEvent) => void;
  openJob: (jobId: string) => void;
  benchmarkArtifact: (path: string, jobId?: string | null, targetEp?: TargetEp | null) => void;
  applyRecipe: (recipe: Recipe) => void;
}

export const useAppStore = create<AppState>((set) => ({
  ...defaults, tab: "import", sourceKind: "hf", huggingfaceUrl: "", onnxPath: "", localModelId: "local-onnx-import",
  sourceRevision: 0, modelMetadata: null, hardwareInfo: null, jobs: [], activeJobId: null, jobError: null,
  benchmarkPath: "", benchmarkJobId: null, benchmarkTargetEp: null, benchmarkDevices: ["cpu", "npu"], benchmarkIterations: 100,
  benchmarkWarmup: 10, benchmarkResults: [], benchmarkRequest: null, benchmarkRunning: false, benchmarkError: null,
  setTab: (tab) => set({ tab }),
  setSourceKind: (sourceKind) => set(s => ({ sourceKind, modelMetadata: null, sourceRevision: s.sourceRevision + 1 })),
  setHuggingfaceUrl: (huggingfaceUrl) => set(s => ({ huggingfaceUrl, modelMetadata: null, sourceRevision: s.sourceRevision + 1 })),
  setOnnxPath: (onnxPath) => set(s => ({ onnxPath, modelMetadata: null, sourceRevision: s.sourceRevision + 1 })),
  setLocalModelId: (localModelId) => set({ localModelId }),
  acceptInspection: (revision, modelMetadata) => set(s => s.sourceRevision === revision && s.sourceKind === "hf" ? { modelMetadata } : {}),
  setHardwareInfo: (hardwareInfo) => set({ hardwareInfo }),
  updateSettings: (settings) => set(settings),
  setJobs: (jobs) => set(s => ({ jobs: jobs.map(job => preserveTerminal(s.jobs.find(j => j.jobId === job.jobId), job)) })),
  upsertJob: (job) => set(s => ({ jobs: s.jobs.some(j => j.jobId === job.jobId)
    ? s.jobs.map(j => j.jobId === job.jobId ? preserveTerminal(j, job) : j) : [job, ...s.jobs] })),
  mergeJobEvent: (event) => set(s => {
    const old = s.jobs.find(j => j.jobId === event.jobId) ?? { jobId: event.jobId, modelId: "Loading model…", status: "running", currentStep: "inspection", progress: 0, logs: [], artifacts: [] };
    const patch: Partial<JobStatus> = {};
    if (event.status !== undefined) patch.status = event.status;
    if (event.currentStep !== undefined) patch.currentStep = event.currentStep;
    if (event.progress !== undefined && Number.isFinite(event.progress)) patch.progress = Math.max(0, Math.min(100, event.progress));
    if (event.error !== undefined) patch.error = event.error;
    if (event.line !== undefined) patch.logs = [...old.logs, { ts: new Date().toISOString(), stream: event.stream ?? "stdout", line: event.line }].slice(-500);
    const job = preserveTerminal(old, { ...old, ...patch });
    return { jobs: s.jobs.some(j => j.jobId === job.jobId) ? s.jobs.map(j => j.jobId === job.jobId ? job : j) : [job, ...s.jobs] };
  }),
  openJob: (activeJobId) => set({ activeJobId, tab: "pipeline" }),
  benchmarkArtifact: (benchmarkPath, benchmarkJobId = null, benchmarkTargetEp = null) => set(s => {
    const device = benchmarkDeviceForEp(benchmarkTargetEp);
    return { benchmarkPath, benchmarkJobId, benchmarkTargetEp, tab: "benchmark",
      benchmarkDevices: device && !s.benchmarkDevices.includes(device) ? [...s.benchmarkDevices, device] : s.benchmarkDevices };
  }),
  applyRecipe: (recipe) => set(s => {
    if (recipe.recipe_version !== "1.0" || !recipe.conversion_request || !recipe.name) throw new Error("Unsupported recipe: expected version 1.0, name and conversion_request.");
    const r = recipe.conversion_request;
    return { ...defaults, precision: r.precision, device: r.device, powerMode: r.powerMode, targetEp: r.targetEp,
      outputDir: r.outputDir ?? "", sourceKind: r.sourceKind ?? recipe.source?.kind ?? "hf",
      huggingfaceUrl: r.url || r.modelId, onnxPath: r.onnxPath ?? recipe.source?.onnx_path ?? "", localModelId: r.modelId,
      calibrationDir: r.calibrationDir ?? recipe.quantization?.calibration?.dir ?? "", evalNpz: r.evalNpz ?? "",
      maxSamples: r.maxSamples ?? recipe.quantization?.calibration?.max_samples ?? defaults.maxSamples,
      maxDegradation: r.maxDegradation ?? recipe.quantization?.accuracy_gate?.max_degradation ?? defaults.maxDegradation,
      iterations: r.iterations ?? defaults.iterations, warmup: r.warmup ?? defaults.warmup,
      modelMetadata: null, sourceRevision: s.sourceRevision + 1 };
  }),
}));

function preserveTerminal(old: JobStatus | undefined, job: JobStatus): JobStatus {
  if (old) {
    const stale = (isTerminal(old.status) && !isTerminal(job.status))
      || (old.status === "cancelling" && ["running", "queued"].includes(job.status))
      || (old.status === "running" && job.status === "queued")
      || (!isTerminal(job.status) && job.progress < old.progress);
    if (stale) return { ...job, status: old.status, currentStep: old.currentStep,
      progress: old.progress, error: old.error, logs: old.logs, artifacts: old.artifacts,
      request: job.request ?? old.request };
  }
  return { ...job, logs: job.logs.slice(-500) };
}

export function conversionRequest(s: AppState): ConversionRequest {
  const modelId = s.sourceKind === "onnx" ? s.localModelId.trim() : s.modelMetadata?.modelId;
  if (!modelId) throw new Error("Inspect the current Hugging Face model before converting.");
  if (s.sourceKind === "onnx" && !s.onnxPath.trim().toLowerCase().endsWith(".onnx")) throw new Error("Choose a local .onnx file.");
  if (!s.evalNpz.trim()) throw new Error("A representative evaluation NPZ is required to validate the converted model.");
  if (["a16w8", "a8w8", "int8"].includes(s.precision) && !s.calibrationDir.trim()) throw new Error("This precision requires a representative calibration directory.");
  if (["fp32", "int4"].includes(s.precision)) throw new Error("This precision is not supported by the conversion worker.");
  if (s.powerMode !== "automatic") throw new Error("Only automatic power mode is supported.");
  if (s.targetEp === "qnn") throw new Error("QNN is not supported by this conversion pipeline.");
  const epDevice = { vitisai: "npu", qnn: "npu", cuda: "gpu", dml: "gpu", cpu: "cpu" };
  if (s.device !== "auto" && epDevice[s.targetEp] !== s.device) throw new Error("Choose a device that matches the target execution provider.");
  if (!Number.isFinite(s.maxDegradation) || s.maxDegradation < 0 || s.maxDegradation > 1) throw new Error("Maximum degradation must be between 0 and 1.");
  if (!Number.isInteger(s.maxSamples) || s.maxSamples < 1 || s.maxSamples > 100000) throw new Error("Calibration samples must be between 1 and 100000.");
  if (!Number.isInteger(s.iterations) || s.iterations < 1 || s.iterations > 100000 || !Number.isInteger(s.warmup) || s.warmup < 0 || s.warmup > 10000) throw new Error("Use 1–100000 iterations and 0–10000 warmup runs.");
  return { modelId, url: s.sourceKind === "hf" ? s.huggingfaceUrl.trim() : "", sourceKind: s.sourceKind,
    onnxPath: s.sourceKind === "onnx" ? s.onnxPath.trim() : null, precision: s.precision, device: s.device,
    powerMode: s.powerMode, targetEp: s.targetEp, outputDir: s.outputDir.trim() || null,
    calibrationDir: s.calibrationDir.trim() || null, evalNpz: s.evalNpz.trim(), maxDegradation: s.maxDegradation,
    maxSamples: s.maxSamples, iterations: s.iterations, warmup: s.warmup };
}
