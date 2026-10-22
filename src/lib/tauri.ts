import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  BenchmarkRequest,
  BenchmarkResult,
  ConversionRequest,
  HardwareInfo,
  JobProgressEvent,
  JobStatus,
  ModelMetadata,
  ModelRecord,
} from "./types";

/**
 * Type-safe wrappers around the Rust Tauri commands.
 * Every function calls the real backend via `invoke` — no mocks.
 */

export async function inspectHfModel(url: string): Promise<ModelMetadata> {
  return invoke<ModelMetadata>("hf_inspect", { url });
}

export async function getHardware(): Promise<HardwareInfo> {
  return invoke<HardwareInfo>("get_hardware_info");
}

export type StartConversionResponse = string | { jobId: string } | { id: string };

function normalizeJobId(res: StartConversionResponse): string {
  if (typeof res === "string") return res;
  if (typeof res === "object" && res !== null) {
    if ("jobId" in res && typeof res.jobId === "string") return res.jobId;
    if ("id" in res && typeof res.id === "string") return res.id;
  }
  throw new Error("Unexpected start_conversion response shape");
}

export async function startConversion(req: ConversionRequest): Promise<string> {
  const raw = await invoke<StartConversionResponse>("start_conversion", { req });
  return normalizeJobId(raw);
}

export async function getJobStatus(id: string): Promise<JobStatus> {
  return invoke<JobStatus>("get_job_status", { id });
}

export async function listJobs(): Promise<JobStatus[]> {
  return invoke<JobStatus[]>("list_jobs");
}

export async function cancelJob(id: string): Promise<void> {
  await invoke<void>("cancel_job", { id });
}

export async function runBenchmark(req: BenchmarkRequest): Promise<BenchmarkResult[]> {
  return invoke<BenchmarkResult[]>("run_benchmark", { req });
}

export async function listModels(): Promise<ModelRecord[]> {
  return invoke<ModelRecord[]>("list_models");
}

export async function onJobProgress(
  handler: (payload: JobProgressEvent) => void,
): Promise<UnlistenFn> {
  return listen<JobProgressEvent>("job://progress", (event) => {
    handler(event.payload);
  });
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
