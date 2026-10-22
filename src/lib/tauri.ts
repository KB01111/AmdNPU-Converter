import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import type { BenchmarkRequest, BenchmarkResult, ConversionRequest, HardwareInfo, JobProgressEvent, JobStatus, ModelMetadata, ModelRecord, Recipe } from "./types";

export const inspectHfModel = (url: string) => invoke<ModelMetadata>("hf_inspect", { url });
export const getHardware = () => invoke<HardwareInfo>("get_hardware_info");
export const startConversion = (request: ConversionRequest) => invoke<JobStatus>("start_conversion", { request });
export const getJobStatus = (jobId: string) => invoke<JobStatus>("get_job_status", { jobId });
export const listJobs = (limit = 100) => invoke<JobStatus[]>("list_jobs", { limit });
export const cancelJob = (jobId: string) => invoke<void>("cancel_job", { jobId });
export const retryJob = (jobId: string) => invoke<JobStatus>("retry_job", { jobId });
export const runBenchmark = (request: BenchmarkRequest) => invoke<BenchmarkResult[]>("run_benchmark", { request });
export const listModels = () => invoke<ModelRecord[]>("list_models");
export const openModelFolder = (modelId: string) => invoke<void>("open_model_folder", { modelId });
export const listRecipes = () => invoke<Recipe[]>("list_recipes");
export const loadRecipe = (path: string) => invoke<Recipe>("load_recipe", { path });
export const saveRecipe = (path: string, recipe: Recipe) => invoke<void>("save_recipe", { path, recipe });
export const exportReport = (path: string, contents: string) => invoke<void>("export_report", { path, contents });
export const onJobProgress = (handler: (payload: JobProgressEvent) => void): Promise<UnlistenFn> =>
  listen<JobProgressEvent>("job://progress", (event) => handler(event.payload));

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
export function runtimeRequired(): void {
  if (!isTauriRuntime()) throw new Error("Open the desktop app to use native dialogs, local files, hardware checks and conversion workers.");
}
export async function choosePath(directory = false, extension?: string): Promise<string | null> {
  runtimeRequired();
  const path = await open({ directory, multiple: false, ...(extension ? { filters: [{ name: extension.toUpperCase(), extensions: [extension] }] } : {}) });
  return typeof path === "string" ? path : null;
}
export async function chooseSavePath(extension = "json"): Promise<string | null> {
  runtimeRequired();
  return save({ filters: [{ name: extension.toUpperCase(), extensions: [extension] }] });
}
export const errorMessage = (error: unknown): string => error instanceof Error ? error.message : typeof error === "string" ? error : JSON.stringify(error);
