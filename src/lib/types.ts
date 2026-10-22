/**
 * TypeScript mirrors of the Rust serde structs exposed via Tauri commands.
 * Rust uses `rename_all = "camelCase"`, so all fields are camelCase here.
 */

export interface ModelFileInfo {
  name: string;
  sizeBytes: number;
  sha256?: string | null;
}

export interface SafetensorsInfo {
  numTensors: number;
  dtype?: string | null;
  totalParams: number;
  hasSafetensorsIndex: boolean;
}

export interface ModelMetadata {
  modelId: string;
  url: string;
  arch: string;
  params: number;
  paramsHuman: string;
  format: string;
  license?: string | null;
  files: ModelFileInfo[];
  safetensors?: SafetensorsInfo | null;
  totalSizeBytes: number;
  configJson?: Record<string, unknown> | null;
}

export interface CpuInfo {
  name: string;
  cores: number;
  threads: number;
  vendor?: string | null;
}

export interface GpuInfo {
  name: string;
  vendor?: string | null;
  vramBytes?: number | null;
}

export interface NpuInfo {
  name: string;
  available: boolean;
  tops?: number | null;
  driverVersion?: string | null;
}

export interface DriverInfo {
  name: string;
  version: string;
  status: string;
}

export type PowerMode = "automatic" | "max-performance" | "battery-efficient";

export interface HardwareInfo {
  cpu: CpuInfo;
  gpus: GpuInfo[];
  npu?: NpuInfo | null;
  executionProviders: string[];
  drivers: DriverInfo[];
  powerMode: PowerMode;
}

export type Precision = "auto" | "fp32" | "bf16" | "a16w8" | "a8w8" | "int8";
export type DevicePreference = "auto" | "npu" | "gpu" | "cpu";
export type TargetEp = "vitisai" | "qnn" | "cuda" | "cpu";

export interface ConversionRequest {
  modelId: string;
  url: string;
  precision: Precision;
  device: DevicePreference;
  powerMode: PowerMode;
  targetEp: TargetEp;
  outputDir?: string | null;
}

export type JobStage =
  | "inspection"
  | "download"
  | "export"
  | "optimize"
  | "quantize"
  | "compile"
  | "validate"
  | "benchmark"
  | "done";

export type JobState = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobStatus {
  id: string;
  modelId: string;
  stage: JobStage;
  progress: number;
  status: JobState;
  logs: string[];
  error?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface JobProgressEvent {
  jobId: string;
  stage: JobStage;
  progress: number;
  message: string;
}

export interface BenchmarkRequest {
  modelPath: string;
  devices: string[];
  iterations?: number | null;
}

export interface BenchmarkResult {
  device: string;
  avgLatencyMs: number;
  throughputTps: number;
  powerW?: number | null;
  peakMemoryMb?: number | null;
}

export interface ModelRecord {
  id: string;
  modelId: string;
  path: string;
  format: string;
  sizeBytes: number;
  createdAt: string;
}

export type AppTab = "import" | "pipeline" | "hardware" | "benchmark" | "library";
