/** Rust serde DTOs use camelCase; recipes retain their versioned JSON keys. */
export interface ModelMetadata {
  modelId: string; arch: string; params: string; format: string; license: string;
  inputTypes: string[]; precision: string; estimatedBytes: number;
  files: { rfilename: string; size?: number | null }[];
  compat: { exportStrategy: string; opset: number; notes: string };
}
export interface HardwareInfo {
  cpu: { brand: string; vendor: string; physicalCores: number; logicalCores: number; frequencyMhz: number };
  gpus: { name: string; vendor: string; driverVersion: string }[];
  npu: { present: boolean; name: string; driverVersion: string };
  winml: { available: boolean; binaryPath: string; version: string };
  osName: string; osVersion: string; totalMemoryGb: number;
  executionProviders: string[]; driverVersions: Record<string, string>;
  providerErrors?: string[];
}
export type PowerMode = "automatic" | "max-performance" | "battery-efficient";
export type Precision = "auto" | "fp32" | "bf16" | "a16w8" | "a8w8" | "int8" | "int4";
export type DevicePreference = "auto" | "npu" | "gpu" | "cpu";
export type TargetEp = "vitisai" | "qnn" | "cuda" | "cpu" | "dml";
export interface ConversionRequest {
  modelId: string; url: string; precision: Precision; device: DevicePreference;
  powerMode: PowerMode; targetEp: TargetEp; outputDir?: string | null;
  sourceKind?: "hf" | "onnx"; onnxPath?: string | null;
  calibrationDir?: string | null; evalNpz?: string | null;
  maxDegradation?: number; maxSamples?: number; iterations?: number; warmup?: number;
}
export type JobStage = "inspection" | "download" | "export" | "prepare" | "optimize" | "quantize" | "compile" | "validate" | "benchmark" | "done";
export interface JobLog { ts: string; stream: string; line: string }
export interface JobStatus {
  jobId: string; modelId: string; status: string; currentStep: string; progress: number;
  logs: JobLog[]; artifacts: string[]; error?: string | null; request?: ConversionRequest | null;
}
export interface JobProgressEvent {
  jobId: string; status?: string; currentStep?: string; progress?: number;
  error?: string | null; stream?: string; line?: string;
}
export interface BenchmarkRequest {
  modelPath: string; devices: string[]; iterations?: number; warmup?: number; jobId?: string | null;
  targetEp?: TargetEp | null;
}
export interface BenchmarkResult {
  device: string; providers: string[]; avgLatencyMs?: number | null;
  p50Ms?: number | null; p90Ms?: number | null; throughputPerS?: number | null;
  npuVerified: boolean; error?: string | null; memoryMb?: number | null;
}
export interface ModelRecord {
  modelId: string; hfId: string; arch: string; params: string; format: string;
  localPath: string; onnxPath: string; compiledPath: string; cacheKey: string; createdAt: string;
  jobId?: string | null; targetEp?: TargetEp | null;
}
export interface Recipe {
  recipe_version: string; name: string; description?: string;
  conversion_request: ConversionRequest;
  source?: { kind?: "hf" | "onnx"; onnx_path?: string; [key: string]: unknown };
  quantization?: { calibration?: { dir?: string; max_samples?: number; [key: string]: unknown }; accuracy_gate?: { max_degradation?: number; [key: string]: unknown }; [key: string]: unknown };
  [key: string]: unknown;
}
export type AppTab = "import" | "pipeline" | "hardware" | "benchmark" | "library";
