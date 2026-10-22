import { beforeEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { startConversion, getJobStatus, cancelJob, runBenchmark } from "./tauri";
import { useAppStore } from "@/stores/useAppStore";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
const request = { modelId: "org/model", url: "org/model", precision: "bf16", device: "npu", powerMode: "automatic", targetEp: "vitisai" } as const;
const job = { jobId: "job-1", modelId: "org/model", currentStep: "download", status: "running", progress: 10, logs: [{ ts: "now", stream: "stdout", line: "download started" }], artifacts: [] };

beforeEach(() => useAppStore.setState({ jobs: [], huggingfaceUrl: "", modelMetadata: null }));

describe("native command contracts", () => {
  it("sends request and keeps the authoritative returned job", async () => {
    vi.mocked(invoke).mockResolvedValue(job);
    expect(await startConversion(request)).toEqual(job);
    expect(invoke).toHaveBeenCalledWith("start_conversion", { request });
  });
  it("uses jobId for status and cancellation", async () => {
    await getJobStatus("job-1");
    expect(invoke).toHaveBeenLastCalledWith("get_job_status", { jobId: "job-1" });
    await cancelJob("job-1");
    expect(invoke).toHaveBeenLastCalledWith("cancel_job", { jobId: "job-1" });
  });
  it("passes benchmark warmup and devices under request", async () => {
    const benchmark = { modelPath: "model.onnx", devices: ["cpu", "npu"], warmup: 10, iterations: 100 };
    await runBenchmark(benchmark);
    expect(invoke).toHaveBeenLastCalledWith("run_benchmark", { request: benchmark });
  });
});

describe("shared workflow state", () => {
  it("clears inspection metadata immediately when the source changes", () => {
    useAppStore.setState({ modelMetadata: { modelId: "org/old" } as never });
    useAppStore.getState().setHuggingfaceUrl("org/new");
    expect(useAppStore.getState().modelMetadata).toBeNull();
  });
  it("keeps independent jobs when the backend uses jobId", () => {
    useAppStore.getState().upsertJob(job as never);
    useAppStore.getState().upsertJob({ ...job, jobId: "job-2" } as never);
    expect(useAppStore.getState().jobs).toHaveLength(2);
  });
  it("keeps live progress when the queued start response arrives after a progress event", () => {
    useAppStore.getState().mergeJobEvent({ jobId: "job-1", status: "running", currentStep: "download", progress: 20, line: "worker started" });
    useAppStore.getState().upsertJob({ ...job, status: "queued", progress: 0, currentStep: "inspection", logs: [], request } as never);
    expect(useAppStore.getState().jobs[0]).toMatchObject({ modelId: "org/model", status: "running", currentStep: "download", progress: 20, request });
    expect(useAppStore.getState().jobs[0].logs[0].line).toBe("worker started");
  });
  it("does not roll cancellation back when a running poll response is delayed", () => {
    useAppStore.getState().upsertJob(job as never);
    useAppStore.getState().mergeJobEvent({ jobId: "job-1", status: "cancelling" });
    useAppStore.getState().upsertJob({ ...job, progress: 5 } as never);
    expect(useAppStore.getState().jobs[0]).toMatchObject({ status: "cancelling", progress: 10 });
  });
  it("merges log-only and terminal events without erasing progress or reopening the job", () => {
    useAppStore.getState().upsertJob(job as never);
    useAppStore.getState().mergeJobEvent({ jobId: "job-1", stream: "stderr", line: "diagnostic" });
    useAppStore.getState().mergeJobEvent({ jobId: "job-1", status: "failed", error: "worker failed" });
    useAppStore.getState().mergeJobEvent({ jobId: "job-1", status: "running", progress: 20 });
    const current = useAppStore.getState().jobs[0];
    expect(current.currentStep).toBe("download");
    expect(current.logs.map(log => log.line)).toEqual(["download started", "diagnostic"]);
    expect(current.status).toBe("failed");
    expect(current.error).toBe("worker failed");
  });
  it("ignores an inspection response after switching sources, even when switching back", () => {
    const store = useAppStore.getState();
    store.setHuggingfaceUrl("org/first");
    const revision = useAppStore.getState().sourceRevision;
    store.setHuggingfaceUrl("org/second");
    store.setHuggingfaceUrl("org/first");
    store.acceptInspection(revision, { modelId: "org/first" } as never);
    expect(useAppStore.getState().modelMetadata).toBeNull();
  });
  it("loads recipe data fields and never derives input from command strings", () => {
    useAppStore.getState().applyRecipe({
      recipe_version: "1.0", name: "local", conversion_request: request,
      source: { kind: "onnx", onnx_path: "C:/models/input.onnx" },
      quantization: { calibration: { dir: "C:/calibration", max_samples: 23 }, accuracy_gate: { max_degradation: 0.03 } },
      commands: { validate: "eval --eval-npz UNSAFE-NOT-DATA" },
    });
    const state = useAppStore.getState();
    expect(state.sourceKind).toBe("onnx");
    expect(state.onnxPath).toBe("C:/models/input.onnx");
    expect(state.calibrationDir).toBe("C:/calibration");
    expect(state.maxSamples).toBe(23);
    expect(state.maxDegradation).toBe(0.03);
    expect(state.evalNpz).toBe("");
    expect(state.modelMetadata).toBeNull();
  });
  it("opens a job and benchmarks an artifact with navigation context intact", () => {
    useAppStore.getState().openJob("job-1");
    expect(useAppStore.getState()).toMatchObject({ tab: "pipeline", activeJobId: "job-1" });
    useAppStore.getState().benchmarkArtifact("C:/output/model.onnx", "job-1");
    expect(useAppStore.getState()).toMatchObject({ tab: "benchmark", benchmarkPath: "C:/output/model.onnx", benchmarkJobId: "job-1" });
  });
});
