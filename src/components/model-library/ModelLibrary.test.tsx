import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import ModelLibrary from "./ModelLibrary";
import { useAppStore } from "@/stores/useAppStore";
import type { JobStatus } from "@/lib/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
const path = "C:/jobs/A/model.cuda.onnx";
const completed: JobStatus = { jobId: "A", modelId: "org/model", status: "completed", currentStep: "done", progress: 100, logs: [], artifacts: [path], request: { modelId: "org/model", url: "org/model", precision: "bf16", device: "gpu", targetEp: "cuda", powerMode: "automatic" } };
const newerFailed: JobStatus = { ...completed, jobId: "B", status: "failed", artifacts: [], request: { ...completed.request!, targetEp: "vitisai" } };
beforeEach(() => {
  Object.defineProperty(window, "__TAURI_INTERNALS__", { value: {}, configurable: true });
  useAppStore.setState({ jobs: [newerFailed, completed], activeJobId: null, tab: "library", benchmarkJobId: null });
});
afterEach(cleanup);

it.each([true, false])("keeps artifact provenance when a newer same-model job failed (persisted context=%s)", async persisted => {
  vi.mocked(invoke).mockResolvedValue([{ modelId: "org/model", hfId: "org/model", arch: "transformer", params: "1M", format: "onnx", localPath: "C:/jobs/A", onnxPath: "", compiledPath: path, cacheKey: "key", createdAt: "now", jobId: persisted ? "A" : null, targetEp: persisted ? "cuda" : null }]);
  render(<QueryClientProvider client={new QueryClient()}><ModelLibrary /></QueryClientProvider>);
  await userEvent.click(await screen.findByRole("button", { name: "Benchmark compiled" }));
  expect(useAppStore.getState()).toMatchObject({ benchmarkPath: path, benchmarkJobId: "A", benchmarkTargetEp: "cuda" });
  await userEvent.click(screen.getByRole("button", { name: "Open conversion" }));
  expect(useAppStore.getState()).toMatchObject({ activeJobId: "A", tab: "pipeline" });
});
