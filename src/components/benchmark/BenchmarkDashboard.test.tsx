import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import BenchmarkDashboard from "./BenchmarkDashboard";
import { useAppStore } from "@/stores/useAppStore";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
beforeEach(() => {
  Object.defineProperty(window, "__TAURI_INTERNALS__", { value: {}, configurable: true });
  useAppStore.setState({ benchmarkRunning: false, benchmarkResults: [], benchmarkDevices: ["cpu", "npu"], benchmarkIterations: 100, benchmarkWarmup: 10 });
  vi.mocked(invoke).mockResolvedValue([]);
});
afterEach(cleanup);

it("passes a CUDA artifact's explicit provider to native benchmarking", async () => {
  useAppStore.getState().benchmarkArtifact("C:/jobs/A/model.cuda.onnx", "A", "cuda");
  render(<BenchmarkDashboard />);
  await userEvent.click(screen.getByRole("button", { name: "Run benchmark" }));
  expect(invoke).toHaveBeenCalledWith("run_benchmark", { request: { modelPath: "C:/jobs/A/model.cuda.onnx", jobId: "A", targetEp: "cuda", devices: ["cpu", "npu", "gpu"], iterations: 100, warmup: 10 } });
});

it("clears artifact provenance when the user changes model path and permits an explicit provider override", async () => {
  useAppStore.getState().benchmarkArtifact("C:/jobs/A/model.cuda.onnx", "A", "cuda");
  render(<BenchmarkDashboard />);
  await userEvent.clear(screen.getByLabelText("Model artifact path"));
  expect(useAppStore.getState()).toMatchObject({ benchmarkJobId: null, benchmarkTargetEp: null });
  await userEvent.selectOptions(screen.getByLabelText("Execution provider"), "dml");
  expect(useAppStore.getState()).toMatchObject({ benchmarkTargetEp: "dml" });
  expect((screen.getByRole("checkbox", { name: "GPU" }) as HTMLInputElement).checked).toBe(true);
  expect((screen.getByRole("checkbox", { name: "GPU" }) as HTMLInputElement).disabled).toBe(true);
});
