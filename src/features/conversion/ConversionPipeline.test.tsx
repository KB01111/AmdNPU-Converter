import { afterEach, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { ReactFlow } from "@xyflow/react";
import ConversionPipeline from "./ConversionPipeline";
import { useAppStore } from "@/stores/useAppStore";
import type { JobStatus } from "@/lib/types";

vi.mock("@xyflow/react", () => ({ ReactFlow: vi.fn(() => null), Background: () => null, Controls: () => null }));
afterEach(cleanup);

it("updates stages and logs while keeping graph nodes stable across unrelated updates", () => {
  const job: JobStatus = { jobId: "A", modelId: "local-model", status: "running", currentStep: "prepare", progress: 10, logs: [], artifacts: [], request: { modelId: "local-model", url: "", sourceKind: "onnx", precision: "bf16", device: "cpu", targetEp: "cpu", powerMode: "automatic" } };
  useAppStore.setState({ jobs: [job], activeJobId: "A" });
  render(<ConversionPipeline />);
  const lastNodes = () => vi.mocked(ReactFlow).mock.lastCall![0].nodes!;
  const initialNodes = lastNodes();
  expect(initialNodes.some(node => node.id === "prepare")).toBe(true);
  expect(initialNodes.some(node => node.id === "download")).toBe(false);
  const renderCount = vi.mocked(ReactFlow).mock.calls.length;
  act(() => useAppStore.setState({ benchmarkPath: "unrelated.onnx" }));
  expect(vi.mocked(ReactFlow).mock.calls.length).toBe(renderCount);
  act(() => useAppStore.getState().mergeJobEvent({ jobId: "A", stream: "stderr", line: "new diagnostic" }));
  expect(screen.getByLabelText("Worker log output").textContent).toContain("stderr: new diagnostic");
  expect(lastNodes()).toBe(initialNodes);
  act(() => useAppStore.getState().mergeJobEvent({ jobId: "A", currentStep: "optimize", progress: 25 }));
  expect(lastNodes()).not.toBe(initialNodes);
  expect(lastNodes().find(node => node.id === "prepare")?.data.label).toBe("prepare · done");
});
