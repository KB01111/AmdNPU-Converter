import { afterEach, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import HuggingFaceImport from "./HuggingFaceImport";
import { useAppStore } from "@/stores/useAppStore";

afterEach(cleanup);

it("selects the verified official Nano repository and resets inspection metadata", async () => {
  useAppStore.setState({ modelMetadata: { modelId: "org/old", arch: "transformer", params: "1M", format: "safetensors", license: "mit", inputTypes: ["input_ids"], precision: "fp32", estimatedBytes: 1000, files: [], compat: { exportStrategy: "transformers", opset: 17, notes: "" } } });
  render(<HuggingFaceImport />);
  await userEvent.click(screen.getByRole("button", { name: "Nano text · BF16" }));
  expect(useAppStore.getState()).toMatchObject({ huggingfaceUrl: "cua-ai/cua-s1-nano-0.1", sourceKind: "hf", precision: "bf16", modelMetadata: null });
});
