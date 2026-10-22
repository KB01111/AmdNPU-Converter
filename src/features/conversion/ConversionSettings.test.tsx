import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import ConversionSettings from "./ConversionSettings";
import { useAppStore } from "@/stores/useAppStore";
import { validateHfUrl } from "@/features/huggingface/HuggingFaceImport";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
beforeEach(() => {
  Object.defineProperty(window, "__TAURI_INTERNALS__", { value: {}, configurable: true });
  useAppStore.setState({ sourceKind: "hf", huggingfaceUrl: "https://huggingface.co/org/model", modelMetadata: null, precision: "bf16", device: "npu", targetEp: "vitisai", powerMode: "automatic" });
  vi.mocked(invoke).mockImplementation(async command => command === "list_recipes" ? [] : undefined);
});
afterEach(cleanup);
const show = () => render(<QueryClientProvider client={new QueryClient()}><ConversionSettings /></QueryClientProvider>);

it("rejects unrelated hosts, unsupported suffixes, and invalid HF path segments", () => {
  for (const url of ["https://example.com/org/model", "org/model/other/main", "org//model", "../model", "org/model/tree/../file", `${"x".repeat(97)}/model`]) {
    expect(validateHfUrl(url)).toBeNull();
  }
});

it("uses labeled native selects and marks unavailable execution providers", () => {
  show();
  expect(screen.getByRole("combobox", { name: "Precision" }).tagName).toBe("SELECT");
  expect((screen.getByRole("option", { name: /QNN/ }) as HTMLOptionElement).disabled).toBe(true);
});

it.each(["https://huggingface.co/org/model", "https://huggingface.co/org/model/tree/main", "org/model/blob/main/config.json", "huggingface.co/org/model/resolve/main/model.onnx"])("saves the canonical HF model ID for %s before inspection", async url => {
  useAppStore.setState({ huggingfaceUrl: url });
  show();
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Recipe JSON path"), "C:/recipes/mine.json");
  await user.click(screen.getByRole("button", { name: "Save recipe" }));
  const call = vi.mocked(invoke).mock.calls.find(([command]) => command === "save_recipe");
  expect(call).toBeDefined();
  expect(call?.[1]).toMatchObject({ path: "C:/recipes/mine.json", recipe: { conversion_request: { modelId: "org/model", url } } });
});

it("omits optional calibration paths when saving a BF16 recipe", async () => {
  useAppStore.setState({ calibrationDir: "" });
  show();
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Recipe JSON path"), "C:/recipes/bf16.json");
  await user.click(screen.getByRole("button", { name: "Save recipe" }));
  const call = [...vi.mocked(invoke).mock.calls].reverse().find(([command]) => command === "save_recipe");
  const recipe = (call?.[1] as {recipe: {quantization: {calibration: {dir?:string}}}}).recipe;
  expect(recipe.quantization.calibration.dir).toBeUndefined();
});
