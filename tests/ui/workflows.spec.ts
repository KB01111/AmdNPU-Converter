import { test, expect } from "@playwright/test";
import { installNativeFixture } from "./native-fixture";

let runtimeErrors: string[];
test.beforeEach(({page}) => {
  runtimeErrors = [];
  page.on("pageerror", error => runtimeErrors.push(error.message));
});
test.afterEach(() => expect(runtimeErrors).toEqual([]));

test("browser preview explains the desktop requirement", async ({page}) => {
  await page.goto("/");
  await expect(page.getByText(/desktop.*required|desktop.*preview|browser preview|desktop app to/i).first()).toBeVisible();
});

test("hardware uses real Rust fields and library opens the failed job", async ({page}) => {
  await installNativeFixture(page);
  await page.goto("/");
  await page.getByRole("button", {name: "Hardware", exact: true}).click();
  await expect(page.getByText("Fixture CPU", {exact: true})).toBeVisible();
  await expect(page.getByText("Fixture NPU", {exact: true})).toBeVisible();
  await expect(page.getByText("Fixture provider setup required", {exact: true})).toBeVisible();
  await page.getByRole("button", {name: "Library", exact: true}).click();
  await page.getByRole("button", {name: /open|view/i}).first().click();
  await expect(page.getByText("Fixture compiler unavailable", {exact: false}).first()).toBeVisible();
  await expect(page.getByText("fixture compiler diagnostic", {exact: false}).first()).toBeVisible();
});

test("benchmark renders actual providers and keeps unverified NPU explicit", async ({page}) => {
  await installNativeFixture(page);
  await page.goto("/");
  await page.getByRole("button", {name: "Benchmark", exact: true}).click();
  await page.getByRole("textbox", {name: "Model artifact path", exact: true}).fill("C:\\fixture\\model.onnx");
  await page.getByRole("button", {name: /run benchmark/i}).click();
  await expect(page.getByText("CPUExecutionProvider", {exact: false}).first()).toBeVisible();
  await expect(page.getByText("Fixture placement evidence unavailable", {exact: false})).toBeVisible();
  await expect(page.getByText(/inferences\/s|inferences per second|inf\/s/i).first()).toBeVisible();
});

test("HF conversion sends validated settings and source edits invalidate inspection", async ({page}) => {
  await installNativeFixture(page);
  await page.goto("/");
  await page.getByLabel("Hugging Face model ID or URL", {exact: true}).fill("fixture/tiny");
  await page.getByRole("button", {name: "Inspect", exact: true}).click();
  await expect(page.getByText("42000", {exact: true})).toBeVisible();
  await page.getByRole("textbox", {name: "Evaluation NPZ (required)", exact: true}).fill("C:\\fixture\\eval.npz");
  await page.getByRole("button", {name: "Start conversion", exact: true}).click();
  await expect(page.getByRole("heading", {name: "Conversion pipeline", exact: true})).toBeVisible();
  const payload = await page.evaluate(() => (window as unknown as {__nativeFixture: {calls: {command:string;args:unknown}[]}}).__nativeFixture.calls.find(call => call.command === "start_conversion")?.args);
  expect(payload).toMatchObject({request: {modelId: "fixture/tiny", url: "fixture/tiny", sourceKind: "hf", targetEp: "vitisai", evalNpz: "C:\\fixture\\eval.npz"}});
  await page.getByRole("button", {name: "Import", exact: true}).click();
  await page.getByLabel("Hugging Face model ID or URL", {exact: true}).fill("fixture/other");
  await expect(page.getByRole("button", {name: "Start conversion", exact: true})).toBeDisabled();
});

test("ONNX import skips HF inspection and library artifacts populate benchmark", async ({page}) => {
  await installNativeFixture(page);
  await page.goto("/");
  await page.getByLabel("Model source", {exact: true}).selectOption("onnx");
  await page.getByRole("textbox", {name: "Local ONNX path", exact: true}).fill("C:\\fixture\\source.onnx");
  await page.getByRole("textbox", {name: "Evaluation NPZ (required)", exact: true}).fill("C:\\fixture\\eval.npz");
  await page.getByRole("button", {name: "Start conversion", exact: true}).click();
  await expect(page.getByRole("heading", {name: "Conversion pipeline", exact: true})).toBeVisible();
  const calls = await page.evaluate(() => (window as unknown as {__nativeFixture: {calls: {command:string;args:unknown}[]}}).__nativeFixture.calls);
  expect(calls.some(call => call.command === "hf_inspect")).toBe(false);
  expect(calls.find(call => call.command === "start_conversion")?.args).toMatchObject({request: {sourceKind: "onnx", onnxPath: "C:\\fixture\\source.onnx"}});
  await page.getByRole("button", {name: "Library", exact: true}).click();
  await page.getByRole("button", {name: "Benchmark compiled", exact: true}).click();
  await expect(page.getByRole("textbox", {name: "Model artifact path", exact: true})).toHaveValue(/model\..*onnx$/);
});

test("compact layout and keyboard source selection remain usable", async ({page}) => {
  await installNativeFixture(page);
  await page.setViewportSize({width: 860, height: 700});
  await page.emulateMedia({reducedMotion: "reduce"});
  await page.goto("/");
  const source = page.getByLabel("Model source", {exact: true});
  await source.focus();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(source).toHaveValue("onnx");
  await expect(page.getByRole("textbox", {name: "Local ONNX path", exact: true})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({path: "test-results/import-desktop.png", fullPage: true});
});
