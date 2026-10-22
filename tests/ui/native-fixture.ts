import type { Page } from "@playwright/test";

// Explicit browser-only Tauri fixtures. These tests do not exercise native execution.
export async function installNativeFixture(page: Page) {
  await page.addInitScript(() => {
    const calls: {command: string; args: Record<string, unknown>}[] = [];
    const callbacks = new Map<number, (value: unknown) => void>();
    let nextCallback = 1;
    let progressHandler = 0;
    const request = {
      modelId: "fixture/tiny", url: "https://huggingface.co/fixture/tiny",
      precision: "bf16", device: "npu", powerMode: "automatic", targetEp: "vitisai",
      evalNpz: "C:\\fixture\\eval.npz",
    };
    const jobs = [{
      jobId: "11111111-1111-4111-8111-111111111111", modelId: request.modelId,
      status: "failed", currentStep: "compile", progress: 70,
      logs: [{ts: "2026-10-22T12:00:00Z", stream: "stderr", line: "fixture compiler diagnostic"}],
      artifacts: [], error: "Fixture compiler unavailable", request,
    }];
    const fixture = {
      calls,
      emit(payload: Record<string, unknown>) {
        callbacks.get(progressHandler)?.({event: "job://progress", id: progressHandler, payload});
      },
    };
    Object.assign(window, {
      isTauri: true,
      __nativeFixture: fixture,
      __TAURI_EVENT_PLUGIN_INTERNALS__: {
        unregisterListener(_event: string, id: number) { callbacks.delete(id); },
      },
      __TAURI_INTERNALS__: {
        transformCallback(callback: (value: unknown) => void) {
          const id = nextCallback++;
          callbacks.set(id, callback);
          return id;
        },
        unregisterCallback(id: number) { callbacks.delete(id); },
        async invoke(command: string, args: Record<string, unknown> = {}) {
          calls.push({command, args});
          switch (command) {
            case "plugin:event|listen": progressHandler = args.handler as number; return progressHandler;
            case "plugin:event|unlisten": return null;
            case "get_hardware_info": return {
              cpu: {brand: "Fixture CPU", vendor: "AMD", physicalCores: 8, logicalCores: 16, frequencyMhz: 3500},
              gpus: [{name: "Fixture GPU", driverVersion: "1.2", vendor: "AMD"}],
              npu: {present: true, name: "Fixture NPU", driverVersion: "1.3"},
              winml: {available: true, binaryPath: "C:\\fixture\\winml.exe", version: "1.0"},
              osName: "Windows", osVersion: "11", totalMemoryGb: 32,
              executionProviders: ["VitisAIExecutionProvider", "CPUExecutionProvider"],
              providerErrors: ["Fixture provider setup required"],
              driverVersions: {NPU: "1.3"},
            };
            case "list_recipes": return [{recipe_version: "1.0", name: "Fixture BF16", conversion_request: request}];
            case "hf_inspect": return {
              modelId: "fixture/tiny", arch: "bert", params: "42000", format: "safetensors",
              license: "mit", inputTypes: ["text"], precision: "fp32", estimatedBytes: 1024,
              files: [{rfilename: "model.safetensors", size: 1024}],
              compat: {exportStrategy: "transformers", opset: 17, notes: "Fixture inspection"},
            };
            case "list_jobs": return jobs;
            case "get_job_status": return jobs.find(job => job.jobId === args.jobId) ?? jobs[0];
            case "start_conversion": {
              const job = {...jobs[0], jobId: "22222222-2222-4222-8222-222222222222", status: "queued", currentStep: "inspection", progress: 0, logs: [], error: "", request: args.request as typeof request};
              jobs.unshift(job);
              return job;
            }
            case "cancel_job": {
              const job = jobs.find(job => job.jobId === args.jobId);
              if (job) job.status = "cancelled";
              return null;
            }
            case "retry_job": return {...jobs[0], status: "queued", error: null};
            case "list_models": return [{
              modelId: "fixture/tiny", hfId: "fixture/tiny", arch: "bert", params: "42000",
              format: "onnx", localPath: "C:\\fixture", onnxPath: "C:\\fixture\\model.onnx",
              compiledPath: "C:\\fixture\\model.vitisai.onnx", cacheKey: "fixture", createdAt: "2026-10-22T12:00:00Z",
              jobId: "11111111-1111-4111-8111-111111111111", targetEp: "vitisai",
            }];
            case "run_benchmark": return [
              {device: "cpu", providers: ["CPUExecutionProvider"], avgLatencyMs: 12, p50Ms: 11, p90Ms: 14, throughputPerS: 83.3, npuVerified: false},
              {device: "npu", providers: ["VitisAIExecutionProvider"], avgLatencyMs: 3, p50Ms: 2.8, p90Ms: 4, throughputPerS: 333.3, npuVerified: false, error: "Fixture placement evidence unavailable"},
            ];
            case "open_model_folder":
            case "export_report":
            case "save_recipe": return null;
            case "plugin:dialog|open": return "C:\\fixture\\model.onnx";
            case "plugin:dialog|save": return "C:\\fixture\\report.json";
            default: throw new Error(`Unexpected fixture command: ${command}`);
          }
        },
      },
    });
  });
}
