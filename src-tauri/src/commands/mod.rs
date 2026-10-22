use crate::{
    error::ConverterError,
    jobs::{ConversionRequest, JobStatus},
    AppState,
};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
};
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRequest {
    pub model_path: String,
    pub devices: Vec<String>,
    #[serde(default)]
    pub iterations: Option<u32>,
    #[serde(default)]
    pub warmup: Option<u32>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub target_ep: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub device: String,
    pub providers: Vec<String>,
    pub avg_latency_ms: Option<f64>,
    pub p50_ms: Option<f64>,
    pub p90_ms: Option<f64>,
    pub throughput_per_s: Option<f64>,
    pub npu_verified: bool,
    pub error: Option<String>,
    pub memory_mb: Option<f64>,
}
fn benchmark_arguments(request: &BenchmarkRequest) -> Result<Vec<String>, ConverterError> {
    crate::jobs::validate_benchmark_numbers(request.iterations, request.warmup)?;
    if request.devices.is_empty()
        || request.devices.len() > 3
        || request
            .devices
            .iter()
            .any(|d| !matches!(d.as_str(), "cpu" | "gpu" | "npu"))
    {
        return Err(ConverterError::Benchmark(
            "Choose cpu, gpu and/or npu devices".into(),
        ));
    }
    let mut args = vec![
        "--model".into(),
        request.model_path.clone(),
        "--devices".into(),
        request.devices.join(","),
        "--iterations".into(),
        request.iterations.unwrap_or(100).to_string(),
        "--warmup".into(),
        request.warmup.unwrap_or(10).to_string(),
    ];
    if let Some(ep) = &request.target_ep {
        let device = match ep.as_str() {
            "cpu" => "cpu",
            "dml" | "cuda" => "gpu",
            "vitisai" => "npu",
            _ => {
                return Err(ConverterError::Benchmark(
                    "Supported target providers: cpu, dml, cuda, vitisai".into(),
                ))
            }
        };
        if !request.devices.iter().any(|d| d == device) {
            return Err(ConverterError::Benchmark(
                "Selected provider requires its matching benchmark device".into(),
            ));
        }
        args.extend(["--ep".into(), ep.clone()]);
    }
    Ok(args)
}
pub fn parse_benchmark_results(stdout: &str) -> Result<Vec<BenchmarkResult>, ConverterError> {
    let json = serde_json::from_str::<serde_json::Value>(stdout).or_else(|_| {
        stdout
            .lines()
            .rev()
            .find_map(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .filter(|v| v.is_array())
            })
            .ok_or_else(|| serde_json::from_str::<serde_json::Value>("invalid").unwrap_err())
    })?;
    let array = json.as_array().filter(|v| !v.is_empty()).ok_or_else(|| {
        ConverterError::Benchmark("Expected a nonempty benchmark result array".into())
    })?;
    array
        .iter()
        .map(|v| {
            let device = v
                .get("device")
                .and_then(|v| v.as_str())
                .filter(|d| matches!(*d, "cpu" | "gpu" | "npu"))
                .ok_or_else(|| ConverterError::Benchmark("Missing benchmark device".into()))?
                .to_string();
            let providers = v
                .get("providers")
                .and_then(|v| v.as_array())
                .map(|v| {
                    v.iter()
                        .filter_map(|p| p.as_str().map(str::to_owned))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let error = v.get("error").and_then(|v| v.as_str()).map(str::to_owned);
            let metric = |key: &str| -> Result<Option<f64>, ConverterError> {
                match v.get(key) {
                    Some(serde_json::Value::Null) | None => Ok(None),
                    Some(value) => value
                        .as_f64()
                        .filter(|n| n.is_finite() && *n >= 0.0)
                        .map(Some)
                        .ok_or_else(|| ConverterError::Benchmark(format!("Invalid {key}"))),
                }
            };
            let avg = metric("latency_ms_mean")?;
            let p50 = metric("latency_ms_p50")?;
            let p90 = metric("latency_ms_p90")?;
            let throughput = metric("throughput_per_s")?;
            if error.is_none()
                && (providers.is_empty()
                    || avg.is_none()
                    || p50.is_none()
                    || p90.is_none()
                    || throughput.is_none())
            {
                return Err(ConverterError::Benchmark(
                    "Successful benchmark is missing metrics or providers".into(),
                ));
            }
            let npu_verified = device == "npu"
                && v.get("npu_verified")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                && providers
                    .iter()
                    .any(|p| p == "VitisAIExecutionProvider" || p == "NPUExecutionProvider");
            let memory = metric("memory_rss_after")?
                .or(metric("memory_rss_before")?)
                .map(|bytes| bytes / 1_048_576.0);
            Ok(BenchmarkResult {
                device,
                providers,
                avg_latency_ms: avg,
                p50_ms: p50,
                p90_ms: p90,
                throughput_per_s: throughput,
                npu_verified,
                error,
                memory_mb: memory,
            })
        })
        .collect()
}
#[tauri::command]
pub async fn hf_inspect(url: String) -> Result<crate::models::hf::ModelMetadata, String> {
    crate::models::hf::hf_inspect(&url)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn get_hardware_info(app: AppHandle) -> Result<crate::hardware::HardwareInfo, String> {
    let mut hardware = tauri::async_runtime::spawn_blocking(crate::hardware::get_hardware_info)
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let discovery = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        crate::winml::bridge::run_worker(
            &app,
            "evaluation/bench.py",
            &["--discover-json".into()],
            None,
            &Arc::new(AtomicBool::new(false)),
        ),
    )
    .await;
    match discovery {
        Ok(Ok(output)) => match crate::hardware::parse_discovery(&output.stdout) {
            Ok(report) => {
                hardware.execution_providers = report.providers;
                hardware.provider_devices = report.devices;
                hardware.provider_errors = report.errors;
                if output.code != 0 && hardware.provider_errors.is_empty() {
                    hardware.provider_errors.push(format!(
                        "Provider discovery failed (exit {}): {}",
                        output.code,
                        output.stderr.trim()
                    ));
                }
            }
            Err(error) => hardware.provider_errors.push(format!(
                "Provider discovery response failed: {error}. {}",
                output.stderr.trim()
            )),
        },
        Ok(Err(error)) => hardware.provider_errors.push(error.to_string()),
        Err(_) => hardware.provider_errors.push(
            "Provider discovery timed out after 30 seconds; the probe process was stopped".into(),
        ),
    }
    hardware.execution_providers.sort();
    hardware.execution_providers.dedup();
    Ok(hardware)
}
#[tauri::command]
pub async fn start_conversion(
    app: AppHandle,
    request: ConversionRequest,
) -> Result<JobStatus, String> {
    crate::jobs::start_conversion(&app, request)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn get_job_status(app: AppHandle, job_id: String) -> Result<JobStatus, String> {
    crate::jobs::get_job_status(&app, &job_id)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn list_jobs(app: AppHandle, limit: Option<i64>) -> Result<Vec<JobStatus>, String> {
    crate::jobs::list_jobs(&app, limit.unwrap_or(50))
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn cancel_job(app: AppHandle, job_id: String) -> Result<(), String> {
    crate::jobs::cancel_job(&app, &job_id)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn retry_job(app: AppHandle, job_id: String) -> Result<JobStatus, String> {
    crate::jobs::retry_job(&app, &job_id)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn run_benchmark(
    app: AppHandle,
    request: BenchmarkRequest,
) -> Result<Vec<BenchmarkResult>, String> {
    async fn execute(
        app: &AppHandle,
        request: BenchmarkRequest,
    ) -> Result<Vec<BenchmarkResult>, ConverterError> {
        let args = benchmark_arguments(&request)?;
        let path = Path::new(&request.model_path);
        if !path.is_absolute() || !path.is_file() {
            return Err(ConverterError::Benchmark(
                "Select an existing absolute model artifact path".into(),
            ));
        }
        if let Some(id) = &request.job_id {
            crate::jobs::get_job_status(app, id).await?;
        }
        let output = crate::winml::bridge::run_worker(
            app,
            "evaluation/bench.py",
            &args,
            None,
            &Arc::new(AtomicBool::new(false)),
        )
        .await?;
        let rows = parse_benchmark_results(&output.stdout).map_err(|e| {
            ConverterError::Benchmark(format!(
                "{e}. Worker exit {}: {}",
                output.code,
                output.stderr.trim()
            ))
        })?;
        if output.code != 0 && rows.iter().all(|r| r.error.is_none()) {
            return Err(ConverterError::Benchmark(format!(
                "Worker failed (exit {}): {}",
                output.code, output.stderr
            )));
        }
        let report = serde_json::to_string(&rows)?;
        app.state::<AppState>()
            .db
            .lock()
            .await
            .save_benchmark_report(
                request.job_id.as_deref().unwrap_or("standalone"),
                &request.model_path,
                &report,
            )?;
        Ok(rows)
    }
    execute(&app, request).await.map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn list_models(app: AppHandle) -> Result<Vec<crate::storage::db::ModelRow>, String> {
    app.state::<AppState>()
        .db
        .lock()
        .await
        .list_models()
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn open_model_folder(app: AppHandle, model_id: String) -> Result<(), String> {
    let models = app
        .state::<AppState>()
        .db
        .lock()
        .await
        .list_models()
        .map_err(|e| e.to_string())?;
    let row = models
        .into_iter()
        .find(|m| m.model_id == model_id)
        .ok_or("Model has no completed artifact folder")?;
    let dir = Path::new(&row.local_path);
    if !dir.is_dir() {
        return Err("Artifact folder no longer exists".into());
    }
    #[allow(deprecated)]
    app.shell()
        .open(dir.to_string_lossy().to_string(), None)
        .map_err(|e| e.to_string())
}

pub type Recipe = serde_json::Value;
pub fn validate_recipe(mut value: Recipe) -> Result<Recipe, ConverterError> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| ConverterError::Other("Recipe must be a JSON object".into()))?;
    if object.get("recipe_version").and_then(|v| v.as_str()) != Some("1.0") {
        return Err(ConverterError::Other(
            "Unsupported recipe_version; expected 1.0".into(),
        ));
    }
    if object
        .get("name")
        .and_then(|v| v.as_str())
        .is_none_or(|s| s.trim().is_empty())
    {
        return Err(ConverterError::Other("Recipe name is required".into()));
    }
    let mut request: ConversionRequest =
        serde_json::from_value(object.get("conversion_request").cloned().ok_or_else(|| {
            ConverterError::Other("Recipe conversion_request is required".into())
        })?)?;
    if let Some(source) = object.get("source") {
        if !source.is_object() {
            return Err(ConverterError::Other(
                "Recipe source must be an object".into(),
            ));
        }
        if let Some(kind) = source.get("kind") {
            let kind = kind
                .as_str()
                .filter(|s| matches!(*s, "hf" | "onnx"))
                .ok_or_else(|| {
                    ConverterError::Other("Recipe source kind must be hf or onnx".into())
                })?;
            request.source_kind = Some(kind.into());
        }
        for key in ["trust_remote_code", "safetensors_only"] {
            if source.get(key).is_some_and(|v| !v.is_boolean()) {
                return Err(ConverterError::Other(format!(
                    "Recipe {key} must be a boolean"
                )));
            }
        }
        if source.get("trust_remote_code").and_then(|v| v.as_bool()) == Some(true)
            || source.get("safetensors_only").and_then(|v| v.as_bool()) == Some(false)
        {
            return Err(ConverterError::Other(
                "Recipes cannot enable remote code or unsafe weights".into(),
            ));
        }
        if let Some(path) = source.get("onnx_path") {
            let path = path
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| {
                    ConverterError::Other("Recipe onnx_path must be a nonempty string".into())
                })?;
            request.source_kind = Some("onnx".into());
            request.onnx_path = Some(path.into());
        }
    }
    if let Some(q) = object.get("quantization") {
        if !q.is_object() {
            return Err(ConverterError::Other(
                "Recipe quantization must be an object".into(),
            ));
        }
        if let Some(path) = q.pointer("/calibration/dir") {
            request.calibration_dir = Some(
                path.as_str()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| {
                        ConverterError::Other(
                            "Recipe calibration dir must be a nonempty string".into(),
                        )
                    })?
                    .into(),
            );
        }
        if let Some(n) = q.pointer("/calibration/max_samples") {
            request.max_samples = Some(n.as_u64().filter(|n| *n > 0 && *n <= 100_000).ok_or_else(
                || ConverterError::Other("Recipe max_samples must be 1..100000".into()),
            )? as u32);
        }
        if let Some(n) = q.pointer("/accuracy_gate/max_degradation") {
            request.max_degradation = Some(
                n.as_f64()
                    .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                    .ok_or_else(|| {
                        ConverterError::Other("Recipe maximum degradation must be 0..1".into())
                    })?,
            );
        }
    }
    request.validate_options()?;
    // The shipped generic HF recipe intentionally leaves this value for the user.
    if request.source_kind.as_deref().unwrap_or("hf") == "hf"
        && request.model_id != "<HF_MODEL_ID>"
        && crate::models::hf::parse_hf_id(&request.url)? != request.model_id
    {
        return Err(ConverterError::Other(
            "Recipe model ID must match the Hugging Face URL".into(),
        ));
    }
    object.insert("conversion_request".into(), serde_json::to_value(request)?);
    object.remove("commands");
    Ok(value)
}
fn read_recipe(path: &Path) -> Result<Recipe, ConverterError> {
    if path.metadata()?.len() > 2 * 1024 * 1024 {
        return Err(ConverterError::Other("Recipe exceeds 2 MiB".into()));
    }
    validate_recipe(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}
#[tauri::command]
pub async fn list_recipes(app: AppHandle) -> Result<Vec<Recipe>, String> {
    fn walk(dir: &Path, result: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_symlink() {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                walk(&path, result)?;
            } else if path.extension().is_some_and(|s| s == "json") {
                result.push(path);
            }
        }
        Ok(())
    }
    let root = crate::winml::bridge::app_resource(&app, "recipes");
    let mut paths = vec![];
    walk(&root, &mut paths).map_err(|e| format!("Recipe resources unavailable: {e}"))?;
    paths.sort();
    let mut recipes = vec![];
    for path in paths {
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let value: Recipe =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if value.get("support_status").and_then(|s| s.as_str()) == Some("pending-worker-support") {
            continue;
        }
        recipes.push(validate_recipe(value).map_err(|e| format!("{}: {e}", path.display()))?);
    }
    Ok(recipes)
}
#[tauri::command]
pub async fn load_recipe(path: String) -> Result<Recipe, String> {
    if !Path::new(&path).is_absolute() {
        return Err("Recipe path must be absolute".into());
    }
    read_recipe(Path::new(&path)).map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn save_recipe(path: String, recipe: Recipe) -> Result<(), String> {
    let recipe = validate_recipe(recipe).map_err(|e| e.to_string())?;
    export_report(
        path,
        serde_json::to_string_pretty(&recipe).map_err(|e| e.to_string())?,
    )
    .await
}
#[tauri::command]
pub async fn export_report(path: String, contents: String) -> Result<(), String> {
    let path = Path::new(&path);
    if !path.is_absolute()
        || contents.len() > 10 * 1024 * 1024
        || path
            .extension()
            .is_none_or(|ext| !ext.eq_ignore_ascii_case("json") && !ext.eq_ignore_ascii_case("txt"))
    {
        return Err("Choose an absolute .json or .txt report path; report limit is 10 MiB".into());
    }
    let parent = path.parent().ok_or("Invalid output path")?;
    if !parent.is_dir() {
        return Err("Output folder does not exist".into());
    }
    std::fs::write(path, contents).map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn benchmark_request_preserves_selected_provider() {
        let mut request:BenchmarkRequest=serde_json::from_value(serde_json::json!({"modelPath":"C:/model.onnx","devices":["gpu","cpu"],"targetEp":"cuda"})).unwrap();
        assert_eq!(serde_json::to_value(&request).unwrap()["targetEp"], "cuda");
        assert!(benchmark_arguments(&request)
            .unwrap()
            .windows(2)
            .any(|pair| pair == ["--ep", "cuda"]));
        request.devices = vec!["cpu".into()];
        assert!(benchmark_arguments(&request).is_err());
    }
    #[test]
    fn benchmark_normalizes_real_units_and_error_rows() {
        let rows=parse_benchmark_results(r#"[{"device":"cpu","providers":["CPUExecutionProvider"],"latency_ms_mean":2.0,"latency_ms_p50":1.0,"latency_ms_p90":3.0,"throughput_per_s":500.0,"memory_rss_after":1048576,"npu_verified":false},{"device":"npu","error":"provider missing"}]"#).unwrap();
        assert_eq!(rows[0].avg_latency_ms, Some(2.0));
        assert_eq!(rows[0].memory_mb, Some(1.0));
        assert_eq!(serde_json::to_value(&rows[0]).unwrap()["memoryMb"], 1.0);
        assert!(rows[1].avg_latency_ms.is_none());
        assert!(!rows[1].npu_verified);
        assert_eq!(
            serde_json::to_value(&rows[0]).unwrap()["throughputPerS"],
            500.0
        );
    }
    #[test]
    fn malformed_benchmark_does_not_invent_zero_metrics() {
        assert!(parse_benchmark_results(r#"[{"device":"cpu"}]"#).is_err());
        assert!(parse_benchmark_results("[]").is_err());
    }
    #[test]
    fn recipe_is_data_only_and_reads_nested_settings() {
        let recipe = serde_json::json!({"recipe_version":"1.0","name":"sample","conversion_request":{"modelId":"org/model","url":"org/model","precision":"bf16","device":"npu","powerMode":"automatic","targetEp":"vitisai"},"commands":{"run":"evil"},"quantization":{"calibration":{"dir":"C:/data","max_samples":22},"accuracy_gate":{"max_degradation":0.01}}});
        let parsed = validate_recipe(recipe).unwrap();
        assert!(parsed.get("commands").is_none());
        assert_eq!(parsed["conversion_request"]["calibrationDir"], "C:/data");
        assert_eq!(parsed["conversion_request"]["maxDegradation"], 0.01);
    }
    #[test]
    fn rejects_unsupported_recipe_schema() {
        assert!(validate_recipe(serde_json::json!({"recipe_version":"99","name":"bad"})).is_err());
    }
    #[test]
    fn rejects_unsupported_recipe_precision() {
        let recipe = serde_json::json!({"recipe_version":"1.0","name":"sample","conversion_request":{"modelId":"org/model","url":"org/model","precision":"int4","device":"npu","powerMode":"automatic","targetEp":"vitisai"}});
        assert!(validate_recipe(recipe).is_err());
    }
    #[test]
    fn shipped_templates_load_without_existing_datasets() {
        let root = crate::winml::bridge::development_root().join("recipes");
        for name in [
            "generic/hf-default.json",
            "generic/onnx-import.json",
            "amd-xdna/npu-default.json",
            "amd-xdna/cnn-a8w8.json",
            "amd-xdna/transformer-bf16.json",
            "cua-s1/forms-a16w8.json",
            "cua-s1/nano-text-bf16.json",
        ] {
            assert!(read_recipe(&root.join(name)).is_ok(), "failed {name}");
        }
    }
}
