use crate::{
    error::ConverterError,
    models::cache,
    storage::db::{JobRow, ModelRow},
    AppState,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversionRequest {
    pub model_id: String,
    pub url: String,
    pub precision: String,
    pub device: String,
    pub power_mode: String,
    pub target_ep: String,
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub source_kind: Option<String>,
    #[serde(default)]
    pub onnx_path: Option<String>,
    #[serde(default)]
    pub calibration_dir: Option<String>,
    #[serde(default)]
    pub eval_npz: Option<String>,
    #[serde(default)]
    pub max_degradation: Option<f64>,
    #[serde(default)]
    pub max_samples: Option<u32>,
    #[serde(default)]
    pub iterations: Option<u32>,
    #[serde(default)]
    pub warmup: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogLine {
    pub ts: String,
    pub stream: String,
    pub line: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub job_id: String,
    pub model_id: String,
    pub status: String,
    pub current_step: String,
    pub progress: f64,
    pub logs: Vec<LogLine>,
    pub artifacts: Vec<String>,
    pub error: Option<String>,
    pub request: Option<ConversionRequest>,
}
impl ConversionRequest {
    pub fn validate_options(&mut self) -> Result<(), ConverterError> {
        if self.model_id.trim().is_empty() || self.model_id.len() > 200 {
            return invalid("A model name is required");
        }
        self.device = self.device.to_lowercase();
        self.target_ep = self.target_ep.to_lowercase();
        self.precision = self.precision.to_lowercase();
        if self.device == "auto" {
            self.device = match self.target_ep.as_str() {
                "cpu" => "cpu",
                "dml" | "cuda" => "gpu",
                "vitisai" => "npu",
                _ => "unknown",
            }
            .into();
        }
        if !matches!(
            (self.device.as_str(), self.target_ep.as_str()),
            ("cpu", "cpu") | ("gpu", "dml") | ("gpu", "cuda") | ("npu", "vitisai")
        ) {
            return invalid(
                "Supported device/provider pairs: cpu/cpu, gpu/dml, gpu/cuda, npu/vitisai",
            );
        }
        if self.power_mode != "automatic" {
            return invalid(
                "Only automatic power mode is supported; configure OS power policy separately",
            );
        }
        if self.precision == "auto" {
            self.precision = "bf16".into();
        }
        if !matches!(self.precision.as_str(), "bf16" | "a16w8" | "a8w8" | "int8") {
            return invalid("Unsupported precision; choose bf16, a16w8, a8w8 or int8");
        }
        if !matches!(self.source_kind.as_deref().unwrap_or("hf"), "hf" | "onnx") {
            return invalid("Unknown source kind");
        }
        if self
            .max_degradation
            .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            return invalid("Maximum degradation must be finite and between 0 and 1");
        }
        if self.max_samples.is_some_and(|n| n == 0 || n > 100_000) {
            return invalid("Calibration samples must be between 1 and 100000");
        }
        validate_benchmark_numbers(self.iterations, self.warmup)
    }
    pub fn validate(&mut self) -> Result<(), ConverterError> {
        self.validate_options()?;
        if self.source_kind.as_deref().unwrap_or("hf") == "onnx" {
            let path = required_path(&self.onnx_path, "ONNX source")?;
            if !path.is_file()
                || path
                    .extension()
                    .is_none_or(|s| !s.eq_ignore_ascii_case("onnx"))
            {
                return invalid("ONNX source must be an existing .onnx file");
            }
            self.onnx_path = Some(path.to_string_lossy().into());
        } else if self.source_kind.as_deref().unwrap_or("hf") == "hf" {
            let id = crate::models::hf::parse_hf_id(&self.url)?;
            if id != self.model_id {
                return invalid("Model ID must match the Hugging Face URL");
            }
            self.url = id;
        } else {
            return invalid("Unknown source kind");
        }
        let eval = required_path(&self.eval_npz, "Representative evaluation NPZ")?;
        if !eval.is_file()
            || eval
                .extension()
                .is_none_or(|s| !s.eq_ignore_ascii_case("npz"))
        {
            return invalid("Evaluation data must be an existing .npz file");
        }
        self.eval_npz = Some(eval.to_string_lossy().into());
        if self.precision != "bf16" || self.calibration_dir.is_some() {
            let path = required_path(
                &self.calibration_dir,
                "Representative calibration directory",
            )?;
            if !path.is_dir() {
                return invalid("Calibration path is not a directory");
            }
            self.calibration_dir = Some(path.to_string_lossy().into());
        }
        if let Some(path) = &self.output_dir {
            let path = Path::new(path);
            if !path.is_absolute() || (path.exists() && !path.is_dir()) {
                return invalid("Output directory must be an absolute directory path");
            }
        }
        Ok(())
    }
}
fn invalid<T>(s: &str) -> Result<T, ConverterError> {
    Err(ConverterError::Other(s.into()))
}
fn required_path(value: &Option<String>, name: &str) -> Result<PathBuf, ConverterError> {
    let p = value
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| ConverterError::Other(format!("{name} is required")))?;
    let path = Path::new(p);
    if !path.is_absolute() {
        return invalid(&format!("{name} must be an absolute path"));
    }
    Ok(std::fs::canonicalize(path)?)
}
pub fn validate_benchmark_numbers(
    iterations: Option<u32>,
    warmup: Option<u32>,
) -> Result<(), ConverterError> {
    if iterations.is_some_and(|n| n == 0 || n > 100_000) || warmup.is_some_and(|n| n > 10_000) {
        return invalid("Benchmark iterations must be 1..100000 and warmup 0..10000");
    }
    Ok(())
}
pub async fn start_conversion(
    app: &AppHandle,
    mut request: ConversionRequest,
) -> Result<JobStatus, ConverterError> {
    request.validate()?;
    let id = Uuid::new_v4();
    let now = Utc::now().to_rfc3339();
    cache::ensure_dirs()?;
    let state = app.state::<AppState>();
    state.db.lock().await.insert_job(&JobRow {
        id: id.to_string(),
        model_id: request.model_id.clone(),
        config_json: serde_json::to_string(&request)?,
        status: "queued".into(),
        current_step: "inspection".into(),
        progress: 0.0,
        log_path: cache::log_path_for_job(&id.to_string())
            .to_string_lossy()
            .into(),
        created_at: now.clone(),
        updated_at: now,
    })?;
    let flag = Arc::new(AtomicBool::new(false));
    state.jobs.lock().await.insert(
        id,
        crate::JobHandle {
            cancel_flag: flag.clone(),
        },
    );
    let status = get_job_status(app, &id.to_string()).await?;
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        run_pipeline(app, id, request, flag).await;
    });
    Ok(status)
}
async fn status_for(app: &AppHandle, row: JobRow) -> Result<JobStatus, ConverterError> {
    let state = app.state::<AppState>();
    let db = state.db.lock().await;
    let (error, artifacts) = db.job_details(&row.id)?;
    Ok(JobStatus {
        job_id: row.id,
        model_id: row.model_id,
        status: row.status,
        current_step: row.current_step,
        progress: row.progress,
        logs: read_tail_logs(&row.log_path, 200),
        artifacts,
        error,
        request: serde_json::from_str(&row.config_json).ok(),
    })
}
pub async fn get_job_status(app: &AppHandle, id: &str) -> Result<JobStatus, ConverterError> {
    let row = app
        .state::<AppState>()
        .db
        .lock()
        .await
        .get_job(id)?
        .ok_or_else(|| ConverterError::JobNotFound(id.into()))?;
    status_for(app, row).await
}
pub async fn list_jobs(app: &AppHandle, limit: i64) -> Result<Vec<JobStatus>, ConverterError> {
    let rows = app
        .state::<AppState>()
        .db
        .lock()
        .await
        .list_jobs(limit.clamp(1, 500))?;
    let mut result = vec![];
    for row in rows {
        result.push(status_for(app, row).await?);
    }
    Ok(result)
}
pub async fn retry_job(app: &AppHandle, id: &str) -> Result<JobStatus, ConverterError> {
    let row = get_job_status(app, id).await?;
    if matches!(row.status.as_str(), "queued" | "running" | "cancelling") {
        return Err(ConverterError::JobClosed(
            "Cannot retry an active job".into(),
        ));
    }
    start_conversion(
        app,
        row.request.ok_or_else(|| {
            ConverterError::Other(
                "This older job has no reproducible request; create a new conversion".into(),
            )
        })?,
    )
    .await
}
pub async fn cancel_job(app: &AppHandle, id: &str) -> Result<(), ConverterError> {
    let uuid = Uuid::parse_str(id).map_err(|_| ConverterError::JobNotFound(id.into()))?;
    let state = app.state::<AppState>();
    let reg = state.jobs.lock().await;
    let handle = reg
        .get(&uuid)
        .ok_or_else(|| ConverterError::JobClosed(id.into()))?;
    let row = state
        .db
        .lock()
        .await
        .get_job(id)?
        .ok_or_else(|| ConverterError::JobNotFound(id.into()))?;
    if !matches!(row.status.as_str(), "queued" | "running" | "cancelling") {
        return Err(ConverterError::JobClosed(id.into()));
    }
    handle.cancel_flag.store(true, Ordering::SeqCst);
    state
        .db
        .lock()
        .await
        .update_job_progress(id, "cancelling", &row.current_step, row.progress)?;
    app.emit(
        "job://progress",
        serde_json::json!({"jobId":id,"status":"cancelling"}),
    )
    .ok();
    Ok(())
}
#[derive(Debug)]
pub struct Stage {
    pub name: &'static str,
    pub script: &'static str,
    pub args: Vec<String>,
    pub artifact: Option<PathBuf>,
}
fn args(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}
pub fn pipeline_stages(req: &ConversionRequest, root: &Path) -> Vec<Stage> {
    let onnx = root.join("model.onnx");
    let opt = root.join("model.opt.onnx");
    let quant = root.join(format!("model.{}.onnx", req.precision));
    let compiled = root.join(format!("model.{}.onnx", req.target_ep));
    let source = root.join("source");
    let o = onnx.to_string_lossy();
    let p = opt.to_string_lossy();
    let q = quant.to_string_lossy();
    let c = compiled.to_string_lossy();
    let s = source.to_string_lossy();
    let mut stages = vec![];
    if req.source_kind.as_deref() == Some("onnx") {
        stages.push(Stage {
            name: "prepare",
            script: "prepare_onnx.py",
            args: args(&[
                "--input",
                req.onnx_path.as_deref().unwrap_or(""),
                "--output",
                &o,
                "--opset",
                "17",
            ]),
            artifact: Some(onnx.clone()),
        });
    } else {
        stages.push(Stage {
            name: "download",
            script: "download.py",
            args: args(&["--model", &req.model_id, "--out", &s]),
            artifact: None,
        });
        if req.model_id.starts_with("cua-ai/cua-s1") {
            let weights = source.join("model.safetensors");
            let config = source.join("config.json");
            stages.push(Stage {
                name: "export",
                script: "exporters/cua-s1/export_cua_s1.py",
                args: args(&[
                    "--weights",
                    &weights.to_string_lossy(),
                    "--config",
                    &config.to_string_lossy(),
                    "--output",
                    &o,
                    "--opset",
                    "17",
                    "--variant",
                    if req.model_id.contains("nano") {
                        "nano-text"
                    } else {
                        "forms"
                    },
                ]),
                artifact: Some(onnx.clone()),
            });
        } else {
            stages.push(Stage {
                name: "export",
                script: "exporters/transformers/export_hf.py",
                args: args(&["--model-id", &s, "--output", &o, "--opset", "17"]),
                artifact: Some(onnx.clone()),
            });
        }
    }
    stages.push(Stage {
        name: "optimize",
        script: "winml/run.py",
        args: args(&["optimize", "--", "--model", &o, "--output", &p]),
        artifact: Some(opt.clone()),
    });
    let mut quant_args = args(&[
        "--input",
        &p,
        "--output",
        &q,
        "--profile",
        &req.precision,
        "--max-samples",
        &req.max_samples.unwrap_or(64).to_string(),
    ]);
    if let Some(dir) = &req.calibration_dir {
        quant_args.extend(args(&["--calibration-dir", dir]));
    }
    stages.push(Stage {
        name: "quantize",
        script: "quark/quantize.py",
        args: quant_args,
        artifact: Some(quant.clone()),
    });
    stages.push(Stage {
        name: "validate",
        script: "evaluation/compare.py",
        args: args(&[
            "--baseline",
            &p,
            "--candidate",
            &q,
            "--eval-npz",
            req.eval_npz.as_deref().unwrap_or(""),
            "--max-degradation",
            &req.max_degradation.unwrap_or(0.02).to_string(),
        ]),
        artifact: None,
    });
    stages.push(Stage {
        name: "compile",
        script: "winml/run.py",
        args: args(&[
            "compile",
            "--",
            "--model",
            &q,
            "--output",
            &c,
            "--device",
            &req.device,
            "--ep",
            &req.target_ep,
        ]),
        artifact: Some(compiled.clone()),
    });
    stages.push(Stage {
        name: "benchmark",
        script: "evaluation/bench.py",
        args: args(&[
            "--model",
            &c,
            "--devices",
            &req.device,
            "--ep",
            &req.target_ep,
            "--iterations",
            &req.iterations.unwrap_or(100).to_string(),
            "--warmup",
            &req.warmup.unwrap_or(10).to_string(),
        ]),
        artifact: None,
    });
    stages
}
async fn progress(app: &AppHandle, id: &str, step: &str, pct: f64) -> Result<(), ConverterError> {
    app.state::<AppState>()
        .db
        .lock()
        .await
        .update_job_progress(id, "running", step, pct)?;
    app.emit(
        "job://progress",
        serde_json::json!({"jobId":id,"status":"running","currentStep":step,"progress":pct}),
    )
    .ok();
    Ok(())
}
async fn execute_pipeline(
    app: &AppHandle,
    id: &str,
    req: &ConversionRequest,
    flag: &Arc<AtomicBool>,
    artifacts: &mut Vec<String>,
) -> Result<(), ConverterError> {
    let root = req
        .output_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| cache::model_dir(&req.model_id))
        .join(id);
    std::fs::create_dir_all(&root)?;
    progress(app, id, "inspection", 0.0).await?;
    let metadata = if req.source_kind.as_deref() != Some("onnx") {
        tokio::select! {
            result=crate::models::hf::hf_inspect(&req.url)=>{
                let meta=result?;
                if meta.format=="onnx"{return invalid("This Hugging Face repository provides ONNX files. Download the ONNX model and its external data, then use local ONNX import.");}
                Some(meta)
            },
            _=crate::winml::bridge::wait_cancel(flag)=>return Err(ConverterError::Cancelled),
        }
    } else {
        None
    };
    let stages = pipeline_stages(req, &root);
    for (index, stage) in stages.iter().enumerate() {
        if flag.load(Ordering::SeqCst) {
            return Err(ConverterError::Cancelled);
        }
        progress(
            app,
            id,
            stage.name,
            (index as f64 + 1.0) * 95.0 / stages.len() as f64,
        )
        .await?;
        let output =
            crate::winml::bridge::run_worker(app, stage.script, &stage.args, Some(id), flag)
                .await?;
        if flag.load(Ordering::SeqCst) {
            return Err(ConverterError::Cancelled);
        }
        let report = root.join(format!("{}.report.txt", stage.name));
        std::fs::write(&report, format!("{}\n{}", output.stdout, output.stderr))?;
        artifacts.push(report.to_string_lossy().into());
        app.state::<AppState>()
            .db
            .lock()
            .await
            .save_job_details(id, None, artifacts)?;
        if output.code != 0 {
            return invalid(&format!(
                "{} failed (exit {}): {}",
                stage.name,
                output.code,
                output.stderr.trim()
            ));
        }
        if let Some(path) = &stage.artifact {
            if !path.is_file() || path.metadata()?.len() == 0 {
                return invalid(&format!(
                    "{} produced no artifact: {}",
                    stage.name,
                    path.display()
                ));
            }
            artifacts.push(path.to_string_lossy().into());
        }
        if stage.name == "benchmark" {
            let rows = crate::commands::parse_benchmark_results(&output.stdout)?;
            app.state::<AppState>()
                .db
                .lock()
                .await
                .save_benchmark_report(
                    id,
                    &root.to_string_lossy(),
                    &serde_json::to_string(&rows)?,
                )?;
            if rows.iter().any(|row| row.error.is_some()) {
                return invalid("Benchmark reported a device failure; see benchmark report");
            }
            if req.device == "npu"
                && !rows
                    .iter()
                    .any(|row| row.device == "npu" && row.npu_verified)
            {
                return invalid("NPU execution remains unverified: no complete NPU node-placement evidence; artifacts and reports retained");
            }
        }
        app.state::<AppState>()
            .db
            .lock()
            .await
            .save_job_details(id, None, artifacts)?;
    }
    if flag.load(Ordering::SeqCst) {
        return Err(ConverterError::Cancelled);
    }
    app.state::<AppState>()
        .db
        .lock()
        .await
        .upsert_model(&ModelRow {
            model_id: req.model_id.clone(),
            hf_id: req.url.clone(),
            arch: metadata
                .as_ref()
                .map(|m| m.arch.clone())
                .unwrap_or_default(),
            params: metadata
                .as_ref()
                .map(|m| m.params.clone())
                .unwrap_or_default(),
            format: "onnx".into(),
            local_path: root.to_string_lossy().into(),
            onnx_path: root
                .join(format!("model.{}.onnx", req.precision))
                .to_string_lossy()
                .into(),
            compiled_path: root
                .join(format!("model.{}.onnx", req.target_ep))
                .to_string_lossy()
                .into(),
            cache_key: cache::cache_key(&[id]),
            created_at: Utc::now().to_rfc3339(),
            job_id: Some(id.into()),
            target_ep: Some(req.target_ep.clone()),
        })?;
    Ok(())
}
async fn run_pipeline(app: AppHandle, id: Uuid, req: ConversionRequest, flag: Arc<AtomicBool>) {
    let id_text = id.to_string();
    let mut artifacts = vec![];
    let result = execute_pipeline(&app, &id_text, &req, &flag, &mut artifacts).await;
    let status = terminal_status(&result);
    let error = if status == "cancelled" {
        None
    } else {
        result.err().map(|e| e.to_string())
    };
    let state = app.state::<AppState>();
    {
        let db = state.db.lock().await;
        if let Err(e) = db
            .save_job_details(&id_text, error.as_deref(), &artifacts)
            .and_then(|_| db.finish_job(&id_text, status))
        {
            crate::winml::bridge::append_log(
                &cache::log_path_for_job(&id_text),
                "stderr",
                &format!("Failed to persist terminal state: {e}"),
            );
        }
    }
    state.jobs.lock().await.remove(&id);
    app.emit("job://progress",serde_json::json!({"jobId":id_text,"status":status,"error":error,"progress":if status=="completed"{Some(100.0)}else{None}})).ok();
}
fn terminal_status(result: &Result<(), ConverterError>) -> &'static str {
    match result {
        Err(ConverterError::Cancelled) => "cancelled",
        Ok(()) => "completed",
        Err(_) => "failed",
    }
}
fn read_tail_logs(path: &str, max_lines: usize) -> Vec<LogLine> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<_> = text.lines().collect();
    lines[lines.len().saturating_sub(max_lines)..]
        .iter()
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|_| LogLine {
                ts: String::new(),
                stream: if line.starts_with("[err]") {
                    "stderr"
                } else {
                    "stdout"
                }
                .into(),
                line: line.to_string(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    pub fn request() -> ConversionRequest {
        serde_json::from_value(serde_json::json!({"modelId":"org/model","url":"org/model","precision":"auto","device":"npu","powerMode":"automatic","targetEp":"vitisai"})).unwrap()
    }
    #[test]
    fn shared_request_accepts_url_and_target_ep() {
        let r = request();
        assert_eq!(serde_json::to_value(r).unwrap()["targetEp"], "vitisai");
    }
    #[test]
    fn unsupported_modes_rejected() {
        let mut r = request();
        r.precision = "int4".into();
        assert!(r.validate().unwrap_err().to_string().contains("precision"));
        r.precision = "bf16".into();
        r.target_ep = "dml".into();
        assert!(r.validate().is_err());
    }
    #[test]
    fn evaluation_is_required() {
        let mut r = request();
        assert!(r.validate().unwrap_err().to_string().contains("evaluation"));
    }
    #[test]
    fn local_source_skips_download_and_hf_export() {
        let mut r = request();
        r.source_kind = Some("onnx".into());
        r.onnx_path = Some("C:/source.onnx".into());
        let s = pipeline_stages(&r, Path::new("C:/job"));
        assert_eq!(s[0].script, "prepare_onnx.py");
        assert!(!s.iter().any(|s| s.name == "download"));
        assert!(s
            .iter()
            .any(|s| s.script == "evaluation/compare.py" && s.args.contains(&"--eval-npz".into())));
    }
    #[test]
    fn cua_source_uses_downloaded_weights() {
        let mut r = request();
        r.model_id = "cua-ai/cua-s1-forms".into();
        let s = pipeline_stages(&r, Path::new("C:/job"));
        assert_eq!(s[1].script, "exporters/cua-s1/export_cua_s1.py");
        assert!(s[1].args.iter().any(|a| a.ends_with("model.safetensors")));
        assert!(!s[1].args.contains(&"--model-id".into()));
    }
    #[test]
    fn benchmark_bounds_reject_zero_and_excessive_work() {
        assert!(validate_benchmark_numbers(Some(0), None).is_err());
        assert!(validate_benchmark_numbers(Some(1), Some(10001)).is_err());
        assert!(validate_benchmark_numbers(Some(100), Some(0)).is_ok());
    }
    #[test]
    fn pipeline_benchmark_uses_selected_provider() {
        let mut r = request();
        r.target_ep = "cuda".into();
        r.device = "gpu".into();
        let stages = pipeline_stages(&r, Path::new("C:/job"));
        let benchmark = stages.iter().find(|s| s.name == "benchmark").unwrap();
        assert!(benchmark
            .args
            .windows(2)
            .any(|pair| pair == ["--ep", "cuda"]));
    }
    #[test]
    fn failed_process_tree_stop_is_not_reported_cancelled() {
        let result = Err(ConverterError::Other("Could not stop worker tree".into()));
        assert_eq!(terminal_status(&result), "failed");
        assert_eq!(
            terminal_status(&Err(ConverterError::Cancelled)),
            "cancelled"
        );
    }
}
