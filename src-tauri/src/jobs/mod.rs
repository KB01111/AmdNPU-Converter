use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

use crate::error::ConverterError;
use crate::models::cache;
use crate::storage::db::JobRow;
use crate::AppState;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversionRequest {
    pub model_id: String,
    pub hf_url: String,
    pub precision: String,
    pub device: String,
    pub power_mode: String,
    pub ep: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub job_id: String,
    pub status: String,
    pub current_step: String,
    pub progress: f64,
}

const STEPS: [(&str, f64); 8] = [
    ("inspection", 5.0),
    ("download", 20.0),
    ("export", 40.0),
    ("optimize", 55.0),
    ("quantize", 70.0),
    ("compile", 85.0),
    ("validate", 93.0),
    ("benchmark", 100.0),
];

// ---------------------------------------------------------------------------
// Public API used by commands
// ---------------------------------------------------------------------------

pub async fn start_conversion(
    app: &AppHandle,
    req: ConversionRequest,
) -> Result<JobStatus, ConverterError> {
    if req.model_id.trim().is_empty() {
        return Err(ConverterError::InvalidModel("empty model_id".to_string()));
    }
    let job_id = Uuid::new_v4();
    let now = Utc::now().to_rfc3339();
    let config_json = serde_json::to_string(&req).unwrap_or_else(|_| "{}".to_string());
    let _ = cache::ensure_dirs();
    let log_path = cache::log_path_for_job(&job_id.to_string())
        .to_string_lossy()
        .to_string();

    let state: tauri::State<'_, AppState> = app.state();
    {
        let db = state.db.lock().await;
        db.insert_job(&JobRow {
            id: job_id.to_string(),
            model_id: req.model_id.clone(),
            config_json,
            status: "queued".to_string(),
            current_step: "inspection".to_string(),
            progress: 0.0,
            log_path,
            created_at: now.clone(),
            updated_at: now,
        })?;
    }
    {
        let mut reg = state.jobs.lock().await;
        reg.insert(
            job_id,
            crate::JobHandle {
                model_id: req.model_id.clone(),
                cancel_flag: Arc::new(AtomicBool::new(false)),
            },
        );
    }

    let status = JobStatus {
        job_id: job_id.to_string(),
        model_id: req.model_id.clone(),
        status: "queued".to_string(),
        current_step: "inspection".to_string(),
        progress: 0.0,
        logs: Vec::new(),
        artifacts: Vec::new(),
    };

    // Spawn the real pipeline.
    let app_c = app.clone();
    tokio::spawn(async move {
        run_pipeline(app_c, job_id, req).await;
    });

    Ok(status)
}

pub async fn get_job_status(
    app: &AppHandle,
    job_id: &str,
) -> Result<JobStatus, ConverterError> {
    let state: tauri::State<'_, AppState> = app.state();
    let db = state.db.lock().await;
    let row = db
        .get_job(job_id)?
        .ok_or_else(|| ConverterError::JobNotFound(job_id.to_string()))?;
    Ok(JobStatus {
        job_id: row.id.clone(),
        model_id: row.model_id.clone(),
        status: row.status.clone(),
        current_step: row.current_step.clone(),
        progress: row.progress,
        logs: read_tail_logs(&row.log_path, 200),
        artifacts: Vec::new(),
    })
}

pub async fn list_jobs(app: &AppHandle, limit: i64) -> Result<Vec<JobStatus>, ConverterError> {
    let state: tauri::State<'_, AppState> = app.state();
    let db = state.db.lock().await;
    let rows = db.list_jobs(limit.clamp(1, 500))?;
    Ok(rows
        .into_iter()
        .map(|r| JobStatus {
            job_id: r.id,
            model_id: r.model_id,
            status: r.status,
            current_step: r.current_step,
            progress: r.progress,
            logs: Vec::new(),
            artifacts: Vec::new(),
        })
        .collect())
}

pub async fn cancel_job(app: &AppHandle, job_id: &str) -> Result<(), ConverterError> {
    let id = Uuid::parse_str(job_id)
        .map_err(|_| ConverterError::JobNotFound(job_id.to_string()))?;
    let state: tauri::State<'_, AppState> = app.state();
    {
        let reg = state.jobs.lock().await;
        if let Some(h) = reg.get(&id) {
            h.cancel_flag.store(true, Ordering::SeqCst);
        } else {
            // No live handle — still mark DB row cancelled if it is open.
            let db = state.db.lock().await;
            if let Some(row) = db.get_job(job_id)? {
                if row.status == "queued" || row.status == "running" {
                    db.finish_job(job_id, "cancelled")?;
                    return Ok(());
                }
            }
            return Err(ConverterError::JobNotFound(job_id.to_string()));
        }
    }
    {
        let db = state.db.lock().await;
        // Mark cancelled; the pipeline task will observe the flag and stop.
        if let Some(row) = db.get_job(job_id)? {
            if row.status == "queued" || row.status == "running" {
                db.update_job_progress(job_id, "cancelled", &row.current_step, row.progress)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

fn is_cancelled(flag: &Arc<AtomicBool>) -> bool {
    flag.load(Ordering::SeqCst)
}

async fn run_pipeline(app: AppHandle, job_id: Uuid, req: ConversionRequest) {
    let id_str = job_id.to_string();
    let flag: Arc<AtomicBool> = {
        let state: tauri::State<'_, AppState> = app.state();
        let reg = state.jobs.lock().await;
        reg.get(&job_id)
            .map(|h| h.cancel_flag.clone())
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)))
    };

    let emit = |status: &str, step: &str, progress: f64| {
        let _ = app.emit(
            "job://progress",
            ProgressEvent {
                job_id: id_str.clone(),
                status: status.to_string(),
                current_step: step.to_string(),
                progress,
            },
        );
    };

    let fail = |msg: String| {
        let app_c = app.clone();
        let id_c = id_str.clone();
        async move {
            let state: tauri::State<'_, AppState> = app_c.state();
            let db = state.db.lock().await;
            let _ = db.finish_job(&id_c, "failed");
            let _ = app_c.emit(
                "job://progress",
                serde_json::json!({"jobId": id_c, "status": "failed", "error": msg}),
            );
        }
    };

    // Mark running.
    {
        let state: tauri::State<'_, AppState> = app.state();
        let db = state.db.lock().await;
        let _ = db.update_job_progress(&id_str, "running", "inspection", 0.0);
    }
    emit("running", "inspection", 0.0);

    let model_dir = cache::model_dir(&req.model_id);
    let _ = std::fs::create_dir_all(&model_dir);
    let onnx_path = model_dir.join("model.onnx");
    let opt_path = model_dir.join("model.opt.onnx");
    let quant_path = model_dir.join(format!("model.{}.onnx", req.precision));
    let compiled_path = model_dir.join(format!("model.{}.bin", req.ep.to_lowercase()));

    for (step, pct) in STEPS {
        if is_cancelled(&flag) {
            let state: tauri::State<'_, AppState> = app.state();
            let db = state.db.lock().await;
            let _ = db.update_job_progress(&id_str, "cancelled", step, pct);
            emit("cancelled", step, pct);
            return;
        }
        {
            let state: tauri::State<'_, AppState> = app.state();
            let db = state.db.lock().await;
            let _ = db.update_job_progress(&id_str, "running", step, pct);
        }
        emit("running", step, pct);

        let code: Result<i32, ConverterError> = match step {
            "inspection" => {
                // Real HF inspection for metadata; then winml inspect for local probe.
                match crate::models::hf::hf_inspect(&req.hf_url).await {
                    Ok(meta) => {
                        // Persist model metadata best-effort.
                        let state: tauri::State<'_, AppState> = app.state();
                        let db = state.db.lock().await;
                        let now = Utc::now().to_rfc3339();
                        let _ = db.upsert_model(&crate::storage::db::ModelRow {
                            model_id: req.model_id.clone(),
                            hf_id: meta.model_id.clone(),
                            arch: meta.arch.clone(),
                            params: meta.params.clone(),
                            format: meta.format.clone(),
                            local_path: model_dir.to_string_lossy().to_string(),
                            onnx_path: String::new(),
                            compiled_path: String::new(),
                            cache_key: String::new(),
                            created_at: now,
                        });
                        crate::winml::bridge::inspect(&req.model_id, &id_str, &app).await
                    }
                    Err(e) => {
                        fail(e.to_string()).await;
                        return;
                    }
                }
            }
            "download" => {
                crate::winml::bridge::run_python_worker(
                    "download.py",
                    &[
                        "--model".to_string(), req.model_id.clone(),
                        "--out".to_string(), model_dir.to_string_lossy().to_string(),
                    ],
                    &id_str,
                    &app,
                )
                .await
            }
            "export" => {
                crate::winml::bridge::export(
                    &req.model_id,
                    &onnx_path.to_string_lossy(),
                    17,
                    &id_str,
                    &app,
                )
                .await
            }
            "optimize" => {
                crate::winml::bridge::optimize(
                    &onnx_path.to_string_lossy(),
                    &opt_path.to_string_lossy(),
                    &id_str,
                    &app,
                )
                .await
            }
            "quantize" => {
                crate::winml::bridge::quantize(
                    &opt_path.to_string_lossy(),
                    &quant_path.to_string_lossy(),
                    &req.precision,
                    &id_str,
                    &app,
                )
                .await
            }
            "compile" => {
                crate::winml::bridge::compile(
                    &quant_path.to_string_lossy(),
                    &compiled_path.to_string_lossy(),
                    &req.ep,
                    &id_str,
                    &app,
                )
                .await
            }
            "validate" => {
                crate::winml::bridge::eval(
                    &compiled_path.to_string_lossy(),
                    "",
                    &id_str,
                    &app,
                )
                .await
            }
            "benchmark" => {
                crate::winml::bridge::perf(&compiled_path.to_string_lossy(), &id_str, &app).await
            }
            _ => Ok(0),
        };

        match code {
            Ok(0) => {}
            Ok(c) => {
                fail(format!("step `{step}` exited with code {c}")).await;
                let state: tauri::State<'_, AppState> = app.state();
                let db = state.db.lock().await;
                let _ = db.finish_job(&id_str, "failed");
                return;
            }
            Err(e) => {
                fail(e.to_string()).await;
                let state: tauri::State<'_, AppState> = app.state();
                let db = state.db.lock().await;
                let _ = db.finish_job(&id_str, "failed");
                return;
            }
        }
    }

    {
        let state: tauri::State<'_, AppState> = app.state();
        let db = state.db.lock().await;
        let _ = db.finish_job(&id_str, "completed");
        // Record final artifacts on the model row.
        let now = Utc::now().to_rfc3339();
        let hw = crate::hardware::get_hardware_info()
            .map(|h| h.cpu.brand)
            .unwrap_or_default();
        let ck = crate::winml::bridge::compute_cache_key(
            &req.model_id, &req.precision, &req.precision, &hw, &req.ep, "",
        );
        let _ = db.upsert_model(&crate::storage::db::ModelRow {
            model_id: req.model_id.clone(),
            hf_id: req.hf_url.clone(),
            arch: String::new(),
            params: String::new(),
            format: "onnx".to_string(),
            local_path: model_dir.to_string_lossy().to_string(),
            onnx_path: onnx_path.to_string_lossy().to_string(),
            compiled_path: compiled_path.to_string_lossy().to_string(),
            cache_key: ck,
            created_at: now,
        });
    }
    emit("completed", "benchmark", 100.0);
}

fn read_tail_logs(path: &str, max_lines: usize) -> Vec<LogLine> {
    if path.is_empty() {
        return Vec::new();
    }
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();
    let start = n.saturating_sub(max_lines);
    lines[start..]
        .iter()
        .map(|l| LogLine {
            ts: Utc::now().to_rfc3339(),
            stream: if l.starts_with("[err]") { "stderr".to_string() } else { "stdout".to_string() },
            line: l.to_string(),
        })
        .collect()
}
