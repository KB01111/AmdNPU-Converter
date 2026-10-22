use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::ShellExt;

use crate::jobs::ConversionRequest;
use crate::models::cache;
use crate::AppState;

// ---------------------------------------------------------------------------
// Shared DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub job_id: String,
    pub ep: String,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub throughput: f64,
    pub raw: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn hf_inspect(url: String) -> Result<crate::models::hf::ModelMetadata, String> {
    crate::models::hf::hf_inspect(&url).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_hardware_info() -> Result<crate::hardware::HardwareInfo, String> {
    crate::hardware::get_hardware_info().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn start_conversion(
    app: AppHandle,
    request: ConversionRequest,
) -> Result<crate::jobs::JobStatus, String> {
    crate::jobs::start_conversion(&app, request)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_job_status(
    app: AppHandle,
    job_id: String,
) -> Result<crate::jobs::JobStatus, String> {
    crate::jobs::get_job_status(&app, &job_id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_jobs(
    app: AppHandle,
    limit: Option<i64>,
) -> Result<Vec<crate::jobs::JobStatus>, String> {
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
pub async fn run_benchmark(
    app: AppHandle,
    job_id: String,
    model_path: String,
    ep: Option<String>,
) -> Result<BenchmarkResult, String> {
    let ep_name = ep.unwrap_or_else(|| "DML".to_string());
    // Prefer the python evaluation worker; fall back to `winml perf`.
    let bench_py = crate::winml::bridge::worker_script("evaluation/bench.py");
    let output: std::process::Output = if bench_py.exists() {
        let py = std::env::current_dir()
            .map(|c| {
                c.join("workers")
                    .join(".venv")
                    .join("Scripts")
                    .join("python.exe")
            })
            .unwrap_or_default();
        let py_bin = if py.exists() {
            py.to_string_lossy().to_string()
        } else {
            "python".to_string()
        };
        std::process::Command::new(py_bin)
            .args([
                bench_py.to_string_lossy().as_ref(),
                "--model",
                model_path.as_str(),
                "--ep",
                ep_name.as_str(),
                "--json",
            ])
            .output()
            .map_err(|e| e.to_string())?
    } else {
        let bin = crate::winml::bridge::winml_binary();
        std::process::Command::new(&bin)
            .args(["perf", "--model", model_path.as_str(), "--json"])
            .output()
            .map_err(|e| format!("benchmark backend missing ({e}); tried worker + winml"))?
    };

    if !output.status.success() {
        return Err(format!(
            "benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    // Workers print JSON on the last non-empty line; parse robustly.
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("{}");
    let v: serde_json::Value =
        serde_json::from_str(json_line).map_err(|e| format!("benchmark parse error: {e}"))?;

    let avg = v.get("avgMs").or_else(|| v.get("avg_ms")).and_then(|x| x.as_f64()).unwrap_or(0.0);
    let p50 = v.get("p50Ms").or_else(|| v.get("p50_ms")).and_then(|x| x.as_f64()).unwrap_or(avg);
    let p90 = v.get("p90Ms").or_else(|| v.get("p90_ms")).and_then(|x| x.as_f64()).unwrap_or(avg);
    let tput = v.get("throughput").and_then(|x| x.as_f64()).unwrap_or_else(|| {
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    });

    // Persist to SQLite best-effort.
    let state: tauri::State<'_, AppState> = app.state();
    let db = state.db.lock().await;
    let _ = db.insert_benchmark(&job_id, &ep_name, avg, p50, p90, tput, json_line);

    Ok(BenchmarkResult {
        job_id,
        ep: ep_name,
        avg_ms: avg,
        p50_ms: p50,
        p90_ms: p90,
        throughput: tput,
        raw: v,
    })
}

#[tauri::command]
pub async fn list_models(
    app: AppHandle,
) -> Result<Vec<crate::storage::db::ModelRow>, String> {
    let state: tauri::State<'_, AppState> = app.state();
    let db = state.db.lock().await;
    db.list_models().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn open_model_folder(app: AppHandle, model_id: String) -> Result<(), String> {
    let dir = if model_id.is_empty() {
        cache::models_dir()
    } else {
        cache::model_dir(&model_id)
    };
    let _ = cache::ensure_dirs();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    app.shell()
        .open(dir.to_string_lossy().to_string(), None)
        .map_err(|e| e.to_string())
}
