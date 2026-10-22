use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tauri::Emitter;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::error::ConverterError;
use crate::models::cache;

// ---------------------------------------------------------------------------
// Binary resolution
// ---------------------------------------------------------------------------

/// Locate the winml executable, searching in order:
/// 1. `<exe_dir>/../bin/winml.exe` (Tauri sidecar layout)
/// 2. `<cwd>/workers/.venv/Scripts/winml.exe`
/// 3. `<cwd>/workers/winml.exe`
/// 4. `PATH` via `which`
pub fn winml_binary() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [
                dir.join("bin").join("winml.exe"),
                dir.join("bin").join("winml"),
                dir.join("winml.exe"),
            ] {
                if cand.exists() {
                    return cand;
                }
            }
            if let Some(parent) = dir.parent() {
                let c = parent.join("bin").join("winml.exe");
                if c.exists() {
                    return c;
                }
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for cand in [
            cwd.join("workers").join(".venv").join("Scripts").join("winml.exe"),
            cwd.join("workers").join("winml.exe"),
            cwd.join("bin").join("winml.exe"),
        ] {
            if cand.exists() {
                return cand;
            }
        }
    }
    if let Ok(p) = which::which("winml.exe") {
        return p;
    }
    if let Ok(p) = which::which("winml") {
        return p;
    }
    PathBuf::from("winml.exe")
}

/// Locate a python worker script under `workers/`.
pub fn worker_script(name: &str) -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        let c = cwd.join("workers").join(name);
        if c.exists() {
            return c;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let c = dir.join("workers").join(name);
            if c.exists() {
                return c;
            }
        }
    }
    PathBuf::from(format!("workers/{name}"))
}

fn python_binary() -> String {
    if let Ok(cwd) = std::env::current_dir() {
        let venv = cwd
            .join("workers")
            .join(".venv")
            .join("Scripts")
            .join("python.exe");
        if venv.exists() {
            return venv.to_string_lossy().to_string();
        }
    }
    "python".to_string()
}

// ---------------------------------------------------------------------------
// Streaming runner
// ---------------------------------------------------------------------------

/// Run winml with streamed stdout/stderr.
/// Each line is appended to `$APPDATA/npu-converter/logs/{job_id}.log`
/// and re-emitted as `job://progress` log events. Returns exit code.
pub async fn run_winml(
    args: &[String],
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    let bin = winml_binary();
    run_process(bin.to_string_lossy().as_ref(), args, job_id, app).await
}

/// Run a python worker script with streamed output.
pub async fn run_python_worker(
    script: &str,
    args: &[String],
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    let script_path = worker_script(script);
    let py = python_binary();
    let mut full = vec![script_path.to_string_lossy().to_string()];
    full.extend_from_slice(args);
    run_process(&py, &full, job_id, app).await
}

async fn run_process(
    program: &str,
    args: &[String],
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    let _ = cache::ensure_dirs();
    let log_path = cache::log_path_for_job(job_id);

    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        ConverterError::WorkerFailed {
            code: -1,
            message: format!("failed to spawn `{program}`: {e}"),
        }
    })?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Stream stdout lines.
    let app_c = app.clone();
    let job = job_id.to_string();
    let log_c = log_path.clone();
    let out_task = tokio::spawn(async move {
        if let Some(out) = stdout {
            let mut lines = BufReader::new(out).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                append_log(&log_c, &format!("[out] {line}"));
                let _ = app_c.emit("job://progress", serde_json::json!({
                    "jobId": job,
                    "stream": "stdout",
                    "line": line,
                }));
            }
        }
    });

    let app_c = app.clone();
    let job = job_id.to_string();
    let log_c = log_path.clone();
    let err_task = tokio::spawn(async move {
        if let Some(err) = stderr {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                append_log(&log_c, &format!("[err] {line}"));
                let _ = app_c.emit("job://progress", serde_json::json!({
                    "jobId": job,
                    "stream": "stderr",
                    "line": line,
                }));
            }
        }
    });

    let status = child.wait().await.map_err(|e| ConverterError::WorkerFailed {
        code: -1,
        message: format!("failed waiting for `{program}`: {e}"),
    })?;
    let _ = tokio::join!(out_task, err_task);
    Ok(status.code().unwrap_or(-1))
}

fn append_log(path: &std::path::Path, line: &str) {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

// ---------------------------------------------------------------------------
// Typed helpers — thin arg builders over run_winml
// ---------------------------------------------------------------------------

pub async fn inspect(
    model: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[ "inspect".to_string(), "--model".to_string(), model.to_string() ],
        job_id,
        app,
    )
    .await
}

pub async fn export(
    model: &str,
    out: &str,
    opset: u32,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "export".to_string(),
            "--model".to_string(), model.to_string(),
            "--out".to_string(), out.to_string(),
            "--opset".to_string(), opset.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn optimize(
    onnx: &str,
    out: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "optimize".to_string(),
            "--in".to_string(), onnx.to_string(),
            "--out".to_string(), out.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn quantize(
    onnx: &str,
    out: &str,
    precision: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "quantize".to_string(),
            "--in".to_string(), onnx.to_string(),
            "--out".to_string(), out.to_string(),
            "--precision".to_string(), precision.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn compile(
    onnx: &str,
    out: &str,
    ep: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "compile".to_string(),
            "--in".to_string(), onnx.to_string(),
            "--out".to_string(), out.to_string(),
            "--ep".to_string(), ep.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn build(
    model: &str,
    out_dir: &str,
    config_json: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "build".to_string(),
            "--model".to_string(), model.to_string(),
            "--out-dir".to_string(), out_dir.to_string(),
            "--config".to_string(), config_json.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn perf(
    compiled: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &["perf".to_string(), "--model".to_string(), compiled.to_string()],
        job_id,
        app,
    )
    .await
}

pub async fn eval(
    compiled: &str,
    dataset: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "eval".to_string(),
            "--model".to_string(), compiled.to_string(),
            "--dataset".to_string(), dataset.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

pub async fn config(
    key: &str,
    value: &str,
    job_id: &str,
    app: &tauri::AppHandle,
) -> Result<i32, ConverterError> {
    run_winml(
        &[
            "config".to_string(),
            "--key".to_string(), key.to_string(),
            "--value".to_string(), value.to_string(),
        ],
        job_id,
        app,
    )
    .await
}

/// sha256 hex over model sha + config + quant + hw + ep + driver.
pub fn compute_cache_key(
    model_sha: &str,
    config: &str,
    quant: &str,
    hw: &str,
    ep: &str,
    driver: &str,
) -> String {
    let mut h = Sha256::new();
    for part in [model_sha, config, quant, hw, ep, driver] {
        h.update(part.as_bytes());
        h.update(b"\x00");
    }
    hex::encode(h.finalize())
}
