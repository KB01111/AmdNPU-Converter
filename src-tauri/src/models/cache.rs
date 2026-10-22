use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Stable app-data root: `%APPDATA%/npu-converter` (or platform equivalent).
pub fn app_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("npu-converter")
}

pub fn models_dir() -> PathBuf {
    app_data_dir().join("models")
}

pub fn cache_dir() -> PathBuf {
    app_data_dir().join("cache")
}

pub fn logs_dir() -> PathBuf {
    app_data_dir().join("logs")
}

pub fn ensure_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all(app_data_dir())?;
    std::fs::create_dir_all(models_dir())?;
    std::fs::create_dir_all(cache_dir())?;
    std::fs::create_dir_all(logs_dir())?;
    Ok(())
}

/// sha256 hex digest used as building block for cache keys.
pub fn cache_key(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for p in parts {
        hasher.update(p.as_bytes());
        hasher.update(b"\x00");
    }
    hex::encode(hasher.finalize())
}

/// Resolve per-model directories.
pub fn model_dir(model_id: &str) -> PathBuf {
    let safe: String = model_id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    models_dir().join(safe)
}

pub fn log_path_for_job(job_id: &str) -> PathBuf {
    logs_dir().join(format!("{job_id}.log"))
}
