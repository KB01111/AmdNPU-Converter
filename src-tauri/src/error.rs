use thiserror::Error;

/// Central error type for the converter backend.
/// Every Tauri command maps this to `String` (via `to_string()`)
/// so the frontend always receives a readable message.
#[derive(Debug, Error)]
pub enum ConverterError {
    #[error("database error: {0}")]
    Db(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("invalid HuggingFace URL or model id: {0}")]
    InvalidModel(String),

    #[error("model not found: {0}")]
    NotFound(String),

    #[error("job not found: {0}")]
    JobNotFound(String),

    #[error("job already cancelled or finished: {0}")]
    JobClosed(String),

    #[error("worker process failed (exit {code}): {message}")]
    WorkerFailed { code: i32, message: String },

    #[error("winml binary not found: {0}")]
    WinmlMissing(String),

    #[error("io error: {0}")]
    Io(String),

    #[error("serialization error: {0}")]
    Serde(String),

    #[error("hardware probe failed: {0}")]
    Hardware(String),

    #[error("benchmark failed: {0}")]
    Benchmark(String),

    #[error("{0}")]
    Other(String),
}

impl From<rusqlite::Error> for ConverterError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Db(e.to_string())
    }
}

impl From<reqwest::Error> for ConverterError {
    fn from(e: reqwest::Error) -> Self {
        Self::Network(e.to_string())
    }
}

impl From<std::io::Error> for ConverterError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

impl From<serde_json::Error> for ConverterError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e.to_string())
    }
}

impl From<anyhow::Error> for ConverterError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e.to_string())
    }
}

/// Helper: convert any Display error into the String Tauri commands return.
pub fn err_to_string<E: ToString>(e: E) -> String {
    e.to_string()
}
