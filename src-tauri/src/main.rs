mod commands;
mod error;
mod hardware;
mod jobs;
mod models;
mod storage;
mod winml;

use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc};

/// Handle to a running conversion job (binary-crate mirror of the lib type).
pub struct JobHandle {
    pub model_id: String,
    pub cancel_flag: Arc<AtomicBool>,
}

/// Global application state (binary-crate mirror of the lib type).
pub struct AppState {
    pub db: tokio::sync::Mutex<storage::db::Db>,
    pub jobs: Arc<tokio::sync::Mutex<HashMap<uuid::Uuid, JobHandle>>>,
}

impl AppState {
    pub fn new(db: storage::db::Db) -> Self {
        Self {
            db: tokio::sync::Mutex::new(db),
            jobs: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }
}

#[tokio::main]
async fn main() {
    let app_data = models::cache::app_data_dir();
    let db = storage::db::init_db(&app_data).expect("failed to initialise SQLite database");
    let state = AppState::new(db);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::hf_inspect,
            commands::get_hardware_info,
            commands::start_conversion,
            commands::get_job_status,
            commands::list_jobs,
            commands::cancel_job,
            commands::run_benchmark,
            commands::list_models,
            commands::open_model_folder,
        ])
        .setup(|_app| {
            // SQLite initialised before manage(); nothing heavy left to do here.
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
