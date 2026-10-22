pub mod commands;
pub mod error;
pub mod hardware;
pub mod jobs;
pub mod models;
pub mod storage;
pub mod winml;

use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc};

use tokio::sync::Mutex;
use uuid::Uuid;

/// Handle to a running conversion job, kept in the in-memory registry.
pub struct JobHandle {
    pub cancel_flag: Arc<AtomicBool>,
}

/// Global application state managed by Tauri.
pub struct AppState {
    pub db: Mutex<storage::db::Db>,
    pub jobs: Arc<Mutex<HashMap<Uuid, JobHandle>>>,
}

impl AppState {
    pub fn new(db: storage::db::Db) -> Self {
        Self {
            db: Mutex::new(db),
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Bootstraps state + runs the Tauri application.
/// Called from `main.rs`.
pub fn run() {
    let app_data = crate::models::cache::app_data_dir();
    let _instance_lock = storage::db::acquire_instance_lock(&app_data)
        .expect("another app instance may already own conversion jobs");
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
            commands::retry_job,
            commands::run_benchmark,
            commands::list_models,
            commands::open_model_folder,
            commands::list_recipes,
            commands::load_recipe,
            commands::save_recipe,
            commands::export_report,
        ])
        .setup(|_app| {
            // SQLite is already initialised above (before manage).
            // Heavy per-boot migrations run synchronously in init_db.
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
