use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::ConverterError;

// ---------------------------------------------------------------------------
// Row types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRow {
    pub id: String,
    pub model_id: String,
    pub config_json: String,
    pub status: String,
    pub current_step: String,
    pub progress: f64,
    pub log_path: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRow {
    pub model_id: String,
    pub hf_id: String,
    pub arch: String,
    pub params: String,
    pub format: String,
    pub local_path: String,
    pub onnx_path: String,
    pub compiled_path: String,
    pub cache_key: String,
    pub created_at: String,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub target_ep: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkRow {
    pub id: i64,
    pub job_id: String,
    pub ep: String,
    pub avg_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub throughput: f64,
    pub raw_json: String,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Db wrapper
// ---------------------------------------------------------------------------

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn job_details(&self, id: &str) -> Result<(Option<String>, Vec<String>), ConverterError> {
        use rusqlite::OptionalExtension;
        let row: Option<(Option<String>, String)> = self
            .conn
            .query_row(
                "SELECT error,artifacts_json FROM job_details WHERE job_id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        match row {
            Some((error, json)) => Ok((error, serde_json::from_str(&json)?)),
            None => Ok((None, vec![])),
        }
    }

    pub fn save_job_details(
        &self,
        id: &str,
        error: Option<&str>,
        artifacts: &[String],
    ) -> Result<(), ConverterError> {
        self.conn.execute("INSERT INTO job_details(job_id,error,artifacts_json) VALUES(?1,?2,?3) ON CONFLICT(job_id) DO UPDATE SET error=excluded.error,artifacts_json=excluded.artifacts_json", params![id,error,serde_json::to_string(artifacts)?])?;
        Ok(())
    }

    pub fn save_benchmark_report(
        &self,
        context: &str,
        model_path: &str,
        json: &str,
    ) -> Result<(), ConverterError> {
        self.conn.execute("INSERT INTO benchmark_reports(context,model_path,raw_json,created_at) VALUES(?1,?2,?3,?4)",params![context,model_path,json,Utc::now().to_rfc3339()])?;
        Ok(())
    }
    pub fn new(conn: Connection) -> Self {
        Self { conn }
    }

    pub fn insert_job(&self, row: &JobRow) -> Result<(), ConverterError> {
        self.conn.execute(
            "INSERT INTO jobs (id, model_id, config_json, status, current_step, progress, log_path, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                row.id,
                row.model_id,
                row.config_json,
                row.status,
                row.current_step,
                row.progress,
                row.log_path,
                row.created_at,
                row.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn update_job_progress(
        &self,
        job_id: &str,
        status: &str,
        step: &str,
        progress: f64,
    ) -> Result<(), ConverterError> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE jobs SET status = ?1, current_step = ?2, progress = ?3, updated_at = ?4 WHERE id = ?5 AND status IN ('queued','running','cancelling')",
            params![status, step, progress, now, job_id],
        )?;
        Ok(())
    }

    pub fn finish_job(&self, job_id: &str, status: &str) -> Result<(), ConverterError> {
        let now = Utc::now().to_rfc3339();
        let progress = if status == "completed" {
            100.0
        } else {
            0.0_f64.max(progress_for(status))
        };
        let _ = progress;
        self.conn.execute(
            "UPDATE jobs SET status = ?1, progress = CASE WHEN ?1 = 'completed' THEN 100.0 ELSE progress END, updated_at = ?2 WHERE id = ?3 AND status IN ('queued','running','cancelling')",
            params![status, now, job_id],
        )?;
        Ok(())
    }

    pub fn list_jobs(&self, limit: i64) -> Result<Vec<JobRow>, ConverterError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_id, config_json, status, current_step, progress, log_path, created_at, updated_at
             FROM jobs ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], |r| {
            Ok(JobRow {
                id: r.get(0)?,
                model_id: r.get(1)?,
                config_json: r.get(2)?,
                status: r.get(3)?,
                current_step: r.get(4)?,
                progress: r.get(5)?,
                log_path: r.get(6)?,
                created_at: r.get(7)?,
                updated_at: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?);
        }
        Ok(out)
    }

    pub fn get_job(&self, job_id: &str) -> Result<Option<JobRow>, ConverterError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_id, config_json, status, current_step, progress, log_path, created_at, updated_at
             FROM jobs WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![job_id], |r| {
            Ok(JobRow {
                id: r.get(0)?,
                model_id: r.get(1)?,
                config_json: r.get(2)?,
                status: r.get(3)?,
                current_step: r.get(4)?,
                progress: r.get(5)?,
                log_path: r.get(6)?,
                created_at: r.get(7)?,
                updated_at: r.get(8)?,
            })
        })?;
        if let Some(r) = rows.next() {
            Ok(Some(r.map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?))
        } else {
            Ok(None)
        }
    }

    pub fn upsert_model(&self, row: &ModelRow) -> Result<(), ConverterError> {
        let transaction = self.conn.unchecked_transaction()?;
        self.conn.execute(
            "INSERT INTO models (model_id, hf_id, arch, params, format, local_path, onnx_path, compiled_path, cache_key, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(model_id) DO UPDATE SET
               hf_id = excluded.hf_id, arch = excluded.arch, params = excluded.params,
               format = excluded.format, local_path = excluded.local_path,
               onnx_path = excluded.onnx_path, compiled_path = excluded.compiled_path,
               cache_key = excluded.cache_key",
            params![
                row.model_id,
                row.hf_id,
                row.arch,
                row.params,
                row.format,
                row.local_path,
                row.onnx_path,
                row.compiled_path,
                row.cache_key,
                row.created_at
            ],
        )?;
        self.conn.execute("INSERT INTO model_details(model_id,job_id,target_ep) VALUES(?1,?2,?3) ON CONFLICT(model_id) DO UPDATE SET job_id=excluded.job_id,target_ep=excluded.target_ep",params![row.model_id,row.job_id,row.target_ep])?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_models(&self) -> Result<Vec<ModelRow>, ConverterError> {
        let mut stmt = self.conn.prepare(
            "SELECT m.model_id, hf_id, arch, params, format, local_path, onnx_path, compiled_path, cache_key, created_at,d.job_id,d.target_ep
             FROM models m LEFT JOIN model_details d ON m.model_id=d.model_id ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ModelRow {
                model_id: r.get(0)?,
                hf_id: r.get(1)?,
                arch: r.get(2)?,
                params: r.get(3)?,
                format: r.get(4)?,
                local_path: r.get(5)?,
                onnx_path: r.get(6)?,
                compiled_path: r.get(7)?,
                cache_key: r.get(8)?,
                created_at: r.get(9)?,
                job_id: r.get(10)?,
                target_ep: r.get(11)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?);
        }
        Ok(out)
    }

    pub fn insert_benchmark(
        &self,
        job_id: &str,
        ep: &str,
        avg_ms: f64,
        p50_ms: f64,
        p90_ms: f64,
        throughput: f64,
        raw_json: &str,
    ) -> Result<(), ConverterError> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO benchmarks (job_id, ep, avg_ms, p50_ms, p90_ms, throughput, raw_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![job_id, ep, avg_ms, p50_ms, p90_ms, throughput, raw_json, now],
        )?;
        Ok(())
    }
}

fn progress_for(status: &str) -> f64 {
    match status {
        "completed" => 100.0,
        _ => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Free functions required by the spec
// ---------------------------------------------------------------------------

/// Initialise (or open) the SQLite database under `<app_data>/converter.db`.
pub fn acquire_instance_lock(app_data_dir: &Path) -> Result<std::fs::File, ConverterError> {
    std::fs::create_dir_all(app_data_dir)?;
    let mut options = std::fs::OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(0);
    }
    let file = options
        .open(app_data_dir.join("instance.lock"))
        .map_err(|e| {
            ConverterError::Other(format!(
                "Cannot acquire application lock; another instance may be running: {e}"
            ))
        })?;
    #[cfg(not(windows))]
    file.try_lock()
        .map_err(|e| ConverterError::Other(format!("Another instance may be running: {e}")))?;
    Ok(file)
}

pub fn init_db(app_data_dir: &Path) -> Result<Db, ConverterError> {
    std::fs::create_dir_all(app_data_dir).map_err(|e| ConverterError::Io(e.to_string()))?;
    let db_path = app_data_dir.join("converter.db");
    let conn = Connection::open(&db_path).map_err(|e| ConverterError::Db(e.to_string()))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         CREATE TABLE IF NOT EXISTS jobs (
           id TEXT PRIMARY KEY,
           model_id TEXT NOT NULL,
           config_json TEXT NOT NULL DEFAULT '{}',
           status TEXT NOT NULL DEFAULT 'queued',
           current_step TEXT NOT NULL DEFAULT 'inspection',
           progress REAL NOT NULL DEFAULT 0,
           log_path TEXT NOT NULL DEFAULT '',
           created_at TEXT NOT NULL,
           updated_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS models (
           model_id TEXT PRIMARY KEY,
           hf_id TEXT NOT NULL DEFAULT '',
           arch TEXT NOT NULL DEFAULT '',
           params TEXT NOT NULL DEFAULT '',
           format TEXT NOT NULL DEFAULT '',
           local_path TEXT NOT NULL DEFAULT '',
           onnx_path TEXT NOT NULL DEFAULT '',
           compiled_path TEXT NOT NULL DEFAULT '',
           cache_key TEXT NOT NULL DEFAULT '',
           created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS model_details(model_id TEXT PRIMARY KEY,job_id TEXT,target_ep TEXT);
         CREATE TABLE IF NOT EXISTS benchmarks (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           job_id TEXT NOT NULL,
           ep TEXT NOT NULL DEFAULT '',
           avg_ms REAL NOT NULL DEFAULT 0,
           p50_ms REAL NOT NULL DEFAULT 0,
           p90_ms REAL NOT NULL DEFAULT 0,
           throughput REAL NOT NULL DEFAULT 0,
           raw_json TEXT NOT NULL DEFAULT '{}',
           created_at TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS job_details(job_id TEXT PRIMARY KEY,error TEXT,artifacts_json TEXT NOT NULL DEFAULT '[]');
         CREATE TABLE IF NOT EXISTS benchmark_reports(id INTEGER PRIMARY KEY,context TEXT NOT NULL,model_path TEXT NOT NULL,raw_json TEXT NOT NULL,created_at TEXT NOT NULL);
         INSERT INTO job_details(job_id,error,artifacts_json) SELECT id,'Interrupted by application restart','[]' FROM jobs WHERE status IN ('queued','running','cancelling') ON CONFLICT(job_id) DO UPDATE SET error=excluded.error;
         UPDATE jobs SET status='interrupted' WHERE status IN ('queued','running','cancelling');",
    )
    .map_err(|e| ConverterError::Db(e.to_string()))?;
    Ok(Db::new(conn))
}

pub fn insert_job(db: &Db, row: &JobRow) -> Result<(), ConverterError> {
    db.insert_job(row)
}

pub fn update_job_progress(
    db: &Db,
    job_id: &str,
    status: &str,
    step: &str,
    progress: f64,
) -> Result<(), ConverterError> {
    db.update_job_progress(job_id, status, step, progress)
}

pub fn finish_job(db: &Db, job_id: &str, status: &str) -> Result<(), ConverterError> {
    db.finish_job(job_id, status)
}

pub fn list_jobs(db: &Db, limit: i64) -> Result<Vec<JobRow>, ConverterError> {
    db.list_jobs(limit)
}

pub fn upsert_model(db: &Db, row: &ModelRow) -> Result<(), ConverterError> {
    db.upsert_model(row)
}

pub fn list_models(db: &Db) -> Result<Vec<ModelRow>, ConverterError> {
    db.list_models()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_keeps_producing_job_and_provider_across_restart() {
        let root = std::env::temp_dir().join(format!("converter-model-{}", uuid::Uuid::new_v4()));
        let db = init_db(&root).unwrap();
        let row:ModelRow=serde_json::from_value(serde_json::json!({"modelId":"org/model","hfId":"org/model","arch":"test","params":"1","format":"onnx","localPath":"C:/job-a","onnxPath":"C:/job-a/model.onnx","compiledPath":"C:/job-a/model.cuda.onnx","cacheKey":"key","createdAt":"now","jobId":"job-a","targetEp":"cuda"})).unwrap();
        db.upsert_model(&row).unwrap();
        drop(db);
        let db = init_db(&root).unwrap();
        let model = serde_json::to_value(&db.list_models().unwrap()[0]).unwrap();
        drop(db);
        std::fs::remove_dir_all(root).unwrap();
        assert_eq!(model["jobId"], "job-a");
        assert_eq!(model["targetEp"], "cuda");
    }
    #[test]
    fn only_one_instance_can_recover_database() {
        let root = std::env::temp_dir().join(format!("converter-lock-{}", uuid::Uuid::new_v4()));
        let first = acquire_instance_lock(&root).unwrap();
        assert!(acquire_instance_lock(&root).is_err());
        drop(first);
        let second = acquire_instance_lock(&root).unwrap();
        drop(second);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn restart_interrupts_open_jobs_and_preserves_terminal_jobs() {
        let root = std::env::temp_dir().join(format!("converter-db-{}", uuid::Uuid::new_v4()));
        let db = init_db(&root).unwrap();
        for (id, status) in [("live", "running"), ("done", "completed")] {
            db.insert_job(&JobRow {
                id: id.into(),
                model_id: "org/model".into(),
                config_json: "{}".into(),
                status: status.into(),
                current_step: "export".into(),
                progress: 40.0,
                log_path: String::new(),
                created_at: "now".into(),
                updated_at: "now".into(),
            })
            .unwrap();
        }
        db.save_job_details("live", None, &["C:/job/model.onnx".into()])
            .unwrap();
        db.save_job_details(
            "done",
            Some("retained terminal detail"),
            &["C:/done/model.onnx".into()],
        )
        .unwrap();
        drop(db);
        let db = init_db(&root).unwrap();
        assert_eq!(db.get_job("live").unwrap().unwrap().status, "interrupted");
        assert_eq!(db.get_job("done").unwrap().unwrap().status, "completed");
        assert!(db
            .job_details("live")
            .unwrap()
            .0
            .unwrap()
            .contains("restart"));
        assert_eq!(db.job_details("live").unwrap().1, vec!["C:/job/model.onnx"]);
        assert_eq!(
            db.job_details("done").unwrap().0.as_deref(),
            Some("retained terminal detail")
        );
        db.update_job_progress("done", "running", "download", 5.0)
            .unwrap();
        db.finish_job("done", "cancelled").unwrap();
        assert_eq!(db.get_job("done").unwrap().unwrap().status, "completed");
        drop(db);
        std::fs::remove_dir_all(root).unwrap();
    }
}
