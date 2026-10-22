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
            "UPDATE jobs SET status = ?1, current_step = ?2, progress = ?3, updated_at = ?4 WHERE id = ?5",
            params![status, step, progress, now, job_id],
        )?;
        Ok(())
    }

    pub fn finish_job(&self, job_id: &str, status: &str) -> Result<(), ConverterError> {
        let now = Utc::now().to_rfc3339();
        let progress = if status == "completed" { 100.0 } else { 0.0_f64.max(progress_for(status)) };
        let _ = progress;
        self.conn.execute(
            "UPDATE jobs SET status = ?1, progress = CASE WHEN ?1 = 'completed' THEN 100.0 ELSE progress END, updated_at = ?2 WHERE id = ?3",
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
            out.push(r.map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))?);
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
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
            })?))
        } else {
            Ok(None)
        }
    }

    pub fn upsert_model(&self, row: &ModelRow) -> Result<(), ConverterError> {
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
        Ok(())
    }

    pub fn list_models(&self) -> Result<Vec<ModelRow>, ConverterError> {
        let mut stmt = self.conn.prepare(
            "SELECT model_id, hf_id, arch, params, format, local_path, onnx_path, compiled_path, cache_key, created_at
             FROM models ORDER BY created_at DESC",
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
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
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
         );",
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
