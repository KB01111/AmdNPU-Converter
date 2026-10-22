use crate::{error::ConverterError, jobs::LogLine, models::cache};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tauri::Manager;
use tokio::io::{AsyncBufReadExt, BufReader};

pub fn development_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}
pub fn resource_path(resources: Option<&Path>, relative: &str) -> PathBuf {
    if let Some(root) = resources {
        let path = root.join(relative);
        if path.is_file() || path.is_dir() {
            return path;
        }
    }
    if !cfg!(debug_assertions) {
        return resources
            .map(Path::to_path_buf)
            .unwrap_or_else(|| {
                std::env::current_exe()
                    .unwrap_or_default()
                    .parent()
                    .unwrap_or(Path::new("/"))
                    .to_path_buf()
            })
            .join(relative);
    }
    development_root().join(relative)
}
pub fn app_resource(app: &tauri::AppHandle, relative: &str) -> PathBuf {
    resource_path(app.path().resource_dir().ok().as_deref(), relative)
}
pub fn worker_environment(script: &str) -> &'static str {
    if script.starts_with("quark/") {
        ".venv-quark"
    } else if script.starts_with("winml/") || (cfg!(windows) && script == "evaluation/bench.py") {
        ".venv-winml"
    } else {
        ".venv"
    }
}
pub fn worker_python(workers: &Path, script: &str) -> PathBuf {
    workers
        .join(worker_environment(script))
        .join(if cfg!(windows) {
            "Scripts/python.exe"
        } else {
            "bin/python"
        })
}
pub fn winml_binary() -> PathBuf {
    if let Some(bin) = std::env::var_os("WINML_BIN") {
        return PathBuf::from(bin);
    }
    let root = worker_env_root();
    for venv in [".venv-winml", ".venv"] {
        let candidate = root.join(venv).join(if cfg!(windows) {
            "Scripts/winml.exe"
        } else {
            "bin/winml"
        });
        if candidate.is_file() {
            return candidate;
        }
    }
    which::which(if cfg!(windows) { "winml.exe" } else { "winml" }).unwrap_or_default()
}
pub fn worker_env_root() -> PathBuf {
    std::env::var_os("NPU_CONVERTER_WORKERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(debug_assertions) {
                development_root().join("workers")
            } else {
                cache::app_data_dir().join("workers")
            }
        })
}
pub struct ProcessOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}
pub async fn wait_cancel(flag: &Arc<AtomicBool>) {
    while !flag.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

pub async fn run_worker(
    app: &tauri::AppHandle,
    script: &str,
    args: &[String],
    job: Option<&str>,
    flag: &Arc<AtomicBool>,
) -> Result<ProcessOutput, ConverterError> {
    let script_path = app_resource(app, &format!("workers/{script}"));
    if !script_path.is_file() {
        return Err(ConverterError::Other(format!(
            "Worker resource missing: {}",
            script_path.display()
        )));
    }
    let root = worker_env_root();
    let python = worker_python(&root, script);
    if !python.is_file() {
        return Err(ConverterError::Other(format!("Missing Python worker environment: {}. Run workers/bootstrap.ps1 and set NPU_CONVERTER_WORKERS_DIR to its workers directory for an installed app.",python.display())));
    }
    let mut command = tokio::process::Command::new(python);
    command
        .arg(script_path)
        .args(args)
        .env(
            "HF_HUB_OFFLINE",
            if script.starts_with("exporters/") {
                "1"
            } else {
                "0"
            },
        )
        .env(
            "TRANSFORMERS_OFFLINE",
            if script.starts_with("exporters/") {
                "1"
            } else {
                "0"
            },
        );
    if script.starts_with("winml/") {
        let bin = winml_binary();
        if !bin.as_os_str().is_empty() {
            command.env("WINML_BIN", bin);
        }
    }
    use tauri::Emitter;
    let app = app.clone();
    let job = job.map(str::to_owned);
    let log: LogCallback = Arc::new(move |stream, line| {
        if let Some(id) = &job {
            append_log(&cache::log_path_for_job(id), stream, line);
            app.emit(
                "job://progress",
                serde_json::json!({"jobId":id,"stream":stream,"line":line}),
            )
            .ok();
        }
    });
    run_command(command, Some(log), flag.clone()).await
}

type LogCallback = Arc<dyn Fn(&str, &str) + Send + Sync>;
async fn stream<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    stream: &str,
    log: Option<LogCallback>,
) -> Result<String, ConverterError> {
    let mut lines = BufReader::new(reader).lines();
    let mut output = String::new();
    while let Some(line) = lines.next_line().await? {
        // Keep subprocess diagnostics bounded even for verbose exporters.
        output.push_str(&line);
        output.push('\n');
        if output.len() > 4 * 1024 * 1024 {
            let mut cut = output.len() - 4 * 1024 * 1024;
            while !output.is_char_boundary(cut) {
                cut += 1;
            }
            output.drain(..cut);
        }
        if let Some(log) = &log {
            log(stream, &line);
        }
    }
    Ok(output)
}

#[cfg(not(windows))]
async fn stop_child(child: &mut tokio::process::Child) -> Result<(), ConverterError> {
    if child.try_wait()?.is_none() {
        child.kill().await?;
    }
    child.wait().await?;
    Ok(())
}

pub async fn run_command(
    mut command: tokio::process::Command,
    log: Option<LogCallback>,
    flag: Arc<AtomicBool>,
) -> Result<ProcessOutput, ConverterError> {
    if flag.load(Ordering::SeqCst) {
        return Err(ConverterError::Cancelled);
    }
    command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    command.creation_flags(
        windows_sys::Win32::System::Threading::CREATE_NO_WINDOW
            | windows_sys::Win32::System::Threading::CREATE_SUSPENDED,
    );
    #[cfg(windows)]
    let ownership = super::process_job::ProcessJob::new()?;
    let mut child = command
        .spawn()
        .map_err(|e| ConverterError::Other(format!("Worker launch failed: {e}")))?;
    #[cfg(windows)]
    if let Err(error) = ownership.assign_and_resume(&child) {
        child.kill().await?;
        child.wait().await?;
        return Err(ConverterError::Other(format!(
            "Worker ownership setup failed: {error}"
        )));
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ConverterError::Other("Missing worker stdout".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ConverterError::Other("Missing worker stderr".into()))?;
    let out = tokio::spawn(stream(stdout, "stdout", log.clone()));
    let err = tokio::spawn(stream(stderr, "stderr", log));
    let status = tokio::select! {
        result=child.wait()=>{
            #[cfg(windows)]
            ownership.terminate_and_wait().await?;
            result.map_err(ConverterError::from)
        },
        _=wait_cancel(&flag)=>{
            #[cfg(windows)]
            {
                ownership.terminate_and_wait().await?;
                child.wait().await?;
                Err(ConverterError::Cancelled)
            }
            #[cfg(not(windows))]
            match stop_child(&mut child).await {Ok(())=>Err(ConverterError::Cancelled),Err(error)=>Err(error)}
        }
    };
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            out.abort();
            err.abort();
            return Err(error);
        }
    };
    let stdout = out
        .await
        .map_err(|e| ConverterError::Other(format!("Worker stdout failed: {e}")))??;
    let stderr = err
        .await
        .map_err(|e| ConverterError::Other(format!("Worker stderr failed: {e}")))??;
    Ok(ProcessOutput {
        code: status.code().unwrap_or(-1),
        stdout,
        stderr,
    })
}
pub fn append_log(path: &Path, stream: &str, line: &str) {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        if let Ok(json) = serde_json::to_string(&LogLine {
            ts: chrono::Utc::now().to_rfc3339(),
            stream: stream.into(),
            line: line.into(),
        }) {
            writeln!(file, "{json}").ok();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn environments_are_isolated() {
        let root = Path::new("C:/workers");
        assert!(worker_python(root, "quark/quantize.py")
            .to_string_lossy()
            .contains(".venv-quark"));
        assert!(worker_python(root, "winml/run.py")
            .to_string_lossy()
            .contains(".venv-winml"));
        assert_eq!(
            worker_environment("evaluation/bench.py"),
            if cfg!(windows) {
                ".venv-winml"
            } else {
                ".venv"
            }
        );
        assert_eq!(worker_environment("evaluation/compare.py"), ".venv");
    }
    #[test]
    fn resources_prefer_installed_mapping() {
        let root = std::env::temp_dir().join(Uuid::new_v4().to_string());
        std::fs::create_dir_all(root.join("workers")).unwrap();
        std::fs::write(root.join("workers/test.py"), "pass").unwrap();
        assert_eq!(
            resource_path(Some(&root), "workers/test.py"),
            root.join("workers/test.py")
        );
        assert!(resource_path(None, "workers/evaluation/bench.py").is_absolute());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[tokio::test]
    #[cfg(windows)]
    async fn cancellation_stops_running_child() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut cmd = tokio::process::Command::new("powershell");
        cmd.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Write-Output $PID; Start-Sleep -Seconds 30",
        ]);
        let pid = Arc::new(std::sync::Mutex::new(None::<u32>));
        let capture = pid.clone();
        let log: LogCallback = Arc::new(move |stream, line| {
            if stream == "stdout" {
                if let Ok(value) = line.trim().parse::<u32>() {
                    *capture.lock().unwrap() = Some(value);
                }
            }
        });
        let control = flag.clone();
        let wait_for_pid = pid.clone();
        let started = std::time::Instant::now();
        tokio::spawn(async move {
            for _ in 0..100 {
                if wait_for_pid.lock().unwrap().is_some() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            control.store(true, Ordering::SeqCst);
        });
        assert!(matches!(
            run_command(cmd, Some(log), flag).await,
            Err(ConverterError::Cancelled)
        ));
        let pid = pid
            .lock()
            .unwrap()
            .expect("worker reached the running state");
        let system = sysinfo::System::new_all();
        assert!(
            system.process(sysinfo::Pid::from_u32(pid)).is_none(),
            "cancelled worker still exists"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(10));
    }
    use uuid::Uuid;
    #[tokio::test]
    #[cfg(windows)]
    async fn successful_worker_does_not_leave_background_descendants() {
        let mut cmd = tokio::process::Command::new("powershell");
        cmd.args(["-NoProfile","-NonInteractive","-Command","$descendant=Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 30' -PassThru; Write-Output $descendant.Id"]);
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            run_command(cmd, None, Arc::new(AtomicBool::new(false))),
        )
        .await
        .expect("normal worker cleanup completed")
        .unwrap();
        assert_eq!(output.code, 0);
        let descendant = output.stdout.trim().parse::<u32>().unwrap();
        assert!(
            sysinfo::System::new_all()
                .process(sysinfo::Pid::from_u32(descendant))
                .is_none(),
            "successful worker leaked a grandchild"
        );
    }
    #[tokio::test]
    #[cfg(windows)]
    async fn dropping_worker_ownership_stops_grandchild() {
        let mut cmd = tokio::process::Command::new("powershell");
        cmd.args(["-NoProfile","-NonInteractive","-Command", "$descendant=Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 30' -PassThru; Write-Output $descendant.Id; Start-Sleep -Seconds 30"]);
        let pid = Arc::new(std::sync::Mutex::new(None::<u32>));
        let capture = pid.clone();
        let log: LogCallback = Arc::new(move |stream, line| {
            if stream == "stdout" {
                if let Ok(value) = line.trim().parse::<u32>() {
                    *capture.lock().unwrap() = Some(value);
                }
            }
        });
        let task = tokio::spawn(run_command(
            cmd,
            Some(log),
            Arc::new(AtomicBool::new(false)),
        ));
        for _ in 0..100 {
            if pid.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let descendant = pid
            .lock()
            .unwrap()
            .expect("worker launched a real grandchild");
        task.abort();
        let _ = task.await;
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let alive = sysinfo::System::new_all()
            .process(sysinfo::Pid::from_u32(descendant))
            .is_some();
        if alive {
            tokio::process::Command::new("taskkill")
                .args(["/PID", &descendant.to_string(), "/T", "/F"])
                .creation_flags(0x08000000)
                .output()
                .await
                .unwrap();
        }
        assert!(!alive, "grandchild survived release of worker ownership");
    }
}
