use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::error::ConverterError;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuInfo {
    pub brand: String,
    pub vendor: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub frequency_mhz: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub driver_version: String,
    pub vendor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NpuInfo {
    pub present: bool,
    pub name: String,
    pub driver_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WinmlInfo {
    pub available: bool,
    pub binary_path: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
    pub npu: NpuInfo,
    pub winml: WinmlInfo,
    pub os_name: String,
    pub os_version: String,
    pub total_memory_gb: f64,
    pub execution_providers: Vec<String>,
    pub driver_versions: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Helpers: PowerShell WMI queries (Windows) with graceful fallback
// ---------------------------------------------------------------------------

fn powershell_query(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command"])
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn wmi_processors() -> Vec<(String, String)> {
    // Returns (Name, DriverVersion-ish) pairs.
    let raw = powershell_query(&[
        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
    ]);
    match raw {
        Some(s) if !s.is_empty() => s.lines().map(|l| (l.trim().to_string(), String::new())).collect(),
        _ => Vec::new(),
    }
}

fn wmi_video_controllers() -> Vec<GpuInfo> {
    let raw = powershell_query(&[
        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterCompatibility | ConvertTo-Json",
    ]);
    let Some(text) = raw else { return Vec::new() };
    let mut out = Vec::new();
    // The output may be a single object or an array; normalise.
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    let items: Vec<serde_json::Value> = match v {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(_) => vec![v],
        _ => Vec::new(),
    };
    for item in items {
        out.push(GpuInfo {
            name: item
                .get("Name")
                .and_then(|x| x.as_str())
                .unwrap_or("Unknown GPU")
                .to_string(),
            driver_version: item
                .get("DriverVersion")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
            vendor: item
                .get("AdapterCompatibility")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    out
}

fn detect_npu() -> NpuInfo {
    // Match XDNA / NPU / AI Boost / Ryzen AI PnP devices.
    let raw = powershell_query(&[
        "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'XDNA|NPU|AI Boost|Ryzen AI|AMD NPU' -or $_.DeviceID -match 'XDNA|NPU' } | Select-Object Name,DriverVersion,DeviceID | ConvertTo-Json",
    ]);
    let mut present = false;
    let mut name = String::new();
    let mut driver = String::new();
    if let Some(text) = raw {
        if !text.is_empty() && text != "null" {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                let first = match &v {
                    serde_json::Value::Array(a) => a.first().cloned().unwrap_or_default(),
                    serde_json::Value::Object(_) => v.clone(),
                    _ => serde_json::Value::Null,
                };
                if first.is_object() {
                    present = true;
                    name = first
                        .get("Name")
                        .and_then(|x| x.as_str())
                        .unwrap_or("AMD NPU")
                        .to_string();
                    driver = first
                        .get("DriverVersion")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                }
            }
        }
    }
    NpuInfo {
        present,
        name,
        driver_version: driver,
    }
}

fn dxdiag_fallback() -> Vec<String> {
    // Best-effort: run dxdiag /t to a temp file and grep display devices.
    let tmp = std::env::temp_dir().join("npu-dxdiag.txt");
    let tmp_str = tmp.to_string_lossy().to_string();
    let ok = std::process::Command::new("dxdiag")
        .args(["/t", tmp_str.as_str()])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        return Vec::new();
    }
    // Wait briefly for dxdiag to flush.
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let text = std::fs::read_to_string(&tmp).unwrap_or_default();
    text.lines()
        .filter(|l| l.trim_start().starts_with("Card name:"))
        .map(|l| l.trim().to_string())
        .collect()
}

fn winml_info() -> WinmlInfo {
    let path = crate::winml::bridge::winml_binary();
    let available = !path.as_os_str().is_empty() && path.exists();
    let mut version = String::new();
    if available {
        if let Ok(out) = std::process::Command::new(&path).arg("--version").output() {
            let s = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            version = s.trim().lines().next().unwrap_or("").to_string();
        }
    }
    WinmlInfo {
        available,
        binary_path: path.to_string_lossy().to_string(),
        version,
    }
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

/// Collect real hardware facts: sysinfo CPU/RAM, WMI GPUs, NPU PnP match,
/// winml sidecar presence, dxdiag fallback, driver versions and EP list.
pub fn get_hardware_info() -> Result<HardwareInfo, ConverterError> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpus = sys.cpus();
    let first = cpus.first();
    let brand = first.map(|c| c.brand().to_string()).unwrap_or_else(|| {
        wmi_processors()
            .first()
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| "Unknown CPU".to_string())
    });
    let vendor = first.map(|c| c.vendor_id().to_string()).unwrap_or_default();
    let freq = first.map(|c| c.frequency()).unwrap_or(0);
    let logical = cpus.len();
    let physical = sys.physical_core_count().unwrap_or(logical);

    let mut gpus = wmi_video_controllers();
    if gpus.is_empty() {
        for card in dxdiag_fallback() {
            gpus.push(GpuInfo {
                name: card,
                driver_version: String::new(),
                vendor: String::new(),
            });
        }
    }

    let npu = detect_npu();
    let winml = winml_info();

    let os_name = System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    let os_version = System::os_version().unwrap_or_default();
    let total_gb = sys.total_memory() as f64 / 1_073_741_824.0;

    let mut drivers = std::collections::HashMap::new();
    for (i, g) in gpus.iter().enumerate() {
        drivers.insert(format!("gpu{i}"), g.driver_version.clone());
    }
    if npu.present {
        drivers.insert("npu".to_string(), npu.driver_version.clone());
    }

    let mut eps = vec!["CPU".to_string(), "DML".to_string()];
    if npu.present {
        eps.push("VitisAI".to_string());
        eps.push("QNN".to_string());
    }
    // CUDA only if an NVIDIA adapter is present.
    if gpus.iter().any(|g| {
        let n = format!("{} {}", g.name, g.vendor).to_lowercase();
        n.contains("nvidia")
    }) {
        eps.push("CUDA".to_string());
    }
    eps.sort();
    eps.dedup();

    Ok(HardwareInfo {
        cpu: CpuInfo {
            brand,
            vendor,
            physical_cores: physical,
            logical_cores: logical,
            frequency_mhz: freq,
        },
        gpus,
        npu,
        winml,
        os_name,
        os_version,
        total_memory_gb: (total_gb * 100.0).round() / 100.0,
        execution_providers: eps,
        driver_versions: drivers,
    })
}

/// Validate a version string with semver (used by commands/diagnostics).
pub fn parse_driver_version(v: &str) -> Option<semver::Version> {
    // Tolerate versions like "31.0.24002.0" by keeping first three parts.
    let clean = v.trim();
    if let Ok(parsed) = semver::Version::parse(clean) {
        return Some(parsed);
    }
    let parts: Vec<&str> = clean.split('.').collect();
    if parts.len() >= 3 {
        let tri = format!("{}.{}.{}", parts[0], parts[1], parts[2]);
        if let Ok(parsed) = semver::Version::parse(&tri) {
            return Some(parsed);
        }
    }
    None
}
