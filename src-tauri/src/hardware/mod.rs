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
    #[serde(default)]
    pub provider_errors: Vec<String>,
    #[serde(default)]
    pub provider_devices: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderDiscovery {
    pub providers: Vec<String>,
    pub devices: Vec<serde_json::Value>,
    pub errors: Vec<String>,
}
pub fn parse_discovery(stdout: &str) -> Result<ProviderDiscovery, ConverterError> {
    serde_json::from_str(stdout)
        .or_else(|original| {
            stdout
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str(line).ok())
                .ok_or(original)
        })
        .map_err(ConverterError::from)
}

// ---------------------------------------------------------------------------
// Helpers: PowerShell WMI queries (Windows) with graceful fallback
// ---------------------------------------------------------------------------

fn powershell_query(args: &[&str]) -> Option<String> {
    let out = background_command("powershell")
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
    let raw =
        powershell_query(&["Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"]);
    match raw {
        Some(s) if !s.is_empty() => s
            .lines()
            .map(|l| (l.trim().to_string(), String::new()))
            .collect(),
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

fn is_npu_name(name: &str) -> bool {
    let name = name.to_lowercase();
    name.contains("xdna")
        || name.contains("ai boost")
        || name.contains("amd ipu")
        || name
            .split(|c: char| !c.is_ascii_alphanumeric())
            .any(|word| word == "npu")
}

fn detect_npu() -> NpuInfo {
    // Signed driver records provide both the accelerator name and actual driver version.
    let raw = powershell_query(&[
        "Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match 'XDNA|\\bNPU\\b|AI Boost|AMD IPU' } | Select-Object @{Name='Name';Expression={$_.DeviceName}},DriverVersion,DeviceID | ConvertTo-Json",
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
                if first
                    .get("Name")
                    .and_then(|v| v.as_str())
                    .is_some_and(is_npu_name)
                {
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
    let ok = background_command("dxdiag")
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
        if let Ok(out) = background_command(&path).arg("--version").output() {
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
        execution_providers: vec![],
        driver_versions: drivers,
        provider_errors: vec![],
        provider_devices: vec![],
    })
}

fn background_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
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

#[cfg(test)]
mod tests {
    #[test]
    fn dynamic_discovery_preserves_real_devices_and_errors() {
        let result=super::parse_discovery(r#"{"providers":["CPUExecutionProvider","VitisAIExecutionProvider"],"devices":[{"provider":"VitisAIExecutionProvider","device":"npu","name":"AMD XDNA"}],"errors":["Other EP unavailable"],"windowsml_available":true}"#).unwrap();
        assert_eq!(result.providers.len(), 2);
        assert_eq!(result.devices[0]["device"], "npu");
        assert_eq!(result.errors, vec!["Other EP unavailable"]);
        assert!(super::parse_discovery("{}").is_err());
    }
    #[test]
    fn cpu_brand_and_input_device_are_not_npus() {
        for name in [
            "Microsoft Input Configuration Device",
            "AMD Ryzen AI 9 HX 370",
            "AMD Ryzen AI CPU",
        ] {
            assert!(!super::is_npu_name(name), "false accelerator: {name}");
        }
        assert!(super::is_npu_name("AMD IPU Device"));
        assert!(super::is_npu_name("AMD NPU"));
        assert!(super::is_npu_name("Intel(R) AI Boost"));
    }
}
