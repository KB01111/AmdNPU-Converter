use reqwest::header;
use serde::{Deserialize, Serialize};

use crate::error::ConverterError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    pub rfilename: String,
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnnxCompat {
    pub export_strategy: String,
    pub opset: u32,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    pub model_id: String,
    pub arch: String,
    pub params: String,
    pub format: String,
    pub license: String,
    pub input_types: Vec<String>,
    pub precision: String,
    pub estimated_bytes: u64,
    pub files: Vec<ModelFile>,
    pub compat: OnnxCompat,
}

// Partial view of the HF model API response.
#[derive(Debug, Deserialize)]
struct HfApiResponse {
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
    #[serde(default)]
    safetensors: Option<HfSafetensors>,
}

#[derive(Debug, Deserialize, Clone)]
struct HfSibling {
    #[serde(default)]
    rfilename: String,
}

#[derive(Debug, Deserialize, Clone)]
struct HfSafetensors {
    #[serde(default)]
    total: Option<u64>,
}

// ---------------------------------------------------------------------------
// URL parsing (no regex dependency — manual parse, same result)
// ---------------------------------------------------------------------------

/// Accepts full HF URLs (`https://huggingface.co/org/model[/...]`)
/// or bare ids (`org/model`). Returns the canonical `owner/repo` id.
pub fn parse_hf_id(input: &str) -> Result<String, ConverterError> {
    let t = input.trim().trim_matches('/');
    if t.is_empty() {
        return Err(ConverterError::InvalidModel(input.to_string()));
    }
    let path = if let Some(rest) = t
        .strip_prefix("https://huggingface.co/")
        .or_else(|| t.strip_prefix("http://huggingface.co/"))
        .or_else(|| t.strip_prefix("huggingface.co/"))
    {
        rest
    } else if t.contains("://") {
        return Err(ConverterError::InvalidModel(input.to_string()));
    } else {
        t
    };
    // First two segments are owner/repo; ignore tree/blob/resolve suffixes.
    let mut segs: Vec<&str> = path.split('/').collect();
    if segs.len() < 2 {
        return Err(ConverterError::InvalidModel(input.to_string()));
    }
    if segs.iter().any(|s| s.is_empty() || *s == "." || *s == "..")
        || (segs.len() > 2 && !matches!(segs[2], "tree" | "blob" | "resolve"))
    {
        return Err(ConverterError::InvalidModel(input.to_string()));
    }
    // Handle URLs like org/model/tree/main, org/model/blob/..., org/model/resolve/...
    if segs.len() > 2 {
        segs.truncate(2);
    }
    let owner = segs[0];
    let repo = segs[1];
    for s in [owner, repo] {
        if s.is_empty()
            || s.len() > 96
            || !s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(ConverterError::InvalidModel(input.to_string()));
        }
    }
    Ok(format!("{owner}/{repo}"))
}

// ---------------------------------------------------------------------------
// Inspection — real HuggingFace API calls
// ---------------------------------------------------------------------------

/// Inspect a HuggingFace model: resolves the id, calls the public HF API,
/// enumerates siblings and HEAD-probes the largest safetensors file for size.
pub async fn hf_inspect(url: &str) -> Result<ModelMetadata, ConverterError> {
    let model_id = parse_hf_id(url)?;
    let api_url = format!("https://huggingface.co/api/models/{model_id}");

    let client = reqwest::Client::builder()
        .user_agent("npu-model-converter/0.1.0")
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| ConverterError::Network(e.to_string()))?;

    let resp = client
        .get(&api_url)
        .header(header::ACCEPT, "application/json")
        .send()
        .await
        .map_err(|e| ConverterError::Network(e.to_string()))?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ConverterError::NotFound(model_id));
    }
    let resp = resp
        .error_for_status()
        .map_err(|e| ConverterError::Network(e.to_string()))?;
    let info: HfApiResponse = resp
        .json()
        .await
        .map_err(|e| ConverterError::Network(e.to_string()))?;

    // Architecture guess from tags / pipeline tag / library.
    let arch = info
        .pipeline_tag
        .clone()
        .or_else(|| {
            info.tags
                .iter()
                .find(|t| t.contains('-') || t.len() > 2)
                .cloned()
        })
        .unwrap_or_else(|| "transformer".to_string());

    let params = info
        .safetensors
        .as_ref()
        .and_then(|s| s.total)
        .map(|t| t.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let format = if info
        .siblings
        .iter()
        .any(|s| s.rfilename.ends_with(".safetensors"))
    {
        "safetensors"
    } else if info.siblings.iter().any(|s| s.rfilename.ends_with(".bin")) {
        "pytorch-bin"
    } else if info.siblings.iter().any(|s| s.rfilename.ends_with(".onnx")) {
        "onnx"
    } else {
        "unknown"
    }
    .to_string();

    let license = info
        .license
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    let input_types = info
        .tags
        .iter()
        .filter(|t| ["text", "image", "audio", "video", "multimodal"].contains(&t.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    // HEAD the largest safetensors shard to get a real byte size.
    let mut files: Vec<ModelFile> = Vec::new();
    let mut estimated_bytes: u64 = 0;

    let mut shard_candidates: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    shard_candidates.sort();
    // Probe at most the first + last shard to bound latency.
    let mut probe: Vec<String> = Vec::new();
    if let Some(first) = shard_candidates.first().cloned() {
        probe.push(first);
    }
    if shard_candidates.len() > 1 {
        if let Some(last) = shard_candidates.last().cloned() {
            probe.push(last);
        }
    }
    let mut probed_total: u64 = 0;
    let mut probed_count = 0;
    for name in &probe {
        let file_url = format!("https://huggingface.co/{model_id}/resolve/main/{name}");
        if let Ok(head) = client.head(&file_url).send().await {
            if let Some(len) = head.headers().get(header::CONTENT_LENGTH) {
                if let Ok(s) = len.to_str() {
                    if let Ok(n) = s.parse::<u64>() {
                        probed_total += n;
                        probed_count += 1;
                    }
                }
            }
        }
    }
    if probed_count > 0 && probed_count == shard_candidates.len() {
        estimated_bytes = probed_total;
    }

    for s in &info.siblings {
        files.push(ModelFile {
            rfilename: s.rfilename.clone(),
            size: None,
        });
    }

    let compat = OnnxCompat {
        export_strategy: export_strategy(&model_id, &format).into(),
        opset: 17,
        notes: format!(
            "library={} arch={}",
            info.library_name.clone().unwrap_or_default(),
            arch
        ),
    };

    Ok(ModelMetadata {
        model_id,
        arch,
        params,
        format,
        license,
        input_types,
        precision: "fp32".to_string(),
        estimated_bytes,
        files,
        compat,
    })
}

fn export_strategy(model_id: &str, format: &str) -> &'static str {
    if model_id.starts_with("cua-ai/cua-s1") {
        "cua-s1"
    } else if format == "onnx" {
        "direct-onnx"
    } else {
        "transformers"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hf_ids_reject_foreign_hosts_and_traversal() {
        for input in [
            "http://evil.test/org/model",
            "https://evil.test/org/model",
            "../model",
            "org/..",
            "org/model/../../evil",
            "https://huggingface.co.evil.test/org/model",
        ] {
            assert!(parse_hf_id(input).is_err(), "accepted {input}");
        }
        assert_eq!(
            parse_hf_id("https://huggingface.co/org/model/tree/main").unwrap(),
            "org/model"
        );
    }
    #[test]
    fn strategies_match_worker_routes() {
        assert_eq!(
            export_strategy("cua-ai/cua-s1-forms", "safetensors"),
            "cua-s1"
        );
        assert_eq!(export_strategy("org/model", "safetensors"), "transformers");
        assert_eq!(export_strategy("org/model", "onnx"), "direct-onnx");
    }
}
