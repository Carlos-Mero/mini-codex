use crate::ui::{COLOR_YELLOW, Spinner, print_api_error, style};
use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::{StatusCode, blocking::Response};
use serde_json::{Map, Value, json};

const API_RETRIES: usize = 7;
const API_RETRY_DELAY_MS: u64 = 1_500;

#[derive(Clone, Debug)]
pub(crate) struct LlmConfig {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) reasoning_effort: String,
    pub(crate) enable_thinking: bool,
}

#[derive(Debug)]
pub(crate) struct LlmReply {
    pub(crate) content: String,
    pub(crate) input_tokens: Option<u64>,
}

pub(crate) fn call_model(
    client: &Client,
    config: &LlmConfig,
    messages: Vec<Value>,
) -> Result<LlmReply> {
    let body = build_request_body(config, messages);
    let url = format!("{}/chat/completions", config.base_url.trim_end_matches('/'));
    let mut last_error = None;

    for attempt in 1..=API_RETRIES {
        let mut spinner = Spinner::start();
        let outcome = (|| -> Result<LlmReply> {
            let response = client
                .post(&url)
                .bearer_auth(&config.api_key)
                .json(&body)
                .send()
                .context("chat completion request failed")?;
            let response = error_for_status(response)?;
            let payload: Value = response.json().context("failed to decode model response")?;
            Ok(LlmReply {
                content: extract_response_text(&payload)?,
                input_tokens: extract_input_tokens(&payload),
            })
        })();
        spinner.stop();

        match outcome {
            Ok(text) => return Ok(text),
            Err(err) => {
                print_api_error(&format!("{err:#}"));
                last_error = Some(err);
                if attempt < API_RETRIES {
                    println!(
                        "{} request failed, retrying ({attempt}/{API_RETRIES})...",
                        style(COLOR_YELLOW, "warning>")
                    );
                    std::thread::sleep(std::time::Duration::from_millis(API_RETRY_DELAY_MS));
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("chat completion failed")))
}

fn build_request_body(config: &LlmConfig, messages: Vec<Value>) -> Value {
    let mut body = json!({
        "model": config.model,
        "messages": messages,
        "stream": false
    });

    if let Some(map) = body.as_object_mut() {
        apply_model_specific_parameters(
            map,
            &config.model,
            &config.reasoning_effort,
            config.enable_thinking,
        );
    }

    body
}

fn apply_model_specific_parameters(
    body: &mut Map<String, Value>,
    model: &str,
    reasoning_effort: &str,
    enable_thinking: bool,
) {
    let family = model_family(model);

    if supports_reasoning_effort(family) {
        body.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }

    if supports_enable_thinking(family) {
        body.insert("enable_thinking".to_string(), Value::Bool(enable_thinking));
    }
}

fn model_family(model: &str) -> &str {
    let normalized = model.trim().to_ascii_lowercase();

    if normalized.starts_with("gpt")
        || normalized.starts_with("o1")
        || normalized.starts_with("o3")
        || normalized.starts_with("o4")
    {
        "openai"
    } else if normalized.starts_with("gemini") {
        "gemini"
    } else if normalized.starts_with("qwen") {
        "qwen"
    } else if normalized.starts_with("deepseek") {
        "deepseek"
    } else if normalized.starts_with("kimi") || normalized.starts_with("moonshot") {
        "moonshot"
    } else if normalized.starts_with("doubao") {
        "doubao"
    } else if normalized.starts_with("hunyuan") {
        "hunyuan"
    } else if normalized.starts_with("glm") || normalized.starts_with("chatglm") {
        "glm"
    } else if normalized.starts_with("yi") {
        "yi"
    } else {
        "other"
    }
}

fn supports_reasoning_effort(family: &str) -> bool {
    matches!(family, "openai" | "gemini")
}

fn supports_enable_thinking(family: &str) -> bool {
    matches!(
        family,
        "qwen" | "deepseek" | "moonshot" | "doubao" | "hunyuan" | "glm" | "yi"
    )
}

fn error_for_status(response: Response) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response
        .text()
        .unwrap_or_else(|err| format!("failed to read error response body: {err}"));
    let detail = format_api_error(status, &body);
    Err(anyhow!("chat completion returned error status: {detail}"))
}

fn extract_response_text(payload: &Value) -> Result<String> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .ok_or_else(|| anyhow!("model response missing choices[0].message.content"))?;

    let text = match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => {
            let mut merged = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    merged.push_str(text);
                }
            }
            merged
        }
        _ => String::new(),
    };

    let text = text.trim().to_string();
    if text.is_empty() {
        bail!("model returned empty content");
    }
    Ok(text)
}

fn extract_input_tokens(payload: &Value) -> Option<u64> {
    payload
        .get("usage")
        .and_then(|usage| {
            usage
                .get("input_tokens")
                .or_else(|| usage.get("prompt_tokens"))
        })
        .and_then(Value::as_u64)
}

fn format_api_error(status: StatusCode, body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return format!("HTTP {status}");
    }

    if let Ok(payload) = serde_json::from_str::<Value>(trimmed) {
        if let Some(message) = payload
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
        {
            let kind = payload
                .get("error")
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("unknown_error");
            return format!("HTTP {status} ({kind}): {message}");
        }
    }

    format!("HTTP {status}: {}", truncate(trimmed, 400))
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n[... output truncated ...]");
    truncated
}
