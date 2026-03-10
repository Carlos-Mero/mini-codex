use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

const DEFAULT_TOKEN_LIMIT: u64 = 128 * 1024;
const GPT5_TOKEN_LIMIT: u64 = 1_000_000;
const COMPACTION_TRIGGER_NUMERATOR: u64 = 4;
const COMPACTION_TRIGGER_DENOMINATOR: u64 = 5;
const HISTORY_SUMMARY_PREFIX: &str = "[history summary]";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct HistoryFile {
    pub(crate) version: u32,
    pub(crate) session_id: String,
    pub(crate) workspace_root: String,
    pub(crate) last_active_at_ms: u128,
    #[serde(default)]
    pub(crate) total_input_tokens: u64,
    #[serde(default)]
    pub(crate) total_output_tokens: u64,
    #[serde(default)]
    pub(crate) total_tokens: u64,
    pub(crate) entries: Vec<HistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum HistoryEntry {
    System {
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
    User {
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
    Assistant {
        content: String,
        #[serde(default)]
        tool_calls: Vec<AssistantToolCall>,
        #[serde(default)]
        estimated_tokens: u64,
    },
    Tool {
        #[serde(default)]
        tool_call_id: String,
        #[serde(default)]
        tool_name: String,
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssistantToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionMode {
    BeforeTurn,
    MidTurn,
}

impl HistoryFile {
    pub(crate) fn push_user(&mut self, content: String) {
        let mut entry = HistoryEntry::User {
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_assistant(&mut self, content: String, tool_calls: Vec<AssistantToolCall>) {
        let mut entry = HistoryEntry::Assistant {
            content,
            tool_calls,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_tool(&mut self, tool_call_id: String, tool_name: String, content: String) {
        let mut entry = HistoryEntry::Tool {
            tool_call_id,
            tool_name,
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_system(&mut self, content: String) {
        let mut entry = HistoryEntry::System {
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn note_api_usage(
        &mut self,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        total_tokens: Option<u64>,
    ) {
        let start = self.last_system_index().unwrap_or(0);
        let entry_count = self.entries[start..].len() as u64;
        let entry_estimate = estimate_entry_tokens(&self.entries, start);
        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(input_tokens.unwrap_or(0));
        self.total_output_tokens = self
            .total_output_tokens
            .saturating_add(output_tokens.unwrap_or(0));
        let fallback_total = input_tokens
            .unwrap_or(0)
            .saturating_add(output_tokens.unwrap_or(0));
        self.total_tokens = self
            .total_tokens
            .saturating_add(total_tokens.unwrap_or(fallback_total));

        let active_estimate = input_tokens
            .or(total_tokens)
            .unwrap_or_else(|| entry_estimate.max(entry_count));
        apply_estimated_tokens(&mut self.entries, start, active_estimate);
    }

    pub(crate) fn active_token_usage(&self) -> u64 {
        let start = self.last_system_index().unwrap_or(0);
        self.entries[start..]
            .iter()
            .map(HistoryEntry::estimated_tokens)
            .sum()
    }

    pub(crate) fn total_token_usage(&self) -> u64 {
        self.total_tokens
    }

    pub(crate) fn needs_compaction(&self, token_limit: u64) -> bool {
        self.active_token_usage()
            >= token_limit.saturating_mul(COMPACTION_TRIGGER_NUMERATOR)
                / COMPACTION_TRIGGER_DENOMINATOR
    }

    pub(crate) fn last_user_content(&self) -> Option<String> {
        self.entries.iter().rev().find_map(|entry| match entry {
            HistoryEntry::User { content, .. } => Some(content.clone()),
            _ => None,
        })
    }

    pub(crate) fn compaction_prompt(&self, mode: CompactionMode) -> String {
        match mode {
            CompactionMode::BeforeTurn => concat!(
                "The conversation is close to the token limit. Briefly summarize the previous ",
                "history that will be useful for future work. Focus on durable context, key ",
                "decisions, important files, constraints, and unresolved issues."
            )
            .to_string(),
            CompactionMode::MidTurn => concat!(
                "The conversation is close to the token limit and the current task is not ",
                "finished yet. Briefly summarize the previous history that will be useful for ",
                "continuing the work. Include the current task, what has already been done, ",
                "important files and constraints, recent tool results, and the most useful next ",
                "steps."
            )
            .to_string(),
        }
    }

    pub(crate) fn apply_compaction(&mut self, summary: String, resume_user: Option<String>) {
        self.push_system(format!("{HISTORY_SUMMARY_PREFIX}\n{}", summary.trim()));
        if let Some(user) = resume_user {
            self.push_user(user);
        }
        let start = self.last_system_index().unwrap_or(0);
        let estimated = estimate_entry_tokens(&self.entries, start);
        apply_estimated_tokens(&mut self.entries, start, estimated.max(1));
    }

    fn last_system_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .rposition(|entry| matches!(entry, HistoryEntry::System { .. }))
    }
}

impl HistoryEntry {
    pub(crate) fn estimated_tokens(&self) -> u64 {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens,
        }
    }

    fn weight(&self) -> u64 {
        match self {
            Self::System { content, .. }
            | Self::User { content, .. }
            | Self::Assistant { content, .. } => {
                let tool_call_weight = match self {
                    Self::Assistant { tool_calls, .. } => tool_calls
                        .iter()
                        .map(|call| text_weight(&call.name) + text_weight(&call.arguments))
                        .sum(),
                    _ => 0,
                };
                text_weight(content) + tool_call_weight
            }
            Self::Tool { content, .. } => text_weight(content),
        }
    }

    fn set_estimated_tokens(&mut self, value: u64) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = value,
        }
    }

    fn reset_estimated_tokens(&mut self) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = 0,
        }
    }

    fn add_estimated_tokens(&mut self, delta: u64) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = estimated_tokens.saturating_add(delta),
        }
    }
}

pub(crate) fn build_messages(workspace_root: &Path, entries: &[HistoryEntry]) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": system_prompt(workspace_root)
    })];

    let active_entries = match entries
        .iter()
        .rposition(|entry| matches!(entry, HistoryEntry::System { .. }))
    {
        Some(index) => &entries[index..],
        None => entries,
    };

    for entry in active_entries {
        match entry {
            HistoryEntry::System { content, .. } => {
                messages.push(json!({"role": "system", "content": content}));
            }
            HistoryEntry::User { content, .. } => {
                messages.push(json!({"role": "user", "content": content}));
            }
            HistoryEntry::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let assistant_content = if tool_calls.is_empty() || !content.trim().is_empty() {
                    Value::String(content.clone())
                } else {
                    Value::Null
                };
                let mut message = json!({"role": "assistant", "content": assistant_content});
                if !tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(
                        tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                messages.push(message);
            }
            HistoryEntry::Tool {
                tool_call_id,
                tool_name,
                content,
                ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content
                }));
            }
        }
    }

    messages
}

pub(crate) fn token_limit_for_model(model: &str) -> u64 {
    let normalized = model.trim().to_ascii_lowercase();
    if normalized.starts_with("gpt-5") {
        GPT5_TOKEN_LIMIT
    } else if normalized.starts_with("deepseek-v3.2") {
        DEFAULT_TOKEN_LIMIT
    } else {
        DEFAULT_TOKEN_LIMIT
    }
}

fn apply_estimated_tokens(entries: &mut [HistoryEntry], start: usize, target_total: u64) {
    if start >= entries.len() {
        return;
    }

    for entry in entries.iter_mut() {
        entry.reset_estimated_tokens();
    }

    distribute_tokens(&mut entries[start..], target_total);
}

fn estimate_entry_tokens(entries: &[HistoryEntry], start: usize) -> u64 {
    if start >= entries.len() {
        return 0;
    }

    let weight = entries[start..]
        .iter()
        .map(HistoryEntry::weight)
        .sum::<u64>();
    weight.saturating_mul(4)
}

fn distribute_tokens(entries: &mut [HistoryEntry], delta: u64) {
    if entries.is_empty() || delta == 0 {
        return;
    }

    let total_weight = entries.iter().map(HistoryEntry::weight).sum::<u64>();
    if total_weight == 0 {
        let base = delta / entries.len() as u64;
        let mut remainder = delta % entries.len() as u64;
        for entry in entries {
            let extra = u64::from(remainder > 0);
            entry.add_estimated_tokens(base + extra);
            remainder = remainder.saturating_sub(1);
        }
        return;
    }

    let mut assigned = 0_u64;
    let last_index = entries.len() - 1;
    for (index, entry) in entries.iter_mut().enumerate() {
        let share = if index == last_index {
            delta.saturating_sub(assigned)
        } else {
            delta.saturating_mul(entry.weight()) / total_weight
        };
        assigned = assigned.saturating_add(share);
        entry.add_estimated_tokens(share);
    }
}

fn text_weight(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

fn system_prompt(workspace_root: &Path) -> String {
    format!(
        concat!(
            "You are Mini Codex, a coding assistant working inside a local workspace.\n",
            "Workspace root: {}.\n",
            "Keep responses concise and focus on completing the user's request.\n",
            "Before changing code or answering questions about the local projects, prefer inspecting the current state.\n",
            "Do not use destructive commands unless clearly necessary, and never access anything outside the workspace root.\n",
            "Use the available shell_tool whenever inspection, editing, running programs, searching, debugging, building, testing, formatting, git inspection, or other workspace tasks require command-line access.\n",
            "Treat the shell_tool as powerful and potentially dangerous.\n",
            "If a shell command fails, read the result carefully and continue by trying another valid approach when possible.\n",
            "When calling shell_tool, the workdir argument is optional and must stay inside the workspace root.\n",
            "Prefer one shell_tool call at a time unless batching multiple independent read-only commands is clearly useful.\n",
            "Before calling a tool, usually include a brief preamble for the user that says what you are about to do and why.
",
            "Keep tool preambles short: typically one sentence and under 20 words.
",
            "If shell access is not needed, answer normally.\n"
        ),
        workspace_root.display()
    )
}
