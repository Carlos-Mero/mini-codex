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
    pub(crate) last_api_input_tokens: Option<u64>,
    #[serde(default)]
    pub(crate) last_accounted_entry_count: usize,
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
        estimated_tokens: u64,
    },
    Tool {
        command: String,
        workdir: String,
        success: bool,
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionMode {
    BeforeTurn,
    MidTurn,
}

impl HistoryFile {
    pub(crate) fn push_user(&mut self, content: String) {
        self.entries.push(HistoryEntry::User {
            content,
            estimated_tokens: 0,
        });
    }

    pub(crate) fn push_assistant(&mut self, content: String) {
        self.entries.push(HistoryEntry::Assistant {
            content,
            estimated_tokens: 0,
        });
    }

    pub(crate) fn push_tool(
        &mut self,
        command: String,
        workdir: String,
        success: bool,
        content: String,
    ) {
        self.entries.push(HistoryEntry::Tool {
            command,
            workdir,
            success,
            content,
            estimated_tokens: 0,
        });
    }

    pub(crate) fn push_system(&mut self, content: String) {
        self.entries.push(HistoryEntry::System {
            content,
            estimated_tokens: 0,
        });
    }

    pub(crate) fn note_api_input_tokens(&mut self, input_tokens: Option<u64>) {
        let Some(current_input_tokens) = input_tokens else {
            self.last_accounted_entry_count = self.entries.len();
            self.last_api_input_tokens = None;
            return;
        };

        let pending_start = self.last_accounted_entry_count.min(self.entries.len());
        let pending_entries = &mut self.entries[pending_start..];
        let delta = match self.last_api_input_tokens {
            Some(previous) if current_input_tokens >= previous => current_input_tokens - previous,
            _ => current_input_tokens,
        };

        distribute_tokens(pending_entries, delta);
        self.last_accounted_entry_count = self.entries.len();
        self.last_api_input_tokens = Some(current_input_tokens);
    }

    pub(crate) fn active_token_usage(&self) -> u64 {
        let start = self.last_system_index().unwrap_or(0);
        self.entries[start..]
            .iter()
            .map(HistoryEntry::estimated_tokens)
            .sum()
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
        self.last_api_input_tokens = None;
        self.last_accounted_entry_count = self.entries.len();
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
            | Self::Assistant { content, .. } => text_weight(content),
            Self::Tool { content, .. } => text_weight(content),
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
            HistoryEntry::Assistant { content, .. } => {
                messages.push(json!({"role": "assistant", "content": content}));
            }
            HistoryEntry::Tool {
                success, content, ..
            } => {
                let content = normalize_tool_content(content);
                messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "Shell command result:\nsuccess: {success}\n{content}"
                    )
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

pub(crate) fn normalize_tool_content(content: &str) -> String {
    let mut lines = content.lines().peekable();

    for prefix in ["command: ", "workdir: ", "exit_code: "] {
        match lines.peek().copied() {
            Some(line) if line.starts_with(prefix) => {
                lines.next();
            }
            _ => return content.to_string(),
        }
    }

    while matches!(lines.peek(), Some(line) if line.trim().is_empty()) {
        lines.next();
    }

    let normalized = lines.collect::<Vec<_>>().join("\n");
    if normalized.is_empty() {
        "(no output)".to_string()
    } else {
        normalized
    }
}

fn system_prompt(workspace_root: &Path) -> String {
    format!(
        concat!(
            "You are Mini Codex, a coding assistant working inside a local workspace.\n",
            "Workspace root: {}.\n",
            "Keep responses concise and focus on completing the user's request.\n",
            "Before changing code, prefer inspecting the current state.\n",
            "Do not use destructive commands unless clearly necessary, and never access anything outside the workspace root.\n",
            "Use the shell tool whenever inspection, editing, running programs, searching, debugging, building, testing, formatting, git inspection, or other workspace tasks require command-line access.\n",
            "Treat the shell tool as powerful and potentially dangerous.\n",
            "If a shell command fails, read the result carefully and continue by trying another valid approach when possible.\n",
            "When you need shell access, reply with plain text using this exact XML-style format:\n",
            "<shell>raw shell command</shell>\n",
            "or, if needed,\n",
            "<shell workspace=\"./relative/path\">raw shell command</shell>\n",
            "The workspace attribute is optional and must stay inside the workspace root.\n",
            "You may include some short preamble before the shell tag, but do not include anything else after it.\n",
            "Only request one shell command at a time.\n",
            "If shell access is not needed, answer normally.\n"
        ),
        workspace_root.display()
    )
}
