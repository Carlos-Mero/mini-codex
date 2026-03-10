mod history;
mod llm;
mod ui;

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Local, TimeZone};
use history::{
    AssistantToolCall, CompactionMode, HistoryEntry, HistoryFile, build_messages,
    token_limit_for_model,
};
use llm::{LlmConfig, ToolCall, call_model};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use ui::{
    COLOR_BLUE, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_GREEN, COLOR_RED, COLOR_YELLOW,
    editor_prompt, print_tool_call, print_tool_result, prompt_for_approval, role_prefix, style,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-5.4";
const DEFAULT_REASONING_EFFORT: &str = "medium";
const API_TIMEOUT_SECS: u64 = 20 * 60;
const LOOP_LIMIT: usize = 64;
#[derive(Clone, Debug)]
struct Config {
    llm: LlmConfig,
    history_token_limit: u64,
    workspace_root: PathBuf,
}

#[derive(Debug)]
struct App {
    client: Client,
    config: Config,
    history_path: PathBuf,
    history: HistoryFile,
    auto_approve: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResumeMode {
    New,
    Select,
    Last,
}

#[derive(Debug)]
struct ShellRequest {
    command: String,
    workdir: Option<String>,
}

#[derive(Debug)]
struct CommandOutcome {
    success: bool,
    tool_content: String,
}

#[derive(Debug)]
struct SessionSummary {
    path: PathBuf,
    history: HistoryFile,
}

#[derive(Debug, Deserialize)]
struct ShellToolArgs {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
}

fn main() -> Result<()> {
    App::load()?.run()
}

impl App {
    fn load() -> Result<Self> {
        let workspace_root = env::current_dir().context("failed to determine current directory")?;
        let env_path = workspace_root.join(".env");
        if env_path.exists() {
            let _ = dotenvy::from_path(&env_path);
        }

        let mut args = env::args().skip(1).peekable();
        let mut model = DEFAULT_MODEL.to_string();
        let mut reasoning_effort = DEFAULT_REASONING_EFFORT.to_string();
        let mut enable_thinking = true;
        let mut auto_approve = false;
        let mut resume = ResumeMode::New;
        let mut history_token_limit = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--auto" => auto_approve = true,
                "--model" => {
                    model = args
                        .next()
                        .ok_or_else(|| anyhow!("--model requires a value"))?;
                }
                "--reasoning-effort" => {
                    reasoning_effort = args
                        .next()
                        .ok_or_else(|| anyhow!("--reasoning-effort requires a value"))?;
                }
                "--enable-thinking" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("--enable-thinking requires a value"))?;
                    enable_thinking = parse_bool_flag("--enable-thinking", &value)?;
                }
                "--token-limit" => {
                    let value = args
                        .next()
                        .ok_or_else(|| anyhow!("--token-limit requires a value"))?;
                    history_token_limit = Some(
                        value
                            .parse::<u64>()
                            .with_context(|| format!("invalid --token-limit value: {value}"))?,
                    );
                }
                "resume" => resume = ResumeMode::Select,
                "--last" => resume = ResumeMode::Last,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }

        let api_key = read_env(&["MINI_CODEX_API_KEY", "OPENAI_API_KEY"])
            .ok_or_else(|| anyhow!("missing MINI_CODEX_API_KEY or OPENAI_API_KEY"))?;
        let base_url = read_env(&["MINI_CODEX_BASE_URL", "OPENAI_BASE_URL"])
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let resolved_history_token_limit =
            history_token_limit.unwrap_or_else(|| token_limit_for_model(&model));
        let config = Config {
            llm: LlmConfig {
                api_key,
                base_url,
                model,
                reasoning_effort,
                enable_thinking,
            },
            history_token_limit: resolved_history_token_limit,
            workspace_root: workspace_root.clone(),
        };
        let (history_path, history) = load_or_create_session(&workspace_root, resume)?;

        Ok(Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(API_TIMEOUT_SECS))
                .build()
                .context("failed to build http client")?,
            config,
            history_path,
            history,
            auto_approve,
        })
    }

    fn run(&mut self) -> Result<()> {
        println!("{}", style(COLOR_BOLD, "mini-codex"));
        println!(
            "{} {}",
            style(COLOR_DIM, "workspace:"),
            self.config.workspace_root.display()
        );
        println!("{} {}", style(COLOR_DIM, "model:"), self.config.llm.model);
        println!(
            "{} {}",
            style(COLOR_DIM, "reasoning effort:"),
            self.config.llm.reasoning_effort
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "enable thinking:"),
            self.config.llm.enable_thinking
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "history token limit:"),
            self.config.history_token_limit
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "session:"),
            self.history.session_id
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "history:"),
            self.history_path.display()
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "approval mode:"),
            if self.auto_approve {
                style(COLOR_YELLOW, "auto")
            } else {
                style(COLOR_CYAN, "ask")
            }
        );
        println!("{} /help", style(COLOR_DIM, "type"));
        self.print_history();

        let mut editor =
            rustyline::DefaultEditor::new().context("failed to initialize line editor")?;

        loop {
            let line = match editor.readline(&editor_prompt("you")) {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    println!();
                    continue;
                }
                Err(rustyline::error::ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(err) => return Err(err).context("failed to read user input"),
            };

            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let _ = editor.add_history_entry(line.as_str());

            match line.as_str() {
                "/exit" | "/quit" => break,
                "/help" => {
                    println!("{}", style(COLOR_DIM, "/help"));
                    println!("{}", style(COLOR_DIM, "/exit"));
                    println!("{}", style(COLOR_DIM, "/auto on"));
                    println!("{}", style(COLOR_DIM, "/auto off"));
                    continue;
                }
                "/auto on" => {
                    self.auto_approve = true;
                    println!("{}", style(COLOR_YELLOW, "auto approval enabled"));
                    continue;
                }
                "/auto off" => {
                    self.auto_approve = false;
                    println!("{}", style(COLOR_YELLOW, "auto approval disabled"));
                    continue;
                }
                _ => {}
            }

            if let Err(err) = self.run_turn(line) {
                eprintln!("{} {err:#}", style(COLOR_RED, "error>"));
            }
        }

        Ok(())
    }

    fn run_turn(&mut self, user_input: String) -> Result<()> {
        self.compact_history_if_needed(CompactionMode::BeforeTurn)?;
        self.history.push_user(user_input);
        self.save_history()?;

        for _ in 0..LOOP_LIMIT {
            self.compact_history_if_needed(CompactionMode::MidTurn)?;
            let messages = build_messages(&self.config.workspace_root, &self.history.entries);
            let reply = call_model(&self.client, &self.config.llm, messages, true)?;
            self.history.note_api_input_tokens(reply.input_tokens);
            let assistant_tool_calls = reply
                .tool_calls
                .iter()
                .map(|call| AssistantToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            self.history
                .push_assistant(reply.content.clone(), assistant_tool_calls);
            self.save_history()?;

            if !reply.content.trim().is_empty() {
                println!(
                    "{}{}",
                    role_prefix("assistant", COLOR_GREEN),
                    reply.content.trim()
                );
            }

            if !reply.tool_calls.is_empty() {
                self.handle_tool_calls(reply.tool_calls)?;
                self.save_history()?;
                continue;
            }

            return Ok(());
        }

        Err(anyhow!(
            "agent loop exceeded {LOOP_LIMIT} steps without producing a final response"
        ))
    }

    fn compact_history_if_needed(&mut self, mode: CompactionMode) -> Result<()> {
        if !self
            .history
            .needs_compaction(self.config.history_token_limit)
        {
            return Ok(());
        }

        let resume_user = if mode == CompactionMode::MidTurn {
            self.history.last_user_content()
        } else {
            None
        };
        self.history.push_user(self.history.compaction_prompt(mode));
        let messages = build_messages(&self.config.workspace_root, &self.history.entries);
        let reply = call_model(&self.client, &self.config.llm, messages, false)?;
        self.history.note_api_input_tokens(reply.input_tokens);
        self.history.apply_compaction(reply.content, resume_user);
        self.save_history()
    }

    fn handle_tool_calls(&mut self, tool_calls: Vec<ToolCall>) -> Result<()> {
        for tool_call in tool_calls {
            match tool_call.name.as_str() {
                "shell_tool" => {
                    let request = parse_shell_tool_args(&tool_call)?;
                    let outcome = self.run_shell(request)?;
                    print_tool_result(
                        &normalize_tool_content_for_display(&outcome.tool_content),
                        outcome.success,
                    );
                    self.history
                        .push_tool(tool_call.id, tool_call.name, outcome.tool_content);
                }
                other => {
                    let content = format!("unsupported tool: {other}");
                    print_tool_result(&content, false);
                    self.history
                        .push_tool(tool_call.id, tool_call.name, content);
                }
            }
        }
        Ok(())
    }

    fn run_shell(&mut self, request: ShellRequest) -> Result<CommandOutcome> {
        let workdir = resolve_workdir(&self.config.workspace_root, request.workdir.as_deref())?;

        let workdir_text = workdir.display().to_string();
        print_tool_call(&request.command, &workdir_text);
        if !self.auto_approve && !prompt_for_approval(&mut self.auto_approve)? {
            return Ok(CommandOutcome {
                success: false,
                tool_content: format_tool_content(
                    &request.command,
                    &workdir.display().to_string(),
                    false,
                    "command rejected by user".to_string(),
                ),
            });
        }

        let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let output = Command::new(&shell)
            .arg("-lc")
            .arg(&request.command)
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to run shell command via {shell}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut content = String::new();
        if !stdout.trim().is_empty() {
            content.push_str("stdout:\n");
            content.push_str(stdout.as_ref());
        }
        if !stderr.trim().is_empty() {
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str("stderr:\n");
            content.push_str(stderr.as_ref());
        }
        if content.trim().is_empty() {
            content.push_str("(no output)");
        }

        Ok(CommandOutcome {
            success: output.status.success(),
            tool_content: format_tool_content(
                &request.command,
                &workdir_text,
                output.status.success(),
                content,
            ),
        })
    }

    fn print_history(&self) {
        if self.history.entries.is_empty() {
            return;
        }

        println!("{}", style(COLOR_BOLD, "resumed history"));
        for entry in &self.history.entries {
            match entry {
                HistoryEntry::System { content, .. } => {
                    println!("{}{}", role_prefix("system", COLOR_YELLOW), content);
                }
                HistoryEntry::User { content, .. } => {
                    println!("{}{}", role_prefix("you", COLOR_BLUE), content);
                }
                HistoryEntry::Assistant { content, .. } => {
                    if !content.trim().is_empty() {
                        println!("{}{}", role_prefix("assistant", COLOR_GREEN), content);
                    }
                }
                HistoryEntry::Tool { content, .. } => {
                    print_tool_result(
                        &normalize_tool_content_for_display(content),
                        tool_content_success(content),
                    );
                }
            }
        }
    }

    fn save_history(&mut self) -> Result<()> {
        self.history.last_active_at_ms = now_millis();
        let text =
            serde_json::to_string_pretty(&self.history).context("failed to encode history")?;
        fs::write(&self.history_path, text).with_context(|| {
            format!(
                "failed to write history file {}",
                self.history_path.display()
            )
        })
    }
}

fn parse_shell_tool_args(tool_call: &ToolCall) -> Result<ShellRequest> {
    let args: ShellToolArgs = serde_json::from_str(&tool_call.arguments).with_context(|| {
        format!(
            "failed to decode arguments for tool {}: {}",
            tool_call.name, tool_call.arguments
        )
    })?;
    let command = args.command.trim().to_string();
    if command.is_empty() {
        bail!("shell_tool requires a non-empty command");
    }
    Ok(ShellRequest {
        command,
        workdir: args
            .workdir
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    })
}

fn format_tool_content(command: &str, workdir: &str, success: bool, output: String) -> String {
    format!("command: {command}\nworkdir: {workdir}\nsuccess: {success}\n\n{output}")
}

fn normalize_tool_content_for_display(content: &str) -> String {
    let mut lines = content.lines().peekable();

    for prefix in ["command: ", "workdir: ", "success: "] {
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

fn tool_content_success(content: &str) -> bool {
    content
        .lines()
        .nth(2)
        .and_then(|line| line.strip_prefix("success: "))
        .map(|value| value.trim() == "true")
        .unwrap_or(false)
}

fn resolve_workdir(workspace_root: &Path, requested: Option<&str>) -> Result<PathBuf> {
    let root = workspace_root.canonicalize().with_context(|| {
        format!(
            "workspace root does not exist: {}",
            workspace_root.display()
        )
    })?;
    let candidate = match requested {
        None | Some("") | Some(".") => root.clone(),
        Some(path) => root.join(path),
    };
    let candidate = candidate
        .canonicalize()
        .with_context(|| format!("workdir does not exist: {}", candidate.display()))?;

    if !candidate.starts_with(&root) {
        bail!("workdir escapes workspace: {}", candidate.display());
    }

    Ok(candidate)
}

fn load_or_create_session(
    workspace_root: &Path,
    resume: ResumeMode,
) -> Result<(PathBuf, HistoryFile)> {
    match resume {
        ResumeMode::New => {
            let history = HistoryFile {
                version: 3,
                session_id: format!("session-{}-{}", now_millis(), std::process::id()),
                workspace_root: workspace_root.display().to_string(),
                last_active_at_ms: now_millis(),
                last_api_input_tokens: None,
                last_accounted_entry_count: 0,
                entries: Vec::new(),
            };
            Ok((
                sessions_root()?.join(format!("{}.json", history.session_id)),
                history,
            ))
        }
        ResumeMode::Last => {
            let mut sessions = list_sessions(workspace_root)?;
            let session = sessions
                .drain(..)
                .next()
                .ok_or_else(|| anyhow!("no previous sessions found for this workspace"))?;
            Ok((session.path, session.history))
        }
        ResumeMode::Select => {
            let sessions = list_sessions(workspace_root)?;
            if sessions.is_empty() {
                bail!("no previous sessions found for this workspace");
            }

            println!("available sessions:");
            for (index, session) in sessions.iter().enumerate() {
                println!("  {}. {}", index + 1, session.history.session_id,);
                println!(
                    "     last active: {}",
                    format_last_active(session.history.last_active_at_ms)
                );
                println!(
                    "     last prompt: {}",
                    format_last_user_prompt(&session.history)
                );
                println!("     path: {}", session.path.display());
            }

            let selected = loop {
                print!("resume which session? [1-{}]> ", sessions.len());
                io::stdout().flush().ok();
                let mut line = String::new();
                io::stdin()
                    .read_line(&mut line)
                    .context("failed to read session selection")?;
                let trimmed = line.trim();
                let index = trimmed
                    .parse::<usize>()
                    .with_context(|| format!("invalid session selection: {trimmed}"))?;
                if (1..=sessions.len()).contains(&index) {
                    break index - 1;
                }
                println!("please enter a number between 1 and {}", sessions.len());
            };

            let session = sessions
                .into_iter()
                .nth(selected)
                .ok_or_else(|| anyhow!("invalid session selection"))?;
            Ok((session.path, session.history))
        }
    }
}

fn list_sessions(workspace_root: &Path) -> Result<Vec<SessionSummary>> {
    let root = sessions_root()?;
    if !root.exists() {
        return Ok(Vec::new());
    }

    let workspace_key = workspace_root.display().to_string();
    let mut sessions = Vec::new();
    for entry in
        fs::read_dir(&root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let history: HistoryFile = match serde_json::from_str(&text) {
            Ok(history) => history,
            Err(_) => continue,
        };
        if history.workspace_root == workspace_key {
            sessions.push(SessionSummary { path, history });
        }
    }

    sessions.sort_by(|left, right| {
        right
            .history
            .last_active_at_ms
            .cmp(&left.history.last_active_at_ms)
    });
    Ok(sessions)
}

fn sessions_root() -> Result<PathBuf> {
    let root = env::temp_dir().join("mini-codex-sessions");
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    Ok(root)
}

fn format_last_active(timestamp_ms: u128) -> String {
    let timestamp_ms = match i64::try_from(timestamp_ms) {
        Ok(value) => value,
        Err(_) => return timestamp_ms.to_string(),
    };
    match Local.timestamp_millis_opt(timestamp_ms).single() {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        None => timestamp_ms.to_string(),
    }
}

fn format_last_user_prompt(history: &HistoryFile) -> String {
    history
        .entries
        .iter()
        .rev()
        .find_map(|entry| match entry {
            HistoryEntry::User { content, .. } => Some(preview_text(content, 100)),
            _ => None,
        })
        .unwrap_or_else(|| "(no user prompt yet)".to_string())
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        preview.push('…');
    }
    if preview.is_empty() {
        "(empty)".to_string()
    } else {
        preview
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn read_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn parse_bool_flag(flag: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => bail!("{flag} expects true or false"),
    }
}

fn print_help() {
    println!("{}", style(COLOR_BOLD, "mini-codex"));
    println!();
    println!("{}", style(COLOR_DIM, "usage:"));
    println!(
        "  mini-codex [--model MODEL] [--reasoning-effort LEVEL] [--enable-thinking BOOL] [--token-limit TOKENS] [--auto]"
    );
    println!(
        "  mini-codex resume [--last] [--model MODEL] [--reasoning-effort LEVEL] [--enable-thinking BOOL] [--token-limit TOKENS] [--auto]"
    );
}
