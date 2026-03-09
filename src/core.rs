mod llm;
mod ui;

use anyhow::{Context, Result, anyhow, bail};
use llm::{LlmConfig, call_model};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
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

#[derive(Debug, Serialize, Deserialize)]
struct HistoryFile {
    version: u32,
    session_id: String,
    workspace_root: String,
    last_active_at_ms: u128,
    entries: Vec<HistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum HistoryEntry {
    User {
        content: String,
    },
    Assistant {
        content: String,
    },
    Tool {
        command: String,
        workdir: String,
        success: bool,
        content: String,
    },
}

#[derive(Debug)]
struct ShellRequest {
    command: String,
    workdir: Option<String>,
}

#[derive(Debug)]
struct ParsedShellReply {
    preamble: Option<String>,
    request: ShellRequest,
}

#[derive(Debug)]
struct CommandOutcome {
    command: String,
    workdir: String,
    success: bool,
    content: String,
}

#[derive(Debug)]
struct SessionSummary {
    path: PathBuf,
    history: HistoryFile,
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
        let config = Config {
            llm: LlmConfig {
                api_key,
                base_url,
                model,
                reasoning_effort,
                enable_thinking,
            },
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
        self.history.entries.push(HistoryEntry::User {
            content: user_input,
        });
        self.save_history()?;

        for _ in 0..LOOP_LIMIT {
            let messages = build_messages(&self.config.workspace_root, &self.history.entries);
            let reply = call_model(&self.client, &self.config.llm, messages)?;
            let shell_reply = parse_shell_reply(&reply);

            self.history.entries.push(HistoryEntry::Assistant {
                content: reply.clone(),
            });
            self.save_history()?;

            if let Some(shell_reply) = shell_reply {
                if let Some(preamble) = shell_reply.preamble {
                    println!("{}{}", role_prefix("assistant", COLOR_GREEN), preamble);
                }

                let outcome = self.run_shell(shell_reply.request)?;
                print_tool_result(&outcome.content, outcome.success);
                self.history.entries.push(HistoryEntry::Tool {
                    command: outcome.command,
                    workdir: outcome.workdir,
                    success: outcome.success,
                    content: outcome.content,
                });
                self.save_history()?;
                continue;
            }

            println!("{}{}", role_prefix("assistant", COLOR_GREEN), reply.trim());
            return Ok(());
        }

        Err(anyhow!(
            "agent loop exceeded {LOOP_LIMIT} steps without producing a final response"
        ))
    }

    fn run_shell(&mut self, request: ShellRequest) -> Result<CommandOutcome> {
        let workdir = resolve_workdir(&self.config.workspace_root, request.workdir.as_deref())?;

        let workdir_text = workdir.display().to_string();
        print_tool_call(&request.command, &workdir_text);
        if !self.auto_approve && !prompt_for_approval(&mut self.auto_approve)? {
            return Ok(CommandOutcome {
                command: request.command,
                workdir: workdir_text,
                success: false,
                content: "command rejected by user".to_string(),
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
        let mut content = format!(
            "command: {}\nworkdir: {}\nexit_code: {}\n",
            request.command,
            workdir.display(),
            output.status.code().unwrap_or(-1)
        );
        if !stdout.trim().is_empty() {
            content.push_str("\nstdout:\n");
            content.push_str(stdout.as_ref());
        }
        if !stderr.trim().is_empty() {
            content.push_str("\nstderr:\n");
            content.push_str(stderr.as_ref());
        }

        Ok(CommandOutcome {
            command: request.command,
            workdir: workdir_text,
            success: output.status.success(),
            content,
        })
    }

    fn print_history(&self) {
        if self.history.entries.is_empty() {
            return;
        }

        println!("{}", style(COLOR_BOLD, "resumed history"));
        for entry in &self.history.entries {
            match entry {
                HistoryEntry::User { content } => {
                    println!("{}{}", role_prefix("you", COLOR_BLUE), content);
                }
                HistoryEntry::Assistant { content } => {
                    println!("{}{}", role_prefix("assistant", COLOR_GREEN), content);
                }
                HistoryEntry::Tool {
                    command,
                    workdir,
                    success,
                    content,
                } => {
                    print_tool_call(command, workdir);
                    print_tool_result(content, *success);
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

fn build_messages(workspace_root: &Path, entries: &[HistoryEntry]) -> Vec<Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": system_prompt(workspace_root)
    })];

    for entry in entries {
        match entry {
            HistoryEntry::User { content } => {
                messages.push(json!({"role": "user", "content": content}));
            }
            HistoryEntry::Assistant { content } => {
                messages.push(json!({"role": "assistant", "content": content}));
            }
            HistoryEntry::Tool {
                command,
                workdir,
                success,
                content,
            } => {
                messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "Shell command result:\ncommand: {command}\nworkdir: {workdir}\nsuccess: {success}\n{content}"
                    )
                }));
            }
        }
    }

    messages
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

fn parse_shell_reply(reply: &str) -> Option<ParsedShellReply> {
    let start = reply.find("<shell")?;
    let rest = &reply[start..];
    let tag_end = rest.find('>')?;
    let open_tag = &rest[..=tag_end];
    let close_start = rest.find("</shell>")?;
    let command = rest[tag_end + 1..close_start].trim();
    if command.is_empty() {
        return None;
    }

    let preamble = reply[..start].trim();

    Some(ParsedShellReply {
        preamble: (!preamble.is_empty()).then(|| preamble.to_string()),
        request: ShellRequest {
            command: command.to_string(),
            workdir: parse_workspace_attr(open_tag),
        },
    })
}

fn parse_workspace_attr(open_tag: &str) -> Option<String> {
    let marker = "workspace=";
    let start = open_tag.find(marker)? + marker.len();
    let quote = open_tag[start..].chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }

    let value = &open_tag[start + 1..];
    let end = value.find(quote)?;
    let workspace = value[..end].trim();
    (!workspace.is_empty()).then(|| workspace.to_string())
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
                version: 1,
                session_id: format!("session-{}-{}", now_millis(), std::process::id()),
                workspace_root: workspace_root.display().to_string(),
                last_active_at_ms: now_millis(),
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
                println!(
                    "  {}. {}  last_active={}  path={}",
                    index + 1,
                    session.history.session_id,
                    session.history.last_active_at_ms,
                    session.path.display()
                );
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
        "  mini-codex [--model MODEL] [--reasoning-effort LEVEL] [--enable-thinking BOOL] [--auto]"
    );
    println!(
        "  mini-codex resume [--last] [--model MODEL] [--reasoning-effort LEVEL] [--enable-thinking BOOL] [--auto]"
    );
}
