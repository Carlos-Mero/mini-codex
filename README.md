# mini-codex

> [!WARNING]
> This project provides only minimal safety protections and is intended for learning purposes, not production use. For real workflows, it is recommended to use [Codex](https://github.com/openai/codex) instead. The author is not responsible for any loss or damage this project may cause.

A tiny learning-oriented Codex clone.

Examples and comparisons with the original OpenAI Codex can be found in [mini-codex-examples](https://github.com/Carlos-Mero/mini-codex-examples)

## Features

- CLI chat over standard input and output
- colored terminal labels and a live working indicator while the model is running
- one tool: `shell`
- `.env` or environment variable configuration for API access
- approval mode and auto-approve mode from command-line flags
- single JSON history file per session under the system temp directory
- interactive resume flow and `resume --last`
- streaming model output with retry on API failures
- agent-loop retry after shell-tool failures by feeding the tool error back to the model
- folded terminal previews for large tool outputs while keeping the full result in session history
- workspace-scoped command validation

## Configuration

Read from `.env` in the current working directory, then overridden by normal environment variables:

- `MINI_CODEX_API_KEY` or `OPENAI_API_KEY`
- `MINI_CODEX_BASE_URL` or `OPENAI_BASE_URL`

## Run

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml
```

Set the model explicitly:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- --model gpt-5.4
```

Set the reasoning effort explicitly:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- --reasoning-effort high
```

`reasoning_effort` is attached for OpenAI- and Gemini-family models. `enable_thinking` is attached for Qwen-, DeepSeek-, and similar model families.

Disable thinking for models that use `enable_thinking` instead of `reasoning_effort`:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- --model some-other-model --enable-thinking false
```

Enable auto approval:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- --auto
```

Resume an earlier session from the current workspace:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- resume
```

Resume the last active session from the current workspace:

```bash
cargo run --manifest-path docs/mini-codex/Cargo.toml -- resume --last
```

## Commands

- `/help`
- `/exit`
- `/auto on`
- `/auto off`

## Safety model

This project keeps the shell tool intentionally lightweight:

- commands are passed as raw shell strings and executed through the local shell
- the agent can use arbitrary shell syntax for workspace tasks
- the main safety mechanism is the system prompt plus optional human approval

This is a lightweight learning setup, not a real OS sandbox.
