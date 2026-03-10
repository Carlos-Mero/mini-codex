# mini-codex

> [!WARNING]
> This project provides only minimal safety protections and is intended for learning purposes, not production use. For real workflows, it is recommended to use [Codex](https://github.com/openai/codex) instead. The author is not responsible for any loss or damage this project may cause.

A tiny learning-oriented Codex clone.

Examples and comparisons with the original OpenAI Codex can be found in [mini-codex-examples](https://github.com/Carlos-Mero/mini-codex-examples)

## Features

- simple CLI chat interface
- built-in `shell` tool for local workspace tasks
- `.env` and environment variable support for API configuration
- manual approval mode or `--auto` for tool execution
- resumable sessions with `resume` and `resume --last`
- streaming responses with basic retry handling
- workspace-scoped command validation

## Configuration

Configuration is loaded in this order:

1. `.env` in the current working directory
2. normal environment variables

Environment variables override values from `.env`.

Supported variables:

- `MINI_CODEX_API_KEY` or `OPENAI_API_KEY`
- `MINI_CODEX_BASE_URL` or `OPENAI_BASE_URL`

### Option 1: use a `.env` file

Create a `.env` file in the directory where you run `mini-codex`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

You can also use the `MINI_CODEX_*` names instead:

```env
MINI_CODEX_API_KEY=your_api_key_here
MINI_CODEX_BASE_URL=https://api.openai.com/v1
```

### Option 2: set shell environment variables

For the current shell session:

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.openai.com/v1
```

To make them persistent, add them to your shell config file such as `~/.zshrc` or `~/.bashrc`, then reload your shell:

```bash
source ~/.zshrc
```

## Install

Install from the current repository:

```bash
cargo install --path .
```

Then run:

```bash
mini-codex
```

If you update the source and want to reinstall:

```bash
cargo install --path . --force
```

## Run

For local development, run directly with Cargo:

```bash
cargo run -- --model gpt-5.4
```

Common examples:

```bash
cargo run -- --reasoning-effort high
cargo run -- --auto
cargo run -- resume
cargo run -- resume --last
```

### Thinking settings

- `reasoning_effort` is used for OpenAI- and Gemini-family models
- `enable_thinking` is used for Qwen-, DeepSeek-, and similar model families

Example:

```bash
cargo run -- --model some-other-model --enable-thinking false
```

## Commands

- `/continue` - retry the previous turn from current history without adding a new user message
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
