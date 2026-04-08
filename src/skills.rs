use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SkillMetadata {
    pub(crate) dir_path: PathBuf,
    pub(crate) path_to_skills_md: PathBuf,
    pub(crate) name: String,
    pub(crate) description: String,
}

pub(crate) fn discover_skills(
    workspace_root: &Path,
    external_roots: &[PathBuf],
) -> Result<Vec<SkillMetadata>> {
    let mut roots = default_skill_roots(workspace_root);
    roots.extend(external_roots.iter().cloned());

    let mut skills = Vec::new();
    for root in roots {
        let canonical_root = match canonicalize_existing_dir(&root)? {
            Some(path) => path,
            None => continue,
        };

        if canonical_root.join("SKILL.md").is_file() {
            if let Some(skill) = load_skill_from_dir(&canonical_root)? {
                skills.push(skill);
            }
            continue;
        }

        let entries = match fs::read_dir(&canonical_root) {
            Ok(entries) => entries,
            Err(err) => {
                return Err(err).with_context(|| {
                    format!(
                        "failed to read skills directory {}",
                        canonical_root.display()
                    )
                });
            }
        };

        for entry in entries {
            let path = entry?.path();
            let Some(path) = canonicalize_existing_dir(&path)? else {
                continue;
            };
            if let Some(skill) = load_skill_from_dir(&path)? {
                skills.push(skill);
            }
        }
    }

    skills.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.dir_path.cmp(&right.dir_path))
    });
    skills.dedup_by(|left, right| left.dir_path == right.dir_path);
    Ok(skills)
}

pub(crate) fn render_skills_section(skills: &[SkillMetadata]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push("## Skills".to_string());
    lines.push("A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.".to_string());
    lines.push("### Available skills".to_string());

    for skill in skills {
        let path_str = skill.path_to_skills_md.to_string_lossy().replace('\\', "/");
        let name = skill.name.as_str();
        let description = skill.description.as_str();
        lines.push(format!("- {name}: {description} (file: {path_str})"));
    }

    lines.push("### How to use skills".to_string());
    lines.push(
        r###"- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, use your shell_tool to read its `SKILL.md`. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."###
            .to_string(),
    );

    Some(lines.join("\n"))
}

pub(crate) fn parse_external_skill_roots(value: &str) -> Vec<PathBuf> {
    env::split_paths(value)
        .filter(|path| !path.as_os_str().is_empty())
        .collect()
}

fn default_skill_roots(workspace_root: &Path) -> Vec<PathBuf> {
    let mut roots = vec![
        workspace_root.join(".mini-codex/skills"),
        workspace_root.join(".agents/skills"),
    ];

    if let Some(home) = home_dir() {
        roots.push(home.join(".mini-codex/skills"));
        roots.push(home.join(".agents/skills"));
    }

    roots
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn canonicalize_existing_dir(path: &Path) -> Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    if !path.is_dir() {
        return Ok(None);
    }
    path.canonicalize()
        .map(Some)
        .with_context(|| format!("failed to resolve {}", path.display()))
}

fn load_skill_from_dir(path: &Path) -> Result<Option<SkillMetadata>> {
    let skill_md = path.join("SKILL.md");
    if !skill_md.is_file() {
        return Ok(None);
    }

    let text = fs::read_to_string(&skill_md)
        .with_context(|| format!("failed to read {}", skill_md.display()))?;
    let Some((name, description)) = parse_skill_frontmatter(&text) else {
        return Ok(None);
    };

    Ok(Some(SkillMetadata {
        dir_path: path.to_path_buf(),
        path_to_skills_md: skill_md,
        name,
        description,
    }))
}

fn parse_skill_frontmatter(text: &str) -> Option<(String, String)> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }

    let mut name = None;
    let mut description = None;

    for line in lines {
        let trimmed = line.trim();
        if trimmed == "---" {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("name:") {
            let parsed = trim_yaml_scalar(value);
            if !parsed.is_empty() {
                name = Some(parsed);
            }
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("description:") {
            let parsed = trim_yaml_scalar(value);
            if !parsed.is_empty() {
                description = Some(parsed);
            }
        }
    }

    match (name, description) {
        (Some(name), Some(description)) => Some((name, description)),
        _ => None,
    }
}

fn trim_yaml_scalar(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{parse_external_skill_roots, parse_skill_frontmatter, render_skills_section};
    use std::path::PathBuf;

    #[test]
    fn parses_skill_frontmatter() {
        let text = "\
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files.
---

# PDF Processing
";

        let parsed = parse_skill_frontmatter(text);
        assert_eq!(
            parsed,
            Some((
                "pdf-processing".to_string(),
                "Extract PDF text, fill forms, merge files.".to_string()
            ))
        );
    }

    #[test]
    fn renders_skills_section() {
        let rendered = render_skills_section(&[super::SkillMetadata {
            dir_path: PathBuf::from("/tmp/pdf-processing"),
            path_to_skills_md: PathBuf::from("/tmp/pdf-processing/SKILL.md"),
            name: "pdf-processing".to_string(),
            description: "Extract PDF text.".to_string(),
        }])
        .expect("skills section");

        assert!(rendered.contains("## Skills"));
        assert!(
            rendered.contains(
                "- pdf-processing: Extract PDF text. (file: /tmp/pdf-processing/SKILL.md)"
            )
        );
    }

    #[test]
    fn parses_external_skill_roots_with_platform_separator() {
        let joined = std::env::join_paths([
            PathBuf::from("/tmp/skills-a"),
            PathBuf::from("/tmp/skills-b"),
        ])
        .expect("joined paths");

        let parsed = parse_external_skill_roots(joined.to_str().expect("utf-8"));
        assert_eq!(
            parsed,
            vec![
                PathBuf::from("/tmp/skills-a"),
                PathBuf::from("/tmp/skills-b")
            ]
        );
    }
}
