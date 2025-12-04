# COMPASS Coding Agent Onboarding Guide

## 1. Project Summary
COMPASS automates discovery, extraction, and maintenance of energy infrastructure ordinances (state/local) using Large Language Models (LLMs), search/crawl tooling, and structured parsing/validation pipelines. It offers:
- Python pipeline (`compass process`) for end-to-end search → retrieval → parsing → validation → structured output.
- Rust crates: `infra-compass-db` (data storage helpers) and `infra-compass-cli` (querying/versioned ordinance data).
- Examples + docs for running extraction demos or local PDF parsing.

Primary outcomes: reproducible structured feature/value extraction and structured ordinance datasets for various ordinances.

## 2. Tech Stack Overview
- Python ≥ 3.12 (core orchestration, LLM calls, extraction & validation).
- Rust (data tooling & CLI distribution). Edition 2024.
- Environment management: **Pixi** (Conda-like) with feature-based environments (`ptest`, `pdev`, `rdev`, etc.). **Always** prefer Pixi over bare pip.
- Packaging/versioning: `setuptools_scm` (version in `compass/_version.py` generated dynamically by tags `vX.Y.Z`).
- Linting/Formatting: Ruff (configured in `pyproject.toml`). Line length 79 (72 for docstrings). Ruff also runs format check.
- Testing: Pytest (unit/integration separation), Coverage (HTML + XML). Rust uses Cargo tests + Clippy + fmt.
- CI: GitHub Actions workflows for Python (`ci-python.yml`), Rust (`ci-rust.yml`), docs, PyPI publishing.
- Docs: Sphinx with `pydata-sphinx-theme`. Build via Pixi task `python-doc` or Makefile (`make html`). Fail on warnings.
- Crawling/Search: Playwright + supplemental packages (rebrowser-playwright, camoufox, crawl4ai, tavily, ddgs, etc.).
- LLM Providers: OpenAI (core), Anthropic optional (`anthropic` extras). Configurable via API key environment variables.

## 3. High-Level Project Structure
```
compass/                Core Python package
  _cli/                 CLI entry points (script `compass`)
  extraction/           Parsing & feature extraction (solar, wind, small_wind)
  validation/           Post-extraction validation (content, graphs, location)
  services/             Abstractions for providers, queues, threading, LLM usage
  utilities/            Reusable helpers (IO, logging, enums, parsing, location)
  scripts/              Higher-level run/process automation helpers
  common/               Shared tree/base abstractions
  web/                  Crawling (website_crawl.py)
crates/                 Rust workspace (cli + db crates)
examples/               Executable tutorials & demo configs
docs/                   Sphinx docs + diagrams
tests/python/unit       Focused tests for utilities/extraction/services/etc.
tests/python/integration Integration tests for orchestrated runs
.github/workflows       CI definitions
Dockerfile              Multi-stage build (Pixi → production image)
pyproject.toml          Python deps, tasks, Ruff, coverage, Pixi config
Cargo.toml              Rust workspace definition
```

## 4. Environment & Commands Cheat Sheet
Always prefer Pixi run/tasks; they encode dependency constraints:
```
# Ad-hoc commands
pixi run <command>

# Python Unit tests / integration / all
pixi run -e pdev tests-u
pixi run -e pdev tests-i
pixi run -e pdev tests-p   # full

# Rust checks & tests
pixi install --frozen -e rdev
pixi run -e rdev cargo check --workspace --locked
pixi run -e rdev tests-r

# Build docs
pixi run -e pdoc python-doc

# Demo (requires OPENAI_API_KEY)
pixi run openai-solar-demo <api_key>

# Build wheels (Python)
pixi run -e pbuild build-wheels
```
Playwright binaries must be installed once per environment: `pixi run playwright install` (or rely on Dockerfile which does this). For production container, the entrypoint sets up environment; `run.sh` expects `JURISDICTIONS` and `COMPASS_CONFIG` env vars to render input files then run `compass process`.

To add a new dependency, use `pixi add <package>`; this updates `pyproject.toml` and the lockfile atomically.
Use `pixi add --feature python-dev <package>` to add a dependency that is only used for development (tests, linting, docs, etc.).

## 5. Coding Guidelines (Python)
- Follow the repository style guide: avoid type hints, keep numpy-style docstrings, prefer protected helpers until APIs stabilize, and mirror existing naming/visibility patterns.
- Follow Ruff configuration (79 char lines, 72 char lines for docstrings, double quotes, numpy docstyle). Run locally: `pixi run -e pdev ruff check .` and `pixi run -e pdev ruff format .` (Pixi includes ruff).
- Do not add comments unless they are absolutely critical to understanding; always prefer descriptive function and variable names instead.
- Use absolute imports under `compass.`.
- Logging: Use `logging.getLogger(__name__)`. Location-aware logging uses queue listener architecture (`utilities/logs.py`). For async tasks, ensure a `LocationFileLog` context per location when writing location-specific logs.
- Version: Never edit `_version.py` manually (auto-generated by `setuptools_scm`). Tag releases `vX.Y.Z`.
- Exceptions: Always raise specific custom exceptions from `compass.exceptions` (e.g., `COMPASSValueError`). Add new exceptions to this module if appropriate.
- Avoid blocking calls in async contexts without `await` or executor offloading.
- When adding extraction logic, mirror structure in existing solar/wind parsing modules (graphs.py, ordinance.py, parse.py) and supply validation counterparts.

## 6. Docstring Guidelines (Python)
- Use numpy-style docstrings to document classes, methods, and functions.
- Avoid type hints.
- Keep docstring length to 72 characters per line.
- Never include a period (".") at the end of the first line of docstrings.
- Do not add a short summary to __init__ methods. Instead, keep the line blank and start the "Parameters" section after a second newline.
- Do not document parameters in the class docstring - do that in the __init__ docstring instead.
- All @property and @cached_property method documentation should be one line long and should start with the return type.
- "Protected" functions and methods should always be documented using only one-line summary docstrings.
- To exclude functions or classes from the public API documentation, start the docstring with the token ``[NOT PUBLIC API]``.

## 7. Coding Guidelines (Rust)
- Workspace-managed deps; update root `Cargo.toml` if adding shared dependency.
- Enforce `pixi run -e rdev cargo fmt --all --check` and `pixi run -e rdev cargo clippy -- -D warnings` locally before PR.
- Versioning: workspace version in root `Cargo.toml`; publish tags starting with `c` (CLI) or `d` (DB) trigger release jobs.

## 8. Testing Strategy
- Unit tests target granular modules (`tests/python/unit/...`). Add new tests adjacent to similar domain (e.g., new utility → `tests/python/unit/utilities/`).
- Integration tests at `tests/python/integration` cover full pipelines.
- Coverage thresholds enforced (`--cov-fail-under=30` for unit suite). Keep defensive code minimal; exclude per coverage config if necessary.
- All python test files (e.g. ``test_scenario.py``) should end with the following block of code:

  .. code-block:: python

      if __name__ == "__main__":
          pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])

  This allows the (single) file to be executed, running only the tests contained within, which is extremely useful when updating/modifying/adding tests in the file.
- Rust tests live in crate `src` using standard Cargo conventions; prefer small, deterministic tests.

## 9. Logging & Observability
- Queue-based logging ensures non-blocking writes; to add new structured exception logging, extend `_JsonExceptionFileHandler` or filters.
- Location-specific run logs + JSON exception summaries stored under analysis run directories.
- Use `COMPASS_DEBUG_LEVEL` env if needed for custom debug filtering (present pattern). Custom log levels defined: `TRACE=5`, `DEBUG_TO_FILE=9`.

## 10. Common Pitfalls & Gotchas
- Playwright version range pinned; mismatched versions can break stealth/c4ai features. Use Pixi lock or `--frozen` installation.
- Do not modify `_version.py` directly; tests or builds may assume dynamic version structure.
- Browser install required for crawl; forgetting `playwright install` causes runtime failures.
- Some services rely on external API keys (OpenAI, Anthropic); code should gracefully handle missing keys (check provider module patterns).

## 11. Implementing a Sample Feature (Reference Flow)
Example: Add a helper to print current package version.
1. Import `from compass import __version__`.
2. Provide CLI subcommand or simple script; place in `compass/scripts/` or extend `_cli/main.py`.
3. Write unit test in `tests/python/unit/` asserting semantic version pattern.
4. Run `pixi run -e pdev tests-u` → refine on failures.

## 12. CI/Release Lifecycle
- PR: Ruff lint → unit tests (locked + multi-OS) → integration tests. Fix lint first to avoid wasted CI cycles.
- Rust: check → fmt/clippy → tests.
- Docs must build with `--fail-on-warning`; ensure new references are valid.

## 13. Contribution & Style Extensions
- Follow Development Guidelines in `docs/source/dev/README.rst` for deeper details.
- Keep functions small; prefer pure helpers in `utilities/` with focused responsibilities.
- Prefer simple dicts over dataclasses.
- Avoid complex inheritance unless mirroring existing pattern in `services/` or `extraction/`.

## 14. Quick Triage Guide for the Agent
- Need data extraction change? Inspect `extraction/<domain>/parse.py` and corresponding `validation/` modules.
- Need provider change (LLM/key mgmt)? See `services/openai.py`, `services/provider.py`.
- Need logging changes? `utilities/logs.py`.
- Need CLI behavior? `_cli/main.py` and `process.py`.

## 15. Security & Secrets
- Never commit API keys. Use environment variables (`OPENAI_API_KEY`, etc.).
- Review `Dockerfile` for production composition; builds wheels then copies source.
