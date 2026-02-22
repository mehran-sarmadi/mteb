# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTEB (Massive Text Embedding Benchmark) is a Python package for benchmarking embedding models across multiple modalities, task types, languages, and domains. It includes 1000+ evaluation tasks, 50+ model integrations, and a Gradio-based leaderboard.

**Current branch (`famteb-v2-dev`)** extends MTEB for Farsi/Persian language support (FamTEB) with prompt-aware model variants.

## Common Commands

```bash
make install          # Dev install (editable + image extras + pre-commit hooks)
make test             # Run tests in parallel (pytest -n auto), excludes dataset/leaderboard tests
make lint             # Auto-fix: ruff format + ruff check --fix + typos
make lint-check       # Check only (CI mode, no auto-fix)
make pr               # Run lint + test (pre-PR checklist)
make typecheck        # mypy mteb
make test-with-coverage  # Tests with coverage report
```

**Run a single test:**
```bash
pytest tests/test_evaluate.py -k "test_name"
```

**Run tests matching a marker:**
```bash
pytest -m test_datasets          # Dataset loading tests (slow, requires downloads)
pytest -m leaderboard_stability  # Leaderboard app tests
```

**CLI usage:**
```bash
mteb run -m <model_name> -t <task1> <task2> --output-folder results
```

## Architecture

### Core Data Flow

```
get_model(name) → EncoderProtocol/CrossEncoderProtocol
get_tasks(tasks=[...]) → tuple[AbsTask, ...]
evaluate(model, tasks) → BenchmarkResults
  └─ For each task: load dataset → encode → compute metrics → cache results
```

### Key Modules

- **`mteb/evaluate.py`** — Main `evaluate()` function orchestrating the pipeline
- **`mteb/abstasks/`** — Abstract base classes for all task types (`AbsTask`, plus type-specific bases like retrieval, classification, clustering, STS, reranking, bitext mining, pair classification)
- **`mteb/abstasks/task_metadata.py`** — `TaskMetadata` dataclass with dataset descriptors, languages, domains, metrics, citations
- **`mteb/tasks/`** — 1000+ concrete task implementations organized by type then language (e.g., `tasks/retrieval/eng/`)
- **`mteb/models/`** — Model loading and protocol definitions
  - `models_protocols.py` — `EncoderProtocol`, `CrossEncoderProtocol`, `SearchProtocol`
  - `model_implementations/` — Provider-specific wrappers (sentence-transformers, OpenAI, Cohere, CLIP, etc.)
  - `get_model_meta.py` — Model discovery registry
- **`mteb/benchmarks/`** — Benchmark definitions (collections of tasks): MTEB, MMTEB, RTEB, etc.
- **`mteb/_evaluators/`** — Metric computation (NDCG, MRR, MAP for retrieval; accuracy/F1 for classification; V-measure for clustering; etc.)
- **`mteb/results/`** — Result storage: `BenchmarkResults` → `ModelResult` → `TaskResult` (JSON-serializable)
- **`mteb/cache.py`** — `ResultCache` for local/remote result caching
- **`mteb/cli/`** — CLI entry point (`mteb run`)
- **`mteb/leaderboard/`** — Gradio web app for interactive results browsing

### Task Type Hierarchy

All tasks inherit from `AbsTask`. Type-specific base classes in `mteb/abstasks/` provide evaluation logic for: Classification, Clustering, Retrieval, STS, PairClassification, Reranking, BitextMining, MultilabelClassification, InstructionRetrieval. Image/multimodal variants exist under `mteb/abstasks/image/`.

### Model Protocol System

Models are loaded via `mteb.get_model(name)` which returns an object satisfying one of the protocols defined in `models_protocols.py`. The `SentenceTransformerEncoderWrapper` handles most HuggingFace models automatically.

## Commit Convention

Commits with these prefixes trigger automatic version bumps and PyPI releases on merge to `main`:

- `fix:` / `model:` / `dataset:` → PATCH
- `feat:` → MINOR
- `breaking:` → MAJOR

Other prefixes (`docs:`, `chore:`, `refactor:`) do not trigger releases.

## Linting Rules

- **Ruff** (format + lint) targeting Python 3.10+, Google docstring convention
- **typos** for spell checking (extensive ignore list for author names, language codes)
- Docstrings not required in `tests/`, `mteb/tasks/`, `mteb/models/model_implementations/`
- Pre-commit hooks run ruff + citation formatting checks

## Testing Notes

- Tests use `pytest-xdist` for parallel execution (`-n auto`)
- Flaky HuggingFace network tests auto-retry 3 times with 10s delay
- Mock tasks/models in `tests/mock_tasks.py` and `tests/mock_models.py` for unit tests
- Coverage excludes `tests/`, `mteb/tasks/`, and `scripts/`
