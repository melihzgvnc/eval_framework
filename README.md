# Eval Framework

A modular Python framework for evaluating AI model outputs locally. Designed for teams that need reproducible, claim-level evaluation of model faithfulness, hallucination, and refusal behavior — without sending data to external evaluation services.

## Architecture

```
eval_framework/
├── datasets/       Load and validate pre-generated model outputs (JSONL, JSON, CSV)
├── claims/         Decompose responses into atomic, verifiable claims
├── evaluators/     Score claims via NLI or LLM-as-a-Judge
├── metrics/        Aggregate into hallucination rate, groundedness, refusal precision/recall
├── reports/        Render results as JSON, Markdown, or interactive HTML
├── pipelines/      Compose components into end-to-end evaluation workflows
├── cache/          Persist expensive LLM/NLI results across runs (SQLite)
├── models/         Model wrappers (OpenAI, HuggingFace)
├── judges/         Prompt templates and schemas for LLM-as-a-Judge
├── nli/            NLI relation definitions
└── core/           Base abstractions and shared types
```

## Quick Start

```python
from eval_framework.datasets import load_dataset
from eval_framework.pipelines import create_nli_pipeline
from eval_framework.cache import SQLiteCache
from eval_framework.reports import ReportBuilder, ReportMetadata, RunStorage

# 1. Load dataset (pre-generated model outputs)
samples, info, validation = load_dataset("data/eval_set.jsonl")

# 2. Create and run pipeline
pipeline = create_nli_pipeline(
    nli_model_path="cross-encoder/nli-deberta-v3-base",
    decomposer_model="gpt-4o",
    cache=SQLiteCache(),
)
result = pipeline.run(samples)

# 3. Inspect results
print(result.summary())
for m in result.metrics:
    print(f"  {m.name}: {m.value:.4f}")

# 4. Generate report
report = ReportBuilder().build(
    metrics=result.metrics,
    samples=result.sample_results,
    metadata=ReportMetadata(
        run_id="run_001", model_name="gpt-4o", evaluator_type="nli",
    ),
)
RunStorage().save(report, samples=result.sample_results)
```

## Evaluation Strategies

### NLI (Natural Language Inference)

Classifies each claim as entailed, contradicted, or neutral against the source text using a local sequence-classification model.

```python
from eval_framework.pipelines import create_nli_pipeline

pipeline = create_nli_pipeline(nli_model_path="cross-encoder/nli-deberta-v3-base")
```

### LLM-as-a-Judge

Uses a language model to score claims on configurable criteria (factuality, helpfulness, coherence, comprehensiveness).

```python
from eval_framework.pipelines import create_judge_pipeline

pipeline = create_judge_pipeline(
    judge_model="gpt-4o",
    criteria="factuality",
    decompose=True,
)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `hallucination_rate` | Proportion of claims not supported by source |
| `groundedness` | Proportion of claims supported by source |
| `refusal_precision` | Of all refusals, how many were correct |
| `refusal_recall` | Of cases requiring refusal, how many were caught |
| `over_refusal_rate` | Of answerable cases, how many were incorrectly refused |

## Dataset Format

The framework evaluates pre-generated model outputs. Datasets can be JSONL, JSON, or CSV with automatic column detection:

```jsonl
{"id": "s1", "context": "Paris is the capital of France.", "response": "Paris is the capital.", "query": "What is the capital?"}
{"id": "s2", "context": "Water boils at 100C.", "response": "Water boils at 100C.", "query": "Boiling point?", "should_refuse": false}
```

Custom column names are supported via explicit mapping:

```python
from eval_framework.datasets import load_dataset, ColumnMapping

samples, _, _ = load_dataset("data.csv", column_mapping=ColumnMapping(
    source_text="ground_truth",
    model_response="llm_output",
))
```

## Caching

Expensive operations (LLM calls, NLI inference) are cached in SQLite by default at `~/.cache/eval_framework/cache.db`. Pass a cache instance to any component:

```python
from eval_framework.cache import SQLiteCache

cache = SQLiteCache()  # or SQLiteCache(path="./my_cache.db")
pipeline = create_nli_pipeline(..., cache=cache)

# After run
print(cache.stats.hit_rate)
```

## Reports

Reports are generated in multiple formats from a single structured `Report` object:

- **JSON** — Machine-readable source of truth
- **Markdown** — Git-friendly, drops into PRs
- **HTML** — Self-contained interactive view with charts

Multi-run comparison reports identify winners per metric and show deltas against a baseline.

## Custom Pipelines

For full control, compose steps manually:

```python
from eval_framework.pipelines import EvalPipeline, DecomposeStep, EvaluateStep, AdaptStep

pipeline = EvalPipeline(
    evaluator=my_evaluator,
    adapter=my_adapter,
    decomposer=my_decomposer,
    metrics=[hallucination_rate, groundedness],
    continue_on_error=True,
    progress_callback=lambda cur, total, sid: print(f"{cur}/{total}"),
)
```

## Requirements

- Python 3.8+
- `openai` — for LLM Judge and claim decomposition
- `transformers` + `torch` — for local NLI models (optional, only if using NLI evaluator)

## License

Internal use.
