# RTMS-RAG — Reflexion Temporal Memory Success RAG

RTMS-RAG is a Reflexion and Temporal-Memory-enhanced Retrieval-Augmented Generation system. It is designed to improve factual grounding and reduce hallucination propagation in agentic RAG workflows by combining retrieval, critic-guided verification, skeptic checking, iterative self-reflection, and optional temporal success-aware memory.

The system was originally developed for grounded legal question answering over judgment PDFs and was later evaluated using QReCC, RAGTruth, and a local LegalBench-style corpus.

---

## Key Goals

- Improve answer grounding using retrieval and critic-based verification.
- Reduce hallucination propagation across multi-step RAG workflows.
- Refine weak answers through Reflexion-style retry logic.
- Use skeptic checks to detect unsupported or risky claims.
- Store and reuse successful episodic memories when they are reliable and low-risk.

---

## RTMS-RAG Architecture

![RTMS-RAG Architecture](docs/figures/rtms_architecture.png)

The architecture follows this general pipeline:

```text
User Question
    ↓
Evidence Retrieval
    ↓
Answer Generation
    ↓
Critic Verification
    ↓
Skeptic Guardrail
    ↓
Self-Reflection and Retry
    ↓
Temporal Success-Aware Memory
    ↓
Final Grounded Answer
```

---

## Quick Features

- Retrieval and reranking using lexical and semantic scoring.
- Critic and skeptic modules for answer verification.
- Reflexion-style self-refinement and guided retries.
- Multi-level reflection: retrieval-level, answer-level, and episode-level.
- Temporal success-aware memory.
- Hallucination-risk scoring for memory reuse.
- PDF ingestion and text chunking support.
- FastAPI `/ask` endpoint for querying the system.
- Evaluation scripts for QReCC, RAGTruth, and LegalBench-style experiments.

---

## Repository Structure

```text
ReflexionTemporalMemorySuccessAgentRAG/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .env.example
│
├── src/
│   └── rtms_rag/
│       ├── __init__.py
│       ├── api.py
│       ├── rag.py
│       ├── retrieve.py
│       ├── ingest.py
│       ├── load_data.py
│       ├── chunking.py
│       ├── yandex_embed.py
│       ├── pinecone_setup.py
│       └── compute_hpr.py
│
├── scripts/
│   ├── run_memory_only_ablation.py
│   ├── evaluate_local_benchmark.py
│   ├── evaluate_ragtruth_fixed_context.py
│   ├── evaluate_cross_repo_fixed_context.py
│   ├── evaluate_qa_hallucination_benchmark.py
│   └── export_predictions_compare.py
│
├── benchmarks/
│   ├── datasets/
│   │   ├── qrecc/
│   │   ├── ragtruth_processed/
│   │   └── legalbench_rag/
│   ├── baseline_results/
│   └── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── corpus/
│
├── memories/
│   ├── raw/
│   └── clean/
│
├── outputs/
│   ├── eval_clean_runs/
│   ├── eval_runs/
│   ├── eval_outputs/
│   ├── predictions/
│   ├── comparisons/
│   └── figures/
│
├── docs/
│   ├── architecture.md
│   ├── usage.md
│   ├── repository_structure.md
│   └── figures/
│       └── rtms_architecture.png
│
├── thesis/
│   ├── sources/
│   └── figures/
│       ├── chapter3/
│       └── results/
│
└── tests/
    ├── test_rag.py
    ├── test_retrieve.py
    ├── test_memory.py
    └── test_metrics.py
```

---

## Directory Description

### `src/rtms_rag/`

Contains the core implementation of the RTMS-RAG system. This directory is the single source of truth for the main code.

Main files:

- `rag.py` — main RTMS-RAG agent logic, answer generation, critic/skeptic checks, reflection, retry loop, and memory scoring.
- `retrieve.py` — retrieval, reranking, lexical scoring, and semantic search logic.
- `ingest.py` — document ingestion and embedding/index creation.
- `load_data.py` — dataset and document loading utilities.
- `chunking.py` — text chunking utilities.
- `yandex_embed.py` — Yandex embedding helper.
- `pinecone_setup.py` — Pinecone vector-store setup.
- `compute_hpr.py` — hallucination propagation rate calculation.
- `api.py` — FastAPI interface for querying the system.

### `scripts/`

Contains runnable experiment, evaluation, and ablation scripts.

Main scripts:

- `run_memory_only_ablation.py`
- `evaluate_local_benchmark.py`
- `evaluate_ragtruth_fixed_context.py`
- `evaluate_cross_repo_fixed_context.py`
- `evaluate_qa_hallucination_benchmark.py`
- `export_predictions_compare.py`

### `benchmarks/`

Contains benchmark-related files, dataset adapters, baseline results, and experiment configurations.

### `data/`

Contains local raw data, processed data, and corpus files. Large data files are not stored directly in the repository.

### `memories/`

Contains temporal memory artifacts used by the Reflexion and success-aware memory mechanism.

### `outputs/`

Contains generated evaluation outputs, predictions, comparison files, logs, and figures.

### `docs/`

Contains repository documentation and architecture figures. The main architecture image used in this README should be stored as:

```text
docs/figures/rtms_architecture.png
```

### `thesis/`

Contains thesis-related source files and figures.

### `tests/`

Contains unit and smoke tests for the core implementation.

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Environment Variables

Create a local `.env` file using `.env.example` as a template.

Example `.env.example`:

```bash
GEMINI_API_KEY=
YANDEX_API_KEY=
YANDEX_IAM_TOKEN=
YANDEX_FOLDER_ID=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
```

Optional Reflexion and memory settings:

```bash
REFLEXION_ENABLE_SELF_REFLECTION=1
REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION=1
REFLEXION_ENABLE_EPISODE_REFLECTION=1
REFLEXION_ENABLE_SKEPTIC=1
REFLEXION_DISABLE_MEMORY=0
REFLEXION_MEMORY_DIR=memories/clean
```

Do not commit real API keys, IAM tokens, or `.env` files to GitHub.

---

## Running Tests

Run the test suite:

```bash
PYTHONPATH=src python -m pytest -q
```

The current tests are lightweight unit and smoke tests. They check core utilities such as chunking, function signatures, and hallucination propagation rate calculation. External LLM, embedding, and vector-store calls should be mocked for stable integration testing.

---

## Ingesting Documents

Run document ingestion:

```bash
PYTHONPATH=src python -m rtms_rag.ingest
```

or, if using a script wrapper:

```bash
PYTHONPATH=src python scripts/ingest.py
```

---

## Running the API

Start the FastAPI server:

```bash
PYTHONPATH=src uvicorn rtms_rag.api:app --reload
```

Example endpoint:

```text
POST /ask
```

The API accepts a question and returns a grounded answer generated through the RTMS-RAG pipeline.

---

## Running Evaluations

Run the local legal benchmark:

```bash
PYTHONPATH=src python scripts/evaluate_local_benchmark.py
```

Run the RAGTruth fixed-context evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_ragtruth_fixed_context.py
```

Run the QReCC/cross-repository fixed-context evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_cross_repo_fixed_context.py
```

Run the QA hallucination benchmark:

```bash
PYTHONPATH=src python scripts/evaluate_qa_hallucination_benchmark.py
```

Run the memory-only ablation:

```bash
PYTHONPATH=src python scripts/run_memory_only_ablation.py
```

Export prediction comparisons:

```bash
PYTHONPATH=src python scripts/export_predictions_compare.py
```

---

## Method Summary

RTMS-RAG follows a propagation-aware RAG workflow. Instead of accepting the first generated answer, the system verifies each candidate answer before returning it or storing it in memory.

General workflow:

1. Retrieve evidence for the user question.
2. Generate a candidate answer from the retrieved context.
3. Use the critic module to check adequacy and grounding.
4. Use the skeptic module to detect unsupported claims and false-premise risks.
5. If the answer is weak, generate reflection guidance.
6. Retry answer generation using the reflection guidance.
7. Store successful and low-risk episodes in temporal memory.
8. Reuse memory only when it is relevant, successful, recent, and low-risk.

---

## Reflexion Logic

The system includes Reflexion-style self-reflection logic. Failed or weak attempts are diagnosed using critic and skeptic feedback. This feedback is transformed into reflection guidance and reused during later attempts.

The implementation supports:

- retrieval-level reflection
- answer-level reflection
- episode-level reflection
- critic-guided retry
- skeptic-guided retry
- memory-aware reflection reuse

---

## Temporal Success-Aware Memory

The memory system stores verified episodes with metadata such as:

- question
- answer
- reflection guidance
- benchmark name
- task type
- success status
- success count
- failure count
- hallucination risk
- timestamp
- source evidence metadata

During later queries, memory entries are scored using:

- semantic similarity
- temporal decay
- success history
- failure history
- hallucination-risk penalty
- benchmark, file, or case matching

Only useful and low-risk memories are reused.

---

## Dataset Notice

Large benchmark datasets, processed chunks, memory artifacts, and evaluation outputs are not stored directly in this repository because they may exceed GitHub file-size limits.

The repository contains:

- source code
- experiment scripts
- configuration files
- documentation
- small examples
- tests

Full datasets and generated artifacts should be downloaded or created locally.

---

## GitHub File Policy

Do not commit:

```text
.env
.env.*
large dataset files
processed chunk files
.arrow files
evaluation outputs
memory JSONL artifacts
local vector indexes
PDF corpora
```

Safe files to commit:

```text
README.md
requirements.txt
.env.example
source code
scripts
tests
small examples
configuration templates
documentation
architecture figures
```

---

## Citation

If this repository is used for academic work, cite the accompanying thesis:

```text
Nasir, E. (2026). Reducing Hallucination Propagation in Agentic Retrieval-Augmented Generation Using Self-Reflective Verification and Temporal Success-Aware Memory. Master's Thesis, Moscow Institute of Physics and Technology.
```

---

## License

This project is released under the terms specified in the `LICENSE` file.

---

## Contributing

- Keep `src/rtms_rag/` as the single source of truth for core logic.
- Do not duplicate source files in the repository root.
- Place experiment runners in `scripts/`.
- Store generated results in `outputs/`.
- Store temporal memory artifacts in `memories/`.
- Do not commit secrets, API keys, tokens, or large generated files.
- Use branches and pull requests for major changes.
