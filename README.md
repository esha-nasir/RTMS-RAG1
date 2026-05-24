# RTMS-RAG — Reflexion Temporal Memory Success RAG

RTMS-RAG implements a Reflexion/Temporal-Memory-enhanced Retrieval-Augmented Generation (RAG) pipeline. It is designed to improve grounding and reduce hallucinations by combining retrieval, critic-guided scoring, iterative reflexion (self-refinement), and optional temporal-success memory for multi-step question answering (originally applied to legal QA over judgment PDFs).

Key goals:
- Improve answer grounding with retrieval + critic checks
- Iteratively refine answers through reflexion and skeptic checks
- Optionally store and reuse successful episodic memories to improve future answers

## Quick Features
- Retrieval + rerank (lexical + semantic)
- Critic/skeptic scoring and guided retries
- Temporal-success memory (save verified, grounded snippets)
- PDF ingestion and OCR fallback
- FastAPI `/ask` endpoint for queries
- Evaluation scripts and benchmarks (QReCC, RagTruth, LegalBench)

## Repository layout
- `src/rtms_rag/` — core package (moved from root). Importable as `rtms_rag`.
- `scripts/` — experiment and evaluation runners.
- `benchmarks/` — datasets and benchmark adapters.
- `data/` — raw and processed datasets.
- `outputs/` — evaluation outputs, predictions, figures.
- `memories/` — temporal memory artifacts (raw and cleaned).
- `tests/` — unit and smoke tests (copied from `RTMS-RAG/tests/`).

## Getting started (quick)
1. Create/activate virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Run unit tests:

```bash
PYTHONPATH=src /path/to/python -m pytest -q
```

3. Ingest documents (example):

```bash
PYTHONPATH=src python scripts/ingest.py
```

4. Start API:

```bash
PYTHONPATH=src uvicorn src.rtms_rag.api:app --reload
```

## Testing and integration
The current test suite contains lightweight unit and smoke tests (signature checks, chunking, HPR metric). These validate core utilities but do not call external LLMs or Pinecone. For stable integration/E2E tests we recommend mocking LLM, embedding, and vector-store calls (or recording fixtures) so reflexion loops can be exercised deterministically.

To run tests locally (example using your `rag-env`):

```bash
cd /Users/eshanasir/ReflexionTemporalMemorySuccessAgentRAG
source /Users/eshanasir/rag-env/bin/activate
PYTHONPATH=src python -m pytest -q
```

## Contributing
- Use branches and open PRs for changes.
- Keep `src/rtms_rag/` as the single source of truth for core logic; update `scripts/` to import from `src`.

If you'd like, I can add a mocked integration test that exercises `generate_answer_with_reflexion` using patched LLM and embedding functions.

# RTMS-RAG1
# RTMS-RAG1
