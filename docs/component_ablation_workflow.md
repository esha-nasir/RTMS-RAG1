# Component Ablation Workflow

This workflow tests whether each part of the Reflexion Temporal Success Memory Agent RAG improves results.

The goal is to change one component at a time while keeping the benchmark file, sample order, `top_k`, model, environment, and evaluator exactly the same.

## 1. Choose The Benchmark

Use the same dataset for every run in one experiment.

Common fixed-context choices:

```bash
data/ragtruth_fixed_context/ragtruth_fixed_context_benchmark.jsonl
data/qrecc_fixed_context_100_last50.jsonl
data/qrecc_fixed_context_100_first50.jsonl
```

For local legal benchmark runs, use:

```bash
data/benchmark_from_chunks
data/legalbenchrag_eval_aligned
```

## 2. Choose The Metrics

For RAGTruth fixed-context generation, compare:

- `generation.hallucination_rate`: lower is better.
- `generation.non_hallucination_rate`: higher is better.
- `hallucination_detection.f1`: higher is better, but only meaningful when evaluating detector behavior.

For QReCC fixed-context generation, compare:

- `qa_quality` fields in `fixed_context_summary.json`.
- `generation.hallucination_rate`.
- `retrieval_proxy.hit` if gold retrieval ids are available.

For local legal benchmark, compare:

- `qa_draft.accuracy`: higher is better.
- `qa_draft.grounded_rate`: higher is better.
- `qa_draft.hallucination_rate`: lower is better.
- `qa_draft.retrieval_hit_rate`: higher is better.
- `retrieval_strict.accuracy` and `retrieval_soft.accuracy`.

Also inspect per-row traces in the predictions JSONL:

- `trace.memory_hits`
- `trace.selected_memories`
- `trace.attempts[].skeptic`
- `trace.attempts[].critique`
- `trace.attempts[].multi_level_reflection`

## 3. Core Ablation Matrix

Run these systems on the same benchmark:

| Run | Reflexion Loop | Skeptic | Temporal Success Memory | Simple Memory | Purpose |
| --- | --- | --- | --- | --- | --- |
| A | off | on | off | off | Vanilla temporal repo answer path |
| B | on | off | off | off | Reflexion without skeptic or memory |
| C | on | on | off | off | Effect of skeptic |
| D | on | on | on | off | Effect of temporal success memory |
| E | on | on | on | on | Simple verbal memory comparison |
| F | on | on | off | off, self-reflection off | Effect of self-reflection |
| G | on | on | off | off, multi-level reflection off | Effect of retrieval/answer reflection |

The most important pairwise comparisons are:

- Skeptic effect: compare B vs C.
- Temporal memory effect: compare C vs D.
- Simple memory vs temporal scoring: compare E vs D.
- Reflexion loop effect: compare A vs C.
- Reflection submodule effect: compare C vs F and C vs G.

## 4. Fixed-Context RAGTruth/QReCC Commands

Set a run id:

```bash
cd /Users/eshanasir/ReflexionTemporalMemorySuccessAgentRAG
source .env
RUN_ID=$(date +%Y%m%d_%H%M%S)_ablation
BENCH=data/ragtruth_fixed_context/ragtruth_fixed_context_benchmark.jsonl
```

Run A, no Reflexion loop:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/A_vanilla_temporal" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run B, Reflexion without skeptic and without memory:

```bash
REFLEXION_ENABLE_SKEPTIC=0 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --disable-memory \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/B_reflexion_no_skeptic_no_memory" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run C, Reflexion with skeptic and without memory:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --disable-memory \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/C_reflexion_skeptic_no_memory" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run D, Reflexion with skeptic and temporal success memory:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
REFLEXION_DISABLE_MEMORY=0 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/D_reflexion_skeptic_temporal_memory" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run E, Reflexion with simple verbal memory:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
REFLEXION_SIMPLE_MEMORY=1 \
REFLEXION_DISABLE_MEMORY=0 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/E_reflexion_simple_memory" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run F, disable self-reflection only:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
REFLEXION_ENABLE_SELF_REFLECTION=0 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --disable-memory \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/F_no_self_reflection" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

Run G, disable multi-level reflection only:

```bash
REFLEXION_ENABLE_SKEPTIC=1 \
REFLEXION_ENABLE_MULTI_LEVEL_REFLECTION=0 \
python evaluate_cross_repo_fixed_context.py \
  --backend temporal \
  --use-reflexion \
  --disable-memory \
  --benchmark-file "$BENCH" \
  --output-dir "eval_clean_runs/${RUN_ID}/G_no_multi_level_reflection" \
  --max-iters 3 \
  --top-k 3 \
  --max-samples 100 \
  --hallucination-source final_answer \
  --detection-target generated_answer
```

## 5. Local Legal Benchmark Commands

Use this when testing retrieval plus generation on a local corpus benchmark:

```bash
cd /Users/eshanasir/ReflexionTemporalMemorySuccessAgentRAG
source .env
RUN_ID=$(date +%Y%m%d_%H%M%S)_legal_ablation
BENCH_DIR=data/legalbenchrag_eval_aligned
QA_FILE=qa_grounded_benchmark.jsonl
```

No memory:

```bash
REFLEXION_DISABLE_MEMORY=1 \
REFLEXION_ENABLE_SKEPTIC=1 \
python evaluate_local_benchmark.py \
  --benchmark-dir "$BENCH_DIR" \
  --qa-benchmark-file "$QA_FILE" \
  --output-dir "eval_clean_runs/${RUN_ID}/temporal_nomemory" \
  --benchmark-name legalbenchrag \
  --use-reflexion \
  --max-iters 3 \
  --top-k 5 \
  --qa-only \
  --judge-qa \
  --export-predictions
```

With temporal memory:

```bash
REFLEXION_DISABLE_MEMORY=0 \
REFLEXION_ENABLE_SKEPTIC=1 \
python evaluate_local_benchmark.py \
  --benchmark-dir "$BENCH_DIR" \
  --qa-benchmark-file "$QA_FILE" \
  --output-dir "eval_clean_runs/${RUN_ID}/temporal_success_memory" \
  --benchmark-name legalbenchrag \
  --use-reflexion \
  --max-iters 3 \
  --top-k 5 \
  --qa-only \
  --judge-qa \
  --export-predictions
```

## 6. Compare Summaries

Read every summary:

```bash
find "eval_clean_runs/${RUN_ID}" -name "*summary.json" -maxdepth 3 -print
```

For each ablation pair, compute deltas:

```text
delta_hallucination_rate = ablated_rate - full_system_rate
delta_accuracy = full_system_accuracy - ablated_accuracy
delta_grounded_rate = full_system_grounded_rate - ablated_grounded_rate
```

Interpretation:

- If removing a component increases hallucination rate, that component helps reduce hallucination.
- If removing a component decreases accuracy or grounded rate, that component helps answer quality.
- If memory gives `memory_hits = 0` for most rows, the memory component was enabled but not actually used enough to prove benefit.
- If skeptic lowers hallucination but raises abstention too much, report the tradeoff instead of calling it purely better.

## 7. Inspect Row-Level Evidence

Open the prediction files for changed rows:

```bash
eval_clean_runs/${RUN_ID}/C_reflexion_skeptic_no_memory/fixed_context_predictions.jsonl
eval_clean_runs/${RUN_ID}/D_reflexion_skeptic_temporal_memory/fixed_context_predictions.jsonl
```

For memory-specific analysis, check:

- Did `memory_hits` change from 0 to greater than 0?
- Were selected memories successful memories?
- Did the final answer change?
- Did hallucination detection change?
- Did the skeptic trim or abstain?

## 8. Report Format

Use a table like this:

| Component Tested | Baseline | Ablated Run | Main Metric Delta | Improved? | Notes |
| --- | --- | --- | --- | --- | --- |
| Skeptic | C | B | hallucination rate increased by X | Yes/No | Note abstention tradeoff |
| Temporal memory | D | C | accuracy changed by X | Yes/No | Include average memory hits |
| Simple vs temporal memory | D | E | hallucination/accuracy changed by X | Yes/No | Checks value of reliability/time decay |
| Self-reflection | C | F | grounded rate changed by X | Yes/No | Checks verbal self-reflection |
| Multi-level reflection | C | G | grounded rate changed by X | Yes/No | Checks retrieval/answer reflection |

## 9. Important Controls

Keep these fixed in every run:

- Same benchmark file.
- Same sample count and row order.
- Same `top_k`.
- Same `max_iters`.
- Same LLM provider and model.
- Same hallucination source, preferably `final_answer` for fair component comparison.
- Same detection target, preferably `generated_answer` when comparing generated answers.

For memory experiments, use a frozen memory file when evaluating. Build memory on a training split, then evaluate on a held-out split. Do not let the test split build the memory that it is later evaluated on.
