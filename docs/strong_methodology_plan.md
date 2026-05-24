# Strong Methodology Plan

## Research Title

Temporal Success-Aware Episodic Memory for Reflexion-Based Retrieval-Augmented Generation

## Core Research Problem

Retrieval-Augmented Generation systems often fail because retrieved evidence is incomplete, weakly ranked, or incorrectly used by the generator. Reflexion-style agents can critique and revise their outputs, but they may still repeat previous mistakes because prior reflections are usually reused without a principled estimate of whether they are relevant, recent, reliable, or hallucination-prone.

This research studies whether a temporal success-aware episodic memory controller can improve grounded answer quality and reduce hallucination in Reflexion-based RAG.

## Main Research Question

Does temporal success-aware episodic memory improve grounded answer quality and reduce hallucination compared with vanilla RAG, Reflexion-only RAG, and simple memory-based RAG?

## Sub-Questions

1. Does Reflexion improve grounded QA performance over vanilla RAG?
2. Does episodic memory improve Reflexion-based RAG?
3. Does weighting memory by temporal decay, reliability, and hallucination risk outperform similarity-only memory retrieval?
4. Which memory scoring component contributes most to performance?
5. Does the proposed memory controller reduce negative transfer from failed past reasoning episodes?

## Hypotheses

H1: Reflexion-based RAG will reduce unsupported answers compared with vanilla RAG.

H2: Reflexion with temporal success-aware memory will outperform Reflexion without memory.

H3: The full memory score will outperform similarity-only memory retrieval.

H4: Hallucination-risk suppression will reduce hallucinated outputs and false unsupported claims.

H5: Temporal decay will improve robustness when older memories become less useful than recent successful episodes.

## Proposed Method

The system has four main layers:

1. Evidence retrieval layer.
2. Answer generation layer.
3. Verification and Reflexion layer.
4. Temporal success-aware episodic memory layer.

The proposed memory score is:

```text
score =
    similarity
  * temporal_weight
  * reliability
  * risk_penalty
  * (1 + metadata_bonus)
```

Where:

```text
similarity        Jaccard token similarity between current and past questions.
temporal_weight   exp(-age_days / decay_days).
reliability       alpha / (alpha + beta).
risk_penalty      1 - hallucination_penalty * hallucination_risk.
metadata_bonus    bonus for same case_no or file_name.
```

The memory is updated after each Reflexion episode. Successful reuse increases `alpha`; failed reuse increases `beta`. This creates a lightweight Bayesian reliability estimate for each memory.

## Experimental Conditions

Use the same retrieval settings, model, decoding parameters, and benchmark split across all systems.

### System Baselines

```text
B0: Vanilla RAG
B1: RAG + skeptic/critic guardrail
B2: Reflexion RAG without memory
B3: Reflexion RAG with similarity-only memory
B4: Reflexion RAG with simple success memory
B5: Reflexion RAG with temporal-only memory
B6: Reflexion RAG with reliability-only memory
B7: Reflexion RAG with hallucination-risk memory
B8: Full temporal success-aware memory
```

The main comparison is:

```text
Vanilla RAG vs Reflexion-only RAG vs Full proposed method
```

The ablation comparison is:

```text
Full proposed method vs removed scoring components
```

## Datasets

Recommended dataset groups:

```text
RAGTruth fixed-context
  Measures hallucination behavior when evidence is fixed.

LegalBench-RAG / privacy_qa / contractnli / cuad / maud
  Measures legal grounded QA and citation-sensitive answer generation.

QReCC fixed-context or conversational QA
  Measures conversational QA transfer behavior.
```

## Data Splitting

To avoid leakage, use one of these two setups.

### Offline Evaluation Setup

```text
Train split       Build initial memory.
Validation split  Tune thresholds and environment parameters.
Test split        Final evaluation only once.
```

This is the cleanest setup for thesis or paper claims.

### Online Learning Setup

```text
Process examples sequentially.
Only previous examples can become memory.
Future examples must never influence earlier predictions.
Report performance over time.
Repeat with multiple shuffled orders.
```

This setup is acceptable if the work is framed as online adaptation.

## Metrics

### Grounded QA Metrics

```text
answer accuracy
groundedness rate
hallucination rate
unsupported claim rate
abstention rate
retrieval hit rate
span hit rate
supported-but-wrong rate
average answer length
```

### Hallucination Detection Metrics

```text
accuracy
precision
recall
F1
false positive rate
false negative rate
span-level precision
span-level recall
span-level F1
```

### Agent Behavior Metrics

```text
memory hit rate
average selected memory score
average reliability of selected memories
average hallucination risk of selected memories
average iterations
early stop rate
latency
token cost
```

## Statistical Analysis

For PhD-level rigor, report:

```text
mean performance across runs
95% bootstrap confidence intervals
paired significance test against baselines
effect size
per-dataset breakdown
per-task breakdown
```

For online memory evaluation, repeat the experiment with several random dataset orders and report mean and variance.

## Ablation Study

Run the proposed method with one component removed at a time:

```text
Full score
No temporal decay
No reliability
No hallucination-risk penalty
No metadata bonus
Similarity only
Random memory
No memory
```

This will show whether each component actually contributes.

## Failure Analysis

Manually inspect a representative sample of errors:

```text
retrieval failure
evidence not present
evidence present but ignored
unsupported inference
over-abstention
citation error
bad memory transfer
critic false positive
critic false negative
```

For each error type, include examples from predictions and explain whether memory helped or hurt.

## Expected Tables

### Main Result Table

```text
System | Accuracy | Groundedness | Hallucination Rate | Abstention Rate | Avg Iterations
```

### Hallucination Detection Table

```text
System | Accuracy | Precision | Recall | F1 | FP | FN
```

### Ablation Table

```text
Variant | Accuracy | Hallucination Rate | Memory Hit Rate | Avg Memory Score
```

### Memory Quality Table

```text
Variant | Selected Memory Reliability | Selected Memory Risk | Success After Reuse
```

## Methodology Chapter Structure

1. Introduction to the problem.
2. Limitations of vanilla RAG and unweighted memory reuse.
3. Proposed architecture.
4. Temporal success-aware memory scoring.
5. Reflexion and verification mechanism.
6. Experimental setup.
7. Baselines and ablations.
8. Metrics.
9. Statistical testing.
10. Results.
11. Failure analysis.
12. Threats to validity.

## Threats to Validity

Important risks to explicitly discuss:

```text
LLM judge bias
benchmark leakage through memory
order sensitivity in online learning
dataset-specific prompt tuning
retrieval dependency
small sample sizes
model/provider variability
cost and latency tradeoffs
```

## Strong Contribution Statement

This research introduces a temporal success-aware episodic memory controller for Reflexion-based RAG agents. Unlike conventional memory retrieval, which primarily relies on semantic similarity, the proposed method ranks previous reasoning episodes using semantic relevance, exponential temporal decay, Bayesian success reliability, hallucination-risk suppression, and task metadata locality. The memory is updated through append-only replay outcome logs, making adaptation auditable. The methodology evaluates whether selective reuse of prior reflections improves grounded answer quality and reduces hallucination compared with vanilla RAG, Reflexion-only RAG, and simpler memory-selection baselines.
