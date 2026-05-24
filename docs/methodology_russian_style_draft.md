# Methodology Draft

## Chapter 3. Experimental Methodology

### 3.1 Research Goal

The goal of this research is to develop and evaluate a Reflexion-based Retrieval-Augmented Generation system with temporal success-aware episodic memory. The proposed system, ReflexionTemporalMemorySuccessAgentRAG, is designed to reduce hallucination propagation and improve grounded answer quality by reusing previous reasoning episodes only when they are relevant, recent, reliable, and low-risk.

In contrast to standard RAG systems, which usually perform retrieval and generation in a single pass, the proposed approach introduces an iterative verification loop and an adaptive memory mechanism. The system does not simply store previous answers. Instead, it stores reasoning episodes together with success and failure statistics, allowing future queries to benefit from reliable past reflections while suppressing memories that previously led to hallucinated or unsupported answers.

### 3.2 Research Tasks

To achieve the research goal, the following tasks are defined:

1. Analyze existing approaches to Retrieval-Augmented Generation, Agentic RAG, Reflexion-based reasoning, and memory-augmented language agents.
2. Implement a baseline RAG pipeline for grounded question answering.
3. Implement a Reflexion loop that generates, critiques, verifies, and revises answers using retrieved evidence.
4. Develop a temporal success-aware episodic memory mechanism for selecting useful prior reasoning episodes.
5. Compare the proposed system with vanilla RAG, self-refinement, simple Reflexion memory, and temporal Reflexion without memory.
6. Evaluate the systems on conversational QA, hallucination-focused, and source-grounded benchmarks.
7. Analyze whether temporal success-aware memory reduces hallucination and improves answer quality compared with non-memory and simple-memory baselines.

### 3.3 Object and Subject of Research

The object of research is retrieval-augmented generation for question answering with large language models.

The subject of research is the use of temporal success-aware episodic memory inside a Reflexion-based Agentic RAG pipeline. Special attention is given to memory selection, memory reliability estimation, hallucination-risk suppression, and iterative answer verification.

### 3.4 Overall System Architecture

The proposed architecture consists of four main components:

1. Evidence retrieval module.
2. Answer generation module.
3. Reflexion and verification module.
4. Temporal success-aware episodic memory module.

At inference time, the system receives a user question and retrieves relevant evidence passages. If memory is enabled, the memory module searches for previous reasoning episodes that may help answer the current question. The generator then produces a candidate answer using the retrieved evidence and selected memory guidance. After generation, the Reflexion module checks the answer for unsupported statements, incomplete reasoning, and possible hallucination. If the answer is judged weak, the system produces reflection guidance and performs another generation attempt. After the final attempt, the outcome is saved as a new memory episode and the statistics of reused memories are updated.

This architecture differs from vanilla RAG in two important ways. First, answer generation is treated as an iterative process rather than a one-step operation. Second, previous reasoning episodes are treated as selective and risk-controlled guidance, not as unconditional prompt additions.

### 3.5 Retrieval Module

The retrieval module provides the factual evidence used by the generator and verifier. In fixed-context experiments, retrieval is controlled so that each system receives comparable evidence under the same top-k setting. This is important because the purpose of the experiment is to compare reasoning and memory mechanisms, not to measure differences caused by unequal retrieval conditions.

The main retrieval parameter is `top_k`, which defines the number of evidence passages available to the answer generator. In the main QReCC and RAGTruth comparisons, `top_k = 3` is used. Retrieval quality is measured using a source-hit proxy where gold evidence identifiers are available.

### 3.6 Reflexion and Verification Loop

The Reflexion mechanism is used to reduce unsupported generation by checking and revising candidate answers. Each reasoning episode may contain several iterations. In the implemented experiments, the maximum number of iterations is set to three.

Each iteration follows this sequence:

1. Generate a candidate answer from the retrieved evidence and optional memory guidance.
2. Apply a skeptic or critic step to identify unsupported claims, false-premise risks, and missing evidence.
3. Produce reflection guidance if the answer is incomplete or insufficiently grounded.
4. Retry answer generation if another iteration is allowed.
5. Return the final answer after the stopping condition is reached.

This loop is designed to prevent hallucinations from propagating through the reasoning process. Instead of accepting the first generated answer, the system explicitly evaluates whether the answer is supported by the available evidence.

### 3.7 Temporal Success-Aware Episodic Memory

The central contribution of this research is the temporal success-aware episodic memory mechanism. Each memory stores information about a previous reasoning episode, including the original question, the reflection text, the final outcome, metadata, and success/failure statistics.

A memory is not selected only because it is semantically similar to the current question. The proposed memory score combines five factors:

```text
memory_score =
    similarity
  * temporal_weight
  * reliability
  * risk_penalty
  * (1 + metadata_bonus)
```

The components are defined as follows:

```text
similarity        = lexical token overlap between current and previous question
temporal_weight   = exp(-age_days / decay_days)
reliability       = alpha / (alpha + beta)
risk_penalty      = max(epsilon, 1 - lambda * hallucination_risk)
metadata_bonus    = bonus for matching file or case metadata
```

The reliability term increases when a memory is reused successfully and decreases when reuse is associated with failure. Hallucination risk is estimated from previous unsuccessful outcomes. Temporal decay prevents old memories from dominating the prompt when more recent and useful reasoning episodes are available.

### 3.8 Memory Update Procedure

After each reasoning episode, the system stores a new memory record if memory saving is enabled for the benchmark profile. The system also updates the statistics of any memories that were reused during the episode.

If the final answer is successful, the reused memories receive a positive update:

```text
alpha = alpha + 1
```

If the final answer is unsuccessful, the reused memories receive a negative update:

```text
beta = beta + 1
```

This update procedure allows the system to estimate memory usefulness over time. Memories that repeatedly help the model become more likely to be reused. Memories associated with hallucination or poor answers become less likely to be selected.

### 3.9 Benchmark Selection

The evaluation uses three benchmark groups, each serving a different methodological purpose.

QReCC is used as the primary benchmark for memory evaluation. It is a conversational question answering benchmark, making it suitable for testing whether previous reasoning episodes can improve later answers. Conversational QA naturally contains context dependency, repeated reasoning patterns, and reformulation challenges, which are appropriate conditions for evaluating episodic memory.

RAGTruth is used as a hallucination-focused benchmark. Its role is to evaluate whether the system can reduce unsupported generation under fixed evidence conditions. In the current setup, RAGTruth is more suitable for evaluating groundedness and hallucination behavior than for proving memory effectiveness.

LegalBench-RAG and local legal QA data are used as domain-specific grounded QA benchmarks. They test whether the system can answer questions using source-sensitive evidence and whether memory helps with structured, citation-aware reasoning.

### 3.10 Experimental Design

The main experimental design compares several systems under matched retrieval and generation settings:

```text
B0: Vanilla RAG
B1: Self-refine / Agentic RAG
B2: Reflexion RAG with simple verbal memory
B3: Temporal Reflexion RAG without active memory
B4: Reflexion RAG with temporal success-aware memory
```

The most important memory-specific comparison is:

```text
Temporal Reflexion without memory
vs
Temporal Reflexion with success-aware memory
```

This comparison isolates the effect of memory from the effect of the Reflexion architecture itself.

For clean memory testing, the preferred QReCC setup is:

```text
First 50 QReCC examples: build memory
Last 50 QReCC examples: evaluate with and without memory
```

This prevents leakage from test examples into memory. The system can only reuse information from previous examples, which matches the intended online memory setting.

### 3.11 Evaluation Metrics

The evaluation uses several metric groups.

Answer quality is measured using:

```text
Answer F1
Exact match rate
```

Hallucination behavior is measured using:

```text
Detector-estimated hallucination rate
Non-hallucination rate
Unsupported answer rate where available
```

Retrieval quality is measured using:

```text
Retrieval hit rate or source-hit proxy
```

Agent and memory behavior are measured using:

```text
Memory hit count
Selected memory score
Reliability of selected memories
Hallucination risk of selected memories
Number of Reflexion iterations
```

QReCC does not provide gold hallucination annotations in the adapted fixed-context setup. Therefore, hallucination results on QReCC should be described as detector-estimated hallucination rates rather than ground-truth hallucination labels.

### 3.12 Main QReCC Results Used for Methodological Validation

The clean QReCC comparison shows that the proposed temporal approach improves answer quality and reduces detector-estimated hallucination compared with vanilla RAG.

```text
Method                         Hallucination   Answer F1   Exact Match
Vanilla RAG                    0.58            0.3341      0.00
Self-refine AgentRAG           0.62            0.3236      0.00
Reflexion simple memory        0.26            0.4066      0.00
Temporal no memory             0.28            0.5238      0.02
Temporal success memory        0.18            0.5199      0.00
```

These results suggest that the Reflexion-based temporal architecture substantially improves QReCC fixed-context QA. The comparison between temporal no-memory and temporal success memory also shows a reduction in detector-estimated hallucination from 0.28 to 0.18, indicating that memory can provide additional benefit when evaluated under a clean split.

### 3.13 Ablation Study

To understand which memory components contribute to performance, the proposed system should be compared with ablated variants:

```text
Full temporal success-aware memory
No temporal decay
No reliability weighting
No hallucination-risk penalty
No metadata bonus
Similarity-only memory
Random memory
No memory
```

This ablation design is necessary because a strong final result alone does not prove which mechanism caused the improvement. The ablation study determines whether the gain comes from temporal recency, reliability estimation, hallucination-risk suppression, or simple similarity matching.

### 3.14 Validity and Limitations

Several limitations must be acknowledged.

First, QReCC in the adapted setup does not contain gold hallucination labels. Therefore, hallucination values on this benchmark are detector-estimated and should be interpreted as an auxiliary signal.

Second, memory-augmented systems can suffer from leakage if future test examples are stored before evaluation. This risk is addressed by using a sequential memory-building protocol where only earlier examples can be used as memory.

Third, improvements may depend on the quality of retrieved evidence. To reduce this confound, systems are compared under the same retrieval depth and benchmark split.

Fourth, LLM-based critics and hallucination detectors may be imperfect. For this reason, answer F1, exact match, retrieval proxy metrics, and manual failure analysis should be reported together with hallucination estimates.

### 3.15 Chapter Summary

This methodology evaluates ReflexionTemporalMemorySuccessAgentRAG as a temporal memory-augmented RAG system. The proposed method combines evidence retrieval, iterative Reflexion, and adaptive episodic memory selection. QReCC is used as the primary benchmark for memory evaluation because conversational QA is naturally suited to testing reuse of previous reasoning episodes. RAGTruth is used for hallucination-focused validation, while legal QA data is used to test grounded reasoning in a specialized domain.

The key methodological comparison is between temporal Reflexion without memory and temporal Reflexion with success-aware memory. This comparison determines whether the proposed memory mechanism provides value beyond the Reflexion architecture itself.

## Chapter 4. Proposed Method

### 4.1 Motivation

The analysis of existing RAG and Agentic RAG approaches shows that retrieval alone is not sufficient to guarantee grounded and reliable answers. A standard RAG system may retrieve relevant evidence but still generate an unsupported answer if the language model ignores the evidence, overgeneralizes from partial context, or introduces external knowledge. Agentic RAG improves this process by adding planning and multi-step reasoning, but it can also amplify errors when an incorrect intermediate assumption is reused in later steps.

Reflexion-based systems address part of this problem by allowing the model to critique and revise its own output. However, a Reflexion loop without memory treats each new query almost independently. It can correct the current answer, but it does not systematically learn which previous reasoning patterns were successful and which ones caused hallucination or unsupported claims.

The proposed method is motivated by this limitation. The main idea is to extend Reflexion-based RAG with temporal success-aware episodic memory. The system should reuse previous reasoning experience only when that experience is likely to help the current question. Therefore, the memory controller must consider not only similarity, but also recency, reliability, hallucination risk, and task-specific metadata.

### 4.2 Proposed System Overview

The proposed system is called ReflexionTemporalMemorySuccessAgentRAG. It combines retrieval-augmented generation, iterative self-verification, and adaptive episodic memory. The system receives a question, retrieves relevant context, selects useful memories, generates an answer, verifies the answer, and updates memory after the episode is completed.

The proposed method contains the following main stages:

1. Receive the user query and benchmark metadata.
2. Retrieve top-k evidence passages from the corpus or fixed context.
3. Select relevant past memories using temporal success-aware scoring.
4. Generate an initial answer from evidence and memory guidance.
5. Critique the answer using skeptic and verification steps.
6. Revise the answer if unsupported or incomplete claims are detected.
7. Save the final reasoning episode as a new memory.
8. Update success or failure statistics of reused memories.

This design allows the system to adapt across episodes without retraining the underlying language model.

### 4.3 Architecture of the Proposed Method

The architecture consists of four interacting modules.

The evidence retrieval module provides factual grounding. It selects the most relevant passages for the current query and passes them to the generator. In fixed-context benchmark settings, this module is controlled to ensure that all compared systems use the same retrieval depth.

The answer generation module produces a candidate answer. Its prompt includes the user question, retrieved evidence, task-specific instructions, and selected memory guidance when memory is enabled.

The Reflexion and verification module evaluates the candidate answer. It identifies unsupported statements, missing evidence, contradiction risks, and false-premise risks. If the generated answer is not sufficiently grounded, the module creates reflection guidance for another generation attempt.

The temporal success-aware memory module stores previous reasoning episodes and ranks them for future reuse. Each memory contains the original question, the final answer or reflection summary, metadata, timestamp, and success/failure statistics.

Together, these modules form an adaptive RAG pipeline in which reasoning quality can improve over time through controlled reuse of previous successful episodes.

### 4.4 Memory Representation

Each memory entry represents one completed reasoning episode. The stored information includes:

```text
memory_id
question
reflection summary
final answer or outcome
benchmark name
task type
file name or case number when available
creation timestamp
last-used timestamp
success count
failure count
alpha
beta
hallucination risk
```

The memory does not function as a simple cache of final answers. Instead, it stores reusable reasoning guidance. This distinction is important because direct answer reuse may cause leakage or incorrect transfer, while reflection-based guidance can help the model avoid previously observed mistakes.

### 4.5 Memory Scoring Algorithm

For each new query, candidate memories are ranked using a composite score:

```text
memory_score =
    similarity
  * temporal_weight
  * reliability
  * risk_penalty
  * (1 + metadata_bonus)
```

The similarity term estimates lexical overlap between the current question and the memory question. In the implemented prototype, Jaccard token similarity is used because it is transparent, deterministic, and easy to audit.

The temporal term gives higher weight to recent memories:

```text
temporal_weight = exp(-age_days / decay_days)
```

The reliability term estimates how often the memory has been useful:

```text
reliability = alpha / (alpha + beta)
```

The hallucination-risk penalty suppresses memories associated with previous failures:

```text
risk_penalty = max(epsilon, 1 - lambda * hallucination_risk)
```

The metadata bonus increases the score when the memory belongs to the same file, case, or source context. This is especially useful for legal and document-grounded QA, where reusing reasoning from the same document is safer than reusing reasoning from an unrelated source.

### 4.6 Memory Selection and Filtering

After scores are computed, the system filters memories before inserting them into the prompt. A memory can be rejected if its score is too low, if it has high hallucination risk, if it lacks useful reflection text, or if it is incompatible with the current benchmark profile.

This conservative filtering is necessary because memory can produce negative transfer. A superficially similar memory may contain a reasoning pattern that was useful in one context but harmful in another. The proposed method therefore treats memory as a controlled source of guidance rather than as unconditional additional context.

For grounded tasks, stricter thresholds are used. This prevents unreliable memories from weakening evidence-based generation.

### 4.7 Reflexion Procedure

The Reflexion procedure improves answer quality through iterative generation and critique. The system first generates a candidate answer using the retrieved evidence and selected memories. Then the skeptic and critic components evaluate whether the answer is supported by the evidence.

If the answer contains unsupported claims, the system creates reflection guidance. This guidance may instruct the next iteration to focus on specific evidence, remove unsupported details, avoid false assumptions, or abstain when the evidence is insufficient.

The procedure can be summarized as:

```text
Input: question, retrieved evidence, selected memories
Output: final grounded answer

for iteration in 1..max_iters:
    generate candidate answer
    evaluate answer with skeptic/critic
    if answer is grounded and complete:
        stop
    else:
        create reflection guidance
        retry generation

save episode to memory
update reused memory statistics
return final answer
```

In the current implementation, `max_iters = 3` is used in the main temporal Reflexion experiments.

### 4.8 Memory Update Algorithm

After the final answer is produced, the system determines whether the episode was successful according to the benchmark profile and available evaluation signals. A new memory is then saved if memory saving is enabled.

For each memory reused during the episode, the system updates its statistics. If the episode is successful, the memory receives a positive update:

```text
alpha_delta = 1
beta_delta = 0
```

If the episode is unsuccessful, the memory receives a negative update:

```text
alpha_delta = 0
beta_delta = 1
```

These updates modify the reliability estimate used in future memory scoring. Over time, memories that consistently support successful answers become more influential, while memories associated with hallucination or failure become less likely to be reused.

### 4.9 Difference from Baseline Methods

Vanilla RAG performs retrieval and answer generation in a single pass. It does not critique its own answer and does not reuse previous reasoning episodes.

Self-refine or simple Agentic RAG adds a limited refinement step, but it does not maintain a persistent success-aware memory.

Simple Reflexion memory can reuse previous reflections, but it does not rank them using temporal decay, reliability, and hallucination-risk suppression.

Temporal Reflexion without memory uses the iterative verification architecture but disables active memory reuse. This baseline is important because it separates the effect of Reflexion from the effect of memory.

The proposed method combines Reflexion with temporal success-aware memory. Its main novelty is not merely storing previous answers, but selecting previous reasoning episodes according to a risk-aware and success-aware scoring function.

### 4.10 Implementation Details

The prototype is implemented as a Python-based RAG pipeline. The main system logic is implemented in `rag.py`. Memory entries are stored in JSONL format, while memory statistics are stored separately to support auditable updates. Benchmark-specific memory paths are used to reduce cross-benchmark contamination.

The system supports benchmark-aware profiles. For example, QReCC is treated as conversational QA, RAGTruth as hallucination-focused fixed-context evaluation, and legal QA as source-sensitive grounded answering. These profiles control whether memory is used, whether memory is saved, whether citations are required, and how conservative the system should be when selecting memory.

The main experimental configuration uses:

```text
top_k = 3
max_iters = 3
memory prior alpha = 1.0
memory prior beta = 1.0
temporal decay = 30 days
hallucination penalty = 0.6
```

### 4.11 Experimental Results

The main memory-oriented evaluation is performed on QReCC because conversational QA is suitable for testing reuse of previous reasoning episodes. The clean QReCC comparison produced the following results:

```text
Method                         Hallucination   Answer F1   Exact Match
Vanilla RAG                    0.58            0.3341      0.00
Self-refine AgentRAG           0.62            0.3236      0.00
Reflexion simple memory        0.26            0.4066      0.00
Temporal no memory             0.28            0.5238      0.02
Temporal success memory        0.18            0.5199      0.00
```

Compared with vanilla RAG, the proposed temporal success-aware memory system reduces detector-estimated hallucination from 0.58 to 0.18 and improves Answer F1 from 0.3341 to 0.5199. Compared with temporal Reflexion without memory, the proposed memory mechanism reduces hallucination from 0.28 to 0.18, while maintaining a similar Answer F1.

These results suggest that the Reflexion architecture improves answer quality, while success-aware memory can further reduce hallucination in the clean QReCC memory setting.

### 4.12 Analysis of Results

The results show that the largest improvement over vanilla RAG comes from adding iterative Reflexion and stricter grounding behavior. This is visible in the improvement from vanilla RAG to the temporal Reflexion baselines.

The comparison between temporal no-memory and temporal success memory is especially important. In the clean QReCC comparison, memory reduces hallucination while preserving answer quality. This supports the hypothesis that selected previous reasoning episodes can help the system avoid unsupported generation.

However, the results should be interpreted carefully. QReCC in the adapted setup does not provide gold hallucination labels, so hallucination values are detector-estimated. Therefore, the strongest claim should be that the proposed method reduces detector-estimated hallucination and improves grounded QA behavior, not that it definitively eliminates hallucination.

The RAGTruth experiments provide a complementary view. They are useful for hallucination-focused validation, but they are less suitable for proving memory usefulness because RAGTruth does not naturally require conversational or temporal reuse of prior reasoning episodes.

### 4.13 Practical Recommendations

The proposed method is most useful in settings where similar questions or reasoning patterns occur over time. Conversational QA, legal QA, technical support, and document-grounded assistants are suitable use cases.

Memory should be enabled conservatively. In high-risk grounded tasks, only memories with sufficient score, low hallucination risk, and compatible metadata should be inserted into the prompt.

For clean experimental evaluation, memory should be built only from earlier examples. Future test examples must not be stored before evaluation, otherwise the result may suffer from leakage.

The recommended experimental setup is:

```text
Build memory on an earlier QReCC split.
Evaluate on a later QReCC split.
Compare temporal no-memory with temporal success memory.
Report Answer F1, exact match, detector-estimated hallucination, and memory behavior metrics.
```

### 4.14 Limitations of the Proposed Method

The proposed method depends on the quality of the hallucination detector or critic. If the critic incorrectly marks a good answer as bad, useful memories may receive negative updates. If it fails to detect unsupported answers, unreliable memories may remain active.

The memory score currently uses lexical similarity. This makes the method transparent and reproducible, but it may miss semantically similar questions with different wording. Future work can compare lexical scoring with dense embedding-based memory retrieval.

The method also requires careful benchmark separation. If memory from test examples is reused during evaluation, the result may overestimate system performance. This is why sequential memory construction is necessary.

Finally, memory does not always improve every benchmark. In tasks where each example is independent, such as some fixed-context hallucination benchmarks, Reflexion may provide most of the benefit while memory contributes less.

### 4.15 Chapter Summary

This chapter presented ReflexionTemporalMemorySuccessAgentRAG, a proposed method for improving grounded question answering through iterative Reflexion and temporal success-aware episodic memory. The method ranks memories using similarity, temporal recency, reliability, hallucination-risk suppression, and metadata locality. It updates memory statistics after each episode, allowing useful memories to become more influential and harmful memories to be suppressed.

The experimental results on QReCC show that the proposed system improves answer quality and reduces detector-estimated hallucination compared with vanilla RAG and simple Reflexion baselines. The memory-specific comparison indicates that temporal success-aware memory can further reduce hallucination beyond temporal Reflexion alone under a clean sequential evaluation setup.
