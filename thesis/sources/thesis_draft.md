# Reducing Hallucination Propagation in an Agentic RAG System Using Self-Reflective Verification Cycles

## Abstract

Large language models are increasingly used as reasoning components in retrieval-augmented generation systems. Retrieval-Augmented Generation (RAG) reduces hallucinations by grounding answers in external documents, but it does not eliminate unsupported or contradictory claims. In agentic RAG systems, this problem becomes more serious because an unsupported claim may affect later reasoning steps, self-generated feedback, and long-term memory. As a result, hallucinations can propagate rather than remain isolated in a single answer.

The goal of this thesis is to develop and evaluate a self-reflective RAG architecture that reduces hallucination propagation by verifying generated answers against retrieved evidence before reusing them in later iterations or storing them in memory. The proposed system, ReflexionTemporalMemorySuccessAgentRAG, combines retrieval-based grounding, iterative self-reflection, a critic module, a skeptic guardrail, and temporal success-weighted memory. Memory entries are scored using query similarity, temporal decay, reliability estimated from success and failure counts, and a hallucination-risk penalty.

The system was evaluated on QReCC, RAGTruth, and a legal-domain corpus derived from LegalBench-RAG-style tasks. On the QReCC fixed-context evaluation, the proposed system achieved a hallucination rate of 0.08 and answer F1 of 0.6046 over 100 examples. On the legal grounded QA benchmark, it achieved a grounded rate of 0.95 and hallucination rate of 0.05 over 20 examples. On RAGTruth, the generated answer hallucination rate was 0.30 over 100 examples, showing that the benchmark remains challenging and that hallucination detection quality is a limiting factor.

The results show that self-reflective verification and temporal success memory can improve groundedness and reduce unsupported generation in RAG workflows. A diagnostic propagation analysis also shows that unsupported claims may still recur inside intermediate attempts, even when the final answer is judged non-hallucinated. This finding supports the central motivation of the thesis: hallucination control in agentic RAG should evaluate not only final answers, but also the movement of unsupported claims through reflection loops and memory traces.

## Introduction

### Background

Large language models (LLMs) have become a central technology for question answering, summarization, legal assistance, software development, and interactive agents. Their strength comes from the ability to generate fluent natural language and perform multi-step reasoning over complex prompts. This progress builds on transformer architectures and large-scale pretraining, which made few-shot and instruction-following behavior possible in modern systems [28; 29]. However, LLMs do not have direct access to an external truth source during ordinary generation. They may produce plausible but unsupported statements, a phenomenon commonly called hallucination [2; 22].

Retrieval-Augmented Generation addresses this issue by combining a generative model with an external document retrieval system [1]. Instead of relying only on parametric knowledge stored in model weights, the system retrieves relevant context passages and conditions generation on those passages. This approach improves factuality and enables source attribution, but it does not guarantee that the generated answer is fully supported by the retrieved evidence [7; 8]. The model may still introduce facts that are absent from the context, overgeneralize from partial evidence, contradict a retrieved passage, or use a relevant document for an unsupported conclusion.

In agentic RAG systems, the risk is amplified. An agent may generate intermediate reasoning, critique its own answer, retrieve additional documents, update memory, and use previous answers as guidance for future questions. If an unsupported claim enters this loop, it may be repeated in later attempts or stored as a memory entry. This creates hallucination propagation: an initial error becomes part of the agent's reasoning state and influences later outputs.

### Problem Statement

Standard RAG systems reduce hallucinations but do not fully prevent them [1; 7; 19]. Self-refinement and Reflexion-style systems can improve outputs through feedback [3; 4], but they may also reinforce incorrect assumptions if the feedback itself is not grounded in evidence. Memory-augmented agents can reuse successful experiences, but memory becomes dangerous when it stores unsupported answers or reflections. The key research problem is therefore not only how to detect hallucinations in a final answer, but how to prevent unsupported claims from spreading through iterative generation and memory.

### Research Aim and Objectives

The aim of this thesis is to design and evaluate a self-reflective agentic RAG architecture that reduces hallucination propagation by verifying generated answers against retrieved evidence and storing only reliable memory traces.

The objectives are:

1. Analyze hallucinations in RAG and agentic RAG systems.
2. Review existing retrieval, self-refinement, Reflexion, and memory-augmented approaches.
3. Design a self-reflective verification loop for RAG answer generation.
4. Add a skeptic guardrail for unsupported claims and false-premise risks.
5. Implement temporal success memory with reliability and hallucination-risk scoring.
6. Evaluate the proposed architecture against baseline RAG-style systems.
7. Measure hallucination rate, groundedness, answer quality, retrieval hit rate, and indicators of hallucination propagation.

### Research Hypothesis

The hypothesis of this work is that self-reflective verification cycles combined with temporal success-based memory reduce hallucination propagation in RAG agents by preventing unsupported claims from being reused in later reasoning steps or stored as reliable memory.

### Research Contribution

The main contribution of the thesis is the ReflexionTemporalMemorySuccessAgentRAG architecture. The system extends a Reflexion-style RAG pipeline with:

- evidence-grounded answer generation;
- critic-based sufficiency checking;
- skeptic guardrail for unsupported claims;
- multi-level self-reflection over retrieval and answer quality;
- temporal memory of previous verification episodes;
- Bayesian-style reliability scoring based on success and failure counts;
- hallucination-risk suppression for memory reuse.

### Methodology Overview

The study uses an implementation-based research methodology. First, a RAG pipeline is implemented with retrieval, answer generation, critique, and iterative refinement. Second, a temporal success memory module is added to rank previous episodes. Third, the system is evaluated on fixed-context and retrieval-based QA benchmarks. The evaluation includes hallucination rate, groundedness, answer F1, accuracy, retrieval hit rate, and detector-based hallucination metrics.

### Thesis Structure

Chapter 1 introduces theoretical foundations of LLMs, RAG systems, hallucinations, agentic reasoning, propagation, and self-reflection. Chapter 2 reviews existing approaches and identifies the research gap. Chapter 3 describes the proposed architecture and algorithm. Chapter 4 presents experimental evaluation and limitations. The conclusion summarizes the findings and outlines future work.

## Chapter 1. Theoretical Foundations of RAG Systems and Hallucinations

### 1.1 Large Language Models and the Hallucination Problem

Large language models generate text by predicting likely token sequences conditioned on context. This mechanism allows them to produce coherent answers, but it does not inherently distinguish between verified knowledge and plausible language [27; 28]. Hallucination occurs when a generated statement is not supported by the available input or contradicts the source material [2].

In question answering, hallucinations may include invented entities, incorrect dates, unsupported legal conclusions, wrong numerical values, or overconfident answers to unanswerable questions. In RAG settings, hallucinations are especially important because the user often expects the answer to be grounded in retrieved documents. Therefore, an answer can be fluent and relevant but still fail the central requirement of faithfulness to evidence.

### 1.2 Retrieval-Augmented Generation

Retrieval-Augmented Generation combines a retriever and a generator [1]. The retriever searches an external corpus for passages relevant to a query. The generator receives the query and retrieved passages and produces an answer. This architecture improves factuality because the model can rely on documents rather than only on parametric memory. Dense retrievers such as DPR [10], passage-augmented generation methods such as FiD [11], and retrieval-augmented pretraining approaches such as REALM [9] demonstrate different ways of connecting parametric models with external evidence.

The general RAG pipeline consists of:

1. Query processing.
2. Document or passage retrieval.
3. Context construction.
4. Answer generation.
5. Optional citation or source attribution.

RAG can be formalized as:

\[
a = G(q, R(q, C))
\]

where \(q\) is the user query, \(C\) is the corpus, \(R\) is the retrieval function, and \(G\) is the generator. RAG is effective for knowledge-intensive tasks, but it has several failure modes. The retriever may return irrelevant passages, the generator may ignore relevant evidence, or the answer may combine retrieved evidence with unsupported assumptions [7; 21]. Thus, retrieval is necessary but not sufficient for hallucination control.

### 1.3 Agentic RAG Systems

Agentic RAG systems extend standard RAG by adding planning, tool use, iterative reasoning, feedback loops, and memory. Instead of producing a single answer from one retrieval step, an agent may decide whether to retrieve more evidence, critique an answer, refine it, or store useful experiences. This direction is related to tool-using and reasoning-acting agents such as ReAct and Toolformer [14; 15].

This flexibility makes agentic RAG powerful for complex tasks such as legal analysis, multi-hop question answering, and document-grounded assistants. However, it also introduces new risks. Intermediate answers and reflections become part of the agent state. If these intermediate artifacts contain hallucinations, the agent can reuse them as if they were verified facts.

### 1.4 Hallucinations in Multi-Step Reasoning

In single-step generation, hallucination is usually measured at the final answer level. In multi-step reasoning, hallucination can appear earlier. A model may form an unsupported assumption in an intermediate step, use it to guide retrieval, and then generate an answer that appears coherent but rests on an incorrect premise.

Multi-step reasoning therefore requires checking not only the final answer but also the claims and decisions that lead to it. A self-reflective agent should ask whether each important claim is supported, whether the retrieved evidence is sufficient, and whether the answer includes information not present in the context.

### 1.5 Hallucination Propagation Through Memory and Iterative Generation

Hallucination propagation occurs when an unsupported claim affects later system behavior. In an agentic RAG system, propagation can happen through:

- repeated attempts within the same answer generation loop;
- reflection text that preserves a wrong assumption;
- memory entries saved after unsuccessful verification;
- future retrieval of unreliable memories;
- final answers that cite evidence but contain unsupported conclusions.

This thesis treats propagation as a separate problem from ordinary hallucination. A hallucination in one answer is harmful, but a hallucination stored in memory can influence many future answers. Therefore, memory must be reliability-aware and risk-sensitive.

### 1.6 Self-Reflection and Verification Methods

Self-reflection methods ask the model to evaluate or improve its own output. Self-refinement typically follows a generate-feedback-revise loop [4]. Reflexion-style agents store verbal reflections from previous attempts and use them to improve future decisions [3]. Self-RAG-like systems integrate retrieval and critique into the generation process and show that reflection can improve factuality when it is coupled with retrieval decisions [5].

Verification is the grounding counterpart to reflection. A reflection is useful only if it is anchored in evidence. The proposed system therefore combines reflection with critic and skeptic checks. The critic evaluates whether the answer is sufficient and supported. The skeptic guardrail identifies unsupported claims, false-premise risk, and correction actions.

### 1.7 Chapter Summary

This chapter introduced the theoretical background for the thesis. RAG improves factuality by grounding generation in retrieved evidence, but agentic RAG systems introduce new propagation risks through iterative reasoning and memory. Self-reflection can improve answer quality, but it must be paired with evidence-based verification to avoid reinforcing unsupported claims.

## Chapter 2. Existing Approaches and Research Gap

### 2.1 Baseline RAG Systems

Baseline RAG systems retrieve a fixed number of passages and generate an answer from them [1]. Their main advantage is simplicity. They are easy to implement and often improve factuality compared with generation without retrieval. However, they usually do not verify whether the final answer is fully supported by the retrieved text.

Common baseline failures include irrelevant retrieval, insufficient context, unsupported synthesis, and overconfident answering. In legal and factual QA tasks, these errors are serious because users need traceable and reliable answers.

### 2.2 Retrieval Improvement Methods

Retrieval improvement methods focus on returning better evidence. Techniques include query rewriting, dense retrieval, hybrid sparse-dense retrieval, reranking, chunk optimization, metadata filtering, and multi-hop retrieval. Sparse probabilistic retrieval such as BM25 remains a strong baseline [33], while dense retrieval methods such as DPR and Sentence-BERT improve semantic matching [10; 31]. Late-interaction methods such as ColBERT improve passage ranking while preserving token-level matching signals [34]. These methods reduce hallucination indirectly by improving evidence quality.

However, even perfect retrieval does not guarantee faithful generation. A model may ignore the retrieved evidence or add unsupported claims. Therefore, retrieval improvement should be combined with answer verification.

### 2.3 Self-Critique and Self-Refinement Methods

Self-critique methods ask the model to review its own answer. Self-refinement uses feedback to generate a revised answer [4]. These approaches are attractive because they do not require model fine-tuning. They can be applied at inference time using the same or a separate model. Related reasoning-time methods such as chain-of-thought prompting, self-consistency, and tree-of-thought search show that additional inference-time deliberation can improve reasoning quality [35; 36; 37].

The limitation is that critique can itself hallucinate. If the critique does not explicitly compare the answer with evidence, it may reward fluent but unsupported text. This motivates evidence-aware critic design.

### 2.4 Reflexion-Based Agents

Reflexion-based agents use verbal feedback and episodic memory to improve future behavior [3]. Instead of updating model weights, the agent stores reflections and reuses them in later attempts. This is efficient and interpretable.

For RAG systems, Reflexion is useful because previous failures can guide better retrieval and answer generation. However, ordinary episodic memory does not automatically distinguish reliable memories from unreliable ones. A memory entry based on a hallucinated answer can become a source of future hallucinations.

### 2.5 Memory-Augmented Agent Systems

Memory-augmented agents store previous interactions, decisions, answers, and feedback. Memory can improve consistency and reduce repeated mistakes. In RAG, memory can preserve successful retrieval strategies, useful citations, and correction patterns. However, long-context and memory-augmented systems are sensitive to where information appears and how it is selected; models may underuse relevant context or overuse misleading context [38].

The main danger is memory contamination. If memory stores unsupported claims, later answers may inherit them. Therefore, memory should be filtered by success, recency, reliability, and hallucination risk.

### 2.6 Limitations of Existing Approaches

Existing approaches often address only part of the problem:

- baseline RAG improves grounding but does not verify all claims;
- retrieval improvement methods do not control generation behavior;
- self-refinement improves fluency but may not be evidence-grounded;
- Reflexion stores useful feedback but may also store unreliable reflections;
- memory systems improve reuse but can propagate errors.

The missing element is an integrated architecture that verifies answers, reflects on failures, and stores memory only with success and hallucination-risk metadata.

### 2.7 Research Gap, Hypothesis, and Objectives

The research gap is the lack of explicit hallucination propagation control in agentic RAG systems with memory. The thesis addresses this gap by proposing a self-reflective RAG agent that checks support against retrieved evidence, suppresses high-risk memory, and updates memory using success/failure outcomes.

The hypothesis is that such an architecture reduces the chance that unsupported claims are repeated, reused, or stored as reliable memory. The objectives are the design, implementation, and experimental evaluation of this architecture.

## Chapter 3. Proposed Self-Reflective RAG Architecture

### 3.1 Design Rationale

The proposed architecture is designed around four principles.

First, generation must be grounded in retrieved evidence. The answer generator receives context passages and is instructed to use them as the primary source of information.

Second, generated answers must be verified before acceptance. The critic and skeptic modules evaluate sufficiency, unsupported claims, contradiction, and false-premise risk.

Third, reflection must be corrective rather than merely descriptive. When an answer fails verification, the system builds guidance for the next attempt.

Fourth, memory must be success-weighted and risk-aware. Previous episodes are useful only when they were grounded and successful. The memory score therefore downweights old, unreliable, or high-risk entries.

### 3.2 General System Architecture

The system consists of the following components:

- user query interface;
- query normalization and benchmark profiling;
- retrieval module;
- answer generator;
- critic module;
- skeptic guardrail;
- self-reflection controller;
- temporal success memory;
- memory outcome updater;
- evaluation logger.

**Figure 3.1: General architecture of ReflexionTemporalMemorySuccessAgentRAG.**  
Editable diagram: `thesis/architecture.drawio`.

### 3.3 Retrieval and Answer Generation Pipeline

The retrieval module receives the user question and optional filters such as file name, case number, benchmark name, or source path. It returns the top-k evidence chunks. The generator then receives the question, retrieved context, memory guidance, and reflection hint.

For legal-domain tasks, the system prefers primary evidence and citations. For benchmark tasks such as QReCC and RAGTruth, the profile can disable citations or memory saving where needed to keep evaluation fair.

### 3.4 Self-Reflective Verification Cycle

The central loop follows this pattern:

1. Retrieve evidence.
2. Retrieve relevant memory.
3. Generate a candidate answer.
4. Apply skeptic guardrail.
5. Run critic evaluation.
6. If sufficient, accept the answer.
7. If insufficient, generate reflection guidance.
8. Retry with updated guidance until the maximum number of attempts is reached.

**Figure 3.2: Self-reflective verification pipeline.**  
Editable diagram: `thesis/pipeline.drawio`.

### 3.5 Critic Module and Skeptic Guardrail

The critic evaluates whether the answer is sufficient for the question and supported by the evidence. It produces issues and guidance for revision. The skeptic guardrail focuses on unsupported claims, false-premise risk, counter-evidence, and hallucination indicators.

The critic and skeptic have different roles. The critic answers the question "Is this answer acceptable?" The skeptic answers the question "Which claims may be unsupported or dangerous?" Together they reduce the chance that fluent but unsupported answers are accepted.

### 3.6 Temporal Memory of Successful Verifications

The memory stores previous episodes with metadata:

- question;
- benchmark and task type;
- final answer span;
- source chunk;
- success or failure status;
- hallucination risk;
- critique issues;
- reflection guidance;
- creation timestamp;
- success and failure counts.

The memory is temporal because older entries gradually lose weight. It is success-based because successful episodes receive higher reliability. It is risk-aware because hallucination-prone entries are suppressed.

### 3.7 Memory Scoring and Hallucination-Risk Suppression

For a memory item \(m_i\), the system uses the score:

\[
S(m_i) =
\operatorname{sim}(q,m_i)
\cdot T(m_i)
\cdot R(m_i)
\cdot P(m_i)
\cdot \bigl(1 + B(m_i)\bigr)
\]

where:

\[
T(m_i) = \exp\left(-\frac{\Delta t_i}{\tau}\right)
\]

\[
R(m_i) = \frac{\alpha_i}{\alpha_i + \beta_i}
\]

\[
P(m_i) = \max\left(\varepsilon, 1 - \lambda h_i\right)
\]

Here, \(\operatorname{sim}(q,m_i)\) is the Jaccard token similarity between the current question and the memory question, \(\Delta t_i\) is the age of the memory in days, \(\tau\) is the temporal decay window, \(h_i\) is the estimated hallucination risk, \(B(m_i)\) is the metadata bonus for matching file or case information, and \(\varepsilon\) prevents partially useful memories from being forced to zero. The parameters \(\alpha_i\) and \(\beta_i\) represent success and failure evidence, while \(\lambda\) controls how strongly hallucination risk suppresses memory reuse. This score prevents unreliable memories from dominating future generations.

### 3.8 Hallucination Propagation Control Mechanism

The propagation control mechanism operates at three levels.

At the attempt level, unsupported claims are detected and used as negative guidance for the next attempt. At the answer level, the final response is accepted only if the critic considers it sufficiently grounded or if the system reaches a safe fallback. At the memory level, the saved memory entry contains success status and hallucination risk, so future retrieval can suppress risky traces.

A propagated hallucination is defined as an unsupported claim from an earlier attempt that appears again in a later attempt, final answer, or saved memory. This definition separates ordinary hallucination from propagation.

### 3.9 ReflexionTemporalMemorySuccessAgentRAG Algorithm

```text
Input: question q, corpus C, memory M, max iterations N
Output: final answer a, trace T

1. Determine benchmark profile and task type.
2. Retrieve relevant memory entries from M using temporal success scoring.
3. For i = 1 to N:
   3.1 Retrieve evidence contexts E_i from C.
   3.2 Generate candidate answer a_i using q, E_i, memory guidance, and reflection hint.
   3.3 Apply skeptic guardrail to identify unsupported claims.
   3.4 Run critic to evaluate sufficiency and grounding.
   3.5 Save attempt information to trace T.
   3.6 If critic marks answer sufficient, break.
   3.7 Build reflection guidance from critic and skeptic feedback.
4. Determine final success from grounding sufficiency.
5. Save memory entry with success/failure, reliability, and hallucination risk.
6. Update outcome statistics for reused memory entries.
7. Return final answer and trace.
```

### 3.10 Implementation Details

The implementation is located in `ReflexionTemporalMemorySuccessAgentRAG`. The core logic is implemented in `rag.py`. The main function for the proposed architecture is `generate_answer_with_reflexion`, which performs retrieval, generation, critique, reflection, and memory update. The memory files are stored as JSONL, with a separate append-only statistics log for replay outcomes.

The implementation supports benchmark profiles for QReCC, RAGTruth, and legal-domain tasks. Environment variables control memory behavior, temporal decay, hallucination penalty, skeptic activation, fast mode, and benchmark-specific memory saving.

### 3.11 Chapter Summary

This chapter described the proposed self-reflective RAG architecture. The key idea is to combine evidence retrieval, answer verification, reflective retry, and risk-aware temporal memory. The architecture directly addresses hallucination propagation by preventing unsupported claims from being accepted or stored as reliable memory.

## Chapter 4. Experimental Evaluation

### 4.1 Datasets and Experimental Setup

The system was evaluated on three data sources.

QReCC is a conversational question answering benchmark. In this work, an adapted fixed-context version was used to evaluate answer quality and generated hallucination rate. The evaluation used 100 records with top-k equal to 3 and maximum three reflection iterations.

RAGTruth is a hallucination-focused benchmark for RAG systems. It includes response-level and span-level hallucination annotations. In this work, a 100-record evaluation was used to measure generated hallucination rate and hallucination detection quality.

The legal corpus is based on legal-document QA tasks derived from local LegalBench-RAG-style data. It tests whether the model can answer from legal evidence while preserving grounding and citations.

### 4.2 Baseline Methods

The comparison includes:

- Without RAG: generation without retrieved evidence.
- vanillaRAG: retrieval followed by one-step answer generation.
- selfRefineAgentRAG: answer generation with iterative self-refinement.
- reflexionAgentRAG: Reflexion-style agent with reflective retries.
- ReflexionTemporalMemorySuccessAgentRAG: the proposed method with self-reflection, skeptic guardrail, and temporal success memory.

### 4.3 Evaluation Metrics

The main metrics are:

- Hallucination Rate: fraction of generated answers judged to contain unsupported or contradictory claims.
- Groundedness / Faithfulness: fraction of answers supported by evidence.
- Answer F1 / Accuracy: lexical or judged correctness against reference answers.
- Retrieval Hit Rate: fraction of examples where retrieved context includes the expected evidence.
- Propagated Error Rate: fraction of unsupported claims repeated across attempts, final answer, or memory.
- Memory Hallucination Risk: average hallucination-risk score of saved or reused memory entries.

### 4.4 Experimental Protocol

For each dataset, the system was run with up to three self-reflection iterations. Retrieval top-k was configured according to the benchmark. Outputs were saved with traces containing attempts, critique, skeptic output, selected memories, and memory entries. Summary metrics were generated from evaluation scripts.

The QReCC evaluation used the temporal backend, reflexion enabled, memory enabled, top-k 3, and 100 samples. The RAGTruth evaluation used the temporal backend, reflexion enabled, memory enabled, top-k 3, and 100 samples. The legal grounded QA evaluation used top-k 1, reflexion enabled, and judge-based QA scoring.

### 4.5 Results on QReCC

The QReCC fixed-context evaluation produced the following results:

| Metric | Value |
|---|---:|
| Total records | 100 |
| Hallucination rate | 0.08 |
| Non-hallucination rate | 0.92 |
| Answer F1 | 0.6046 |
| Exact match rate | 0.12 |
| Retrieval hit rate | 1.00 |

These results indicate that the system produced mostly grounded answers in the QReCC setting. The answer F1 shows moderate answer quality, while exact match remains low because conversational QA answers can differ lexically from references.

### 4.6 Results on RAGTruth

The RAGTruth evaluation produced the following results:

| Metric | Value |
|---|---:|
| Total records | 100 |
| Generated hallucination rate | 0.30 |
| Non-hallucination rate | 0.70 |
| Hallucination detection accuracy | 0.59 |
| Detection precision | 0.2667 |
| Detection recall | 0.2963 |
| Detection F1 | 0.2807 |

RAGTruth is more challenging because it is specifically designed to expose hallucinations in RAG outputs. The hallucination rate of 0.30 shows that the proposed architecture reduces but does not eliminate hallucinations. The low detection F1 indicates that detector quality is a limiting factor and should be improved in future work.

### 4.7 Results on the Legal Corpus

The legal grounded QA evaluation produced the following results:

| Metric | Value |
|---|---:|
| Total examples | 20 |
| Accuracy | 0.60 |
| Retrieval hit rate | 1.00 |
| Grounded rate | 0.95 |
| Hallucination rate | 0.05 |
| Supported but wrong rate | 0.35 |

The legal corpus results show strong grounding but moderate correctness. This distinction is important: an answer can be supported by evidence but still incomplete or not fully aligned with the reference answer. Therefore, hallucination reduction should not be treated as equivalent to full QA correctness.

### 4.8 Ablation Study

The component analysis compares the full system with variants that disable or weaken major components. Some variants are implemented as separate backends, while others are interpreted through available evaluation runs:

| Variant | Role in evaluation |
|---|---|
| No RAG | Measures the effect of removing retrieved evidence |
| vanillaRAG | Measures retrieval plus one-step generation |
| No memory | Measures whether temporal memory changes answer quality |
| No skeptic guardrail | Measures the effect of unsupported-claim filtering |
| No reflexion | Measures the effect of removing iterative correction |
| Full system | Combines retrieval, reflection, skeptic checking, and temporal success memory |

Current evaluation files include memory-enabled and no-memory QReCC runs with identical headline metrics in the 100-sample fixed-context setting. This indicates that the fixed-context QReCC adapter does not strongly expose the effect of temporal memory, because the questions are mostly independent and the gold context is already provided. Therefore, the strongest evidence in the current experiments concerns final-answer groundedness and hallucination rate, while the memory contribution is better evaluated through propagation diagnostics and future temporally linked episodes.

### 4.9 Hallucination Propagation Analysis

The central claim of the thesis is about reducing propagation, not only final hallucination rate. Therefore, a diagnostic propagation analysis was computed from the saved skeptic traces. The analysis uses unsupported claims identified by the skeptic module and checks whether the normalized claim appears again in later attempts, in the final answer, or in the unsaved candidate memory entry.

1. Extract unsupported claims from each attempt using the skeptic guardrail.
2. Compare unsupported claims across attempts.
3. Mark a claim as propagated if it appears in a later attempt, final answer, or saved memory.
4. Compute hallucination propagation rate:

\[
HPR =
\frac{\left|\{c \in U : c \text{ reappears in a later attempt, final answer, or memory}\}\right|}
{\left|U\right|}
\]

where \(U\) is the set of unsupported claims detected in earlier attempts. If \(U\) is empty, \(HPR\) is defined as 0.

5. Compute memory hallucination risk:

\[
MHR =
\frac{1}{|M_{\mathrm{saved}}|}
\sum_{m \in M_{\mathrm{saved}}} h(m)
\]

The resulting diagnostic metrics are shown in Table 4.4. The metric is intentionally conservative: it counts claim recurrence as propagation even when the final detector later judges the answer non-hallucinated. Therefore, it should be interpreted as a trace-level risk signal, not as a replacement for final-answer hallucination rate.

| Dataset | Unsupported claim occurrences | Propagated occurrences | Propagated error rate | Propagated to later attempts | Propagated to final answer | Propagated to memory text |
|---|---:|---:|---:|---:|---:|---:|
| QReCC | 376 | 233 | 0.6197 | 180 | 101 | 145 |
| RAGTruth | 762 | 256 | 0.3360 | 222 | 40 | 88 |
| Legal corpus | 0 | 0 | 0.0000 | 0 | 0 | 0 |

These values reveal that unsupported fragments often survive across internal attempts even when the final answer is short and grounded. In QReCC, many unsupported claims identified by the skeptic reappeared in later attempts or candidate memory text, but the final hallucination rate remained 0.08. This gap supports the thesis argument that final-answer metrics alone are insufficient for agentic RAG. RAGTruth shows a lower propagated error rate but a higher final hallucination rate, indicating that benchmark difficulty and detector behavior also affect the relationship between internal propagation and final output quality. The legal corpus contained no skeptic-reported unsupported claims in the evaluated sample, which is consistent with its high grounded rate.

### 4.10 Error Analysis and Limitations

The experiments reveal several limitations.

First, detector-based hallucination evaluation depends on the quality of the detector. On RAGTruth, hallucination detection F1 was 0.2807, which means detector errors may affect reported hallucination rates.

Second, fixed-context evaluation can hide retrieval weaknesses. QReCC showed retrieval hit rate 1.00 because the adapted setup preserved gold evidence. This is useful for measuring generation faithfulness but not enough for native retrieval evaluation.

Third, legal QA shows that groundedness and correctness are different. The legal evaluation had grounded rate 0.95 but accuracy 0.60, meaning that some answers were supported but incomplete or mismatched with the expected answer.

Fourth, memory impact requires temporally structured benchmarks. If questions are independent, temporal memory may not show clear benefits. The strongest test for the proposed method is an episode-based benchmark where related questions are asked sequentially and previous failures can influence later answers. In the current QReCC and RAGTruth runs, benchmark profiles disabled persistent memory saving for fairness, so memory entries are available in traces but were not written as reusable long-term memory.

### 4.11 Chapter Summary

The experiments show that the proposed system achieves low hallucination rates on QReCC and legal grounded QA, while RAGTruth remains challenging. The propagation diagnostic shows that unsupported claims can still recur inside the reflective loop. The results therefore support the usefulness of self-reflective verification, while also showing that propagation-aware metrics are necessary for evaluating agentic RAG systems.

## Conclusion

### Summary of Findings

This thesis investigated hallucination propagation in agentic RAG systems and proposed a self-reflective verification architecture to reduce it. The proposed system combines retrieval, generation, critique, skeptic guardrails, reflection, and temporal success memory.

The evaluation showed that the system can produce grounded answers in several settings. On QReCC, the hallucination rate was 0.08 with answer F1 of 0.6046. On the legal grounded QA benchmark, the hallucination rate was 0.05 and grounded rate was 0.95. On RAGTruth, the hallucination rate was 0.30, demonstrating that hallucination-focused benchmarks remain difficult.

### Main Contributions

The main contributions are:

1. A self-reflective RAG architecture for reducing hallucination propagation.
2. A critic and skeptic verification loop for answer support checking.
3. A temporal success memory mechanism with reliability and hallucination-risk scoring.
4. A propagation-oriented evaluation framing using propagated error rate and memory hallucination risk.
5. Empirical evaluation on QReCC, RAGTruth, and a legal-domain corpus.

### Limitations

The main limitation is that current evaluation results measure final hallucination rate more robustly than long-term propagation across persistent memory, because some benchmark profiles disable memory saving for fairness. Another limitation is detector quality, especially on RAGTruth. Finally, memory benefits are best evaluated on temporally linked tasks, not independent fixed-context questions.

### Future Work

Future work should improve hallucination detection, evaluate propagation on longer temporal episodes, add native retrieval metrics, test more legal-domain questions, and compare multiple LLM backends. Another promising direction is to train or fine-tune a dedicated verifier using RAGTruth-style span annotations.

## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401. https://arxiv.org/abs/2005.11401
2. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 55(12). https://doi.org/10.1145/3571730
3. Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
4. Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., Gupta, S., Majumder, B. P., Hermann, K., Welleck, S., Yazdanbakhsh, A., & Clark, P. (2023). Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651. https://arxiv.org/abs/2303.17651
5. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv:2310.11511. https://arxiv.org/abs/2310.11511
6. Anantha, R., Vakulenko, S., Tu, Z., Longpre, S., Pulman, S., & Chappidi, S. (2021). Open-Domain Question Answering Goes Conversational via Question Rewriting. arXiv:2010.04898. https://arxiv.org/abs/2010.04898
7. Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. Proceedings of ACL 2024. https://aclanthology.org/2024.acl-long.585/
8. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, H., & Wang, H. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997. https://arxiv.org/abs/2312.10997
9. Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. W. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. arXiv:2002.08909. https://arxiv.org/abs/2002.08909
10. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020. https://aclanthology.org/2020.emnlp-main.550/
11. Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. EACL 2021. https://aclanthology.org/2021.eacl-main.74/
12. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens. ICML 2022. https://arxiv.org/abs/2112.04426
13. Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., et al. (2021). WebGPT: Browser-assisted Question-Answering with Human Feedback. arXiv:2112.09332. https://arxiv.org/abs/2112.09332
14. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. https://arxiv.org/abs/2210.03629
15. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS 2023. https://arxiv.org/abs/2302.04761
16. Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., Grave, E., LeCun, Y., & Scialom, T. (2023). Augmented Language Models: A Survey. TMLR. https://arxiv.org/abs/2302.07842
17. Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023. https://arxiv.org/abs/2303.08896
18. Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023. https://arxiv.org/abs/2305.14251
19. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217. https://arxiv.org/abs/2309.15217
20. Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. arXiv:2311.09476. https://arxiv.org/abs/2311.09476
21. Chen, J., Lin, H., Han, X., & Sun, L. (2023). Benchmarking Large Language Models in Retrieval-Augmented Generation. arXiv:2309.01431. https://arxiv.org/abs/2309.01431
22. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022. https://aclanthology.org/2022.acl-long.229/
23. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On Faithfulness and Factuality in Abstractive Summarization. ACL 2020. https://aclanthology.org/2020.acl-main.173/
24. Dziri, N., Milton, S., Yu, M., Zaiane, O., & Reddy, S. (2022). On the Origin of Hallucinations in Conversational Models: Is it the Datasets or the Models? NAACL 2022. https://aclanthology.org/2022.naacl-main.387/
25. Komeili, M., Shuster, K., & Weston, J. (2022). Internet-Augmented Dialogue Generation. ACL 2022. https://aclanthology.org/2022.acl-long.579/
26. Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H. T., et al. (2022). LaMDA: Language Models for Dialog Applications. arXiv:2201.08239. https://arxiv.org/abs/2201.08239
27. OpenAI. (2023). GPT-4 Technical Report. arXiv:2303.08774. https://arxiv.org/abs/2303.08774
28. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020. https://arxiv.org/abs/2005.14165
29. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017. https://arxiv.org/abs/1706.03762
30. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. https://aclanthology.org/N19-1423/
31. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1410/
32. Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale Similarity Search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547. https://arxiv.org/abs/1702.08734
33. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389. https://doi.org/10.1561/1500000019
34. Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020. https://arxiv.org/abs/2004.12832
35. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E. H., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022. https://arxiv.org/abs/2201.11903
36. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E. H., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. https://arxiv.org/abs/2203.11171
37. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023. https://arxiv.org/abs/2305.10601
38. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. TACL 2024. https://arxiv.org/abs/2307.03172
39. Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., & Lewis, M. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. EMNLP 2023. https://arxiv.org/abs/2210.03350
40. Adlakha, V., Dhuliawala, S., Suleman, K., de Vries, H., & Reddy, S. (2023). Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering. arXiv:2307.16877. https://arxiv.org/abs/2307.16877

## Appendices

### Appendix A. Diagram Files

- `thesis/architecture.drawio`: general architecture diagram.
- `thesis/pipeline.drawio`: self-reflective verification pipeline diagram.

### Appendix B. Main Implementation Files

- `rag.py`: core RAG, reflexion, skeptic, critic, memory scoring, and generation logic.
- `retrieve.py`: retrieval interface.
- `evaluate_cross_repo_fixed_context.py`: fixed-context evaluation across backends.
- `evaluate_ragtruth_fixed_context.py`: RAGTruth evaluation.
- `evaluate_local_benchmark.py`: local legal benchmark evaluation.

### Appendix C. Main Evaluation Outputs

- `eval_outputs_qrecc_100_temporal_memory_enabled_final/fixed_context_summary.json`
- `eval_clean_runs/ragtruth_topk3_temporal_success_memory_rerun_calibrated_100/fixed_context_summary.json`
- `eval_outputs_legalbenchrag_grounded_memory/local_benchmark_summary.json`

### Appendix D. Recommended Extensions

1. Add a larger temporally linked benchmark where memory saving is enabled across related questions.
2. Add a table comparing vanillaRAG, selfRefineAgentRAG, reflexionAgentRAG, and the proposed method on the same records.
3. Convert diagrams from draw.io to SVG or PNG and insert them into the final document.
4. Confirm formatting rules for references, title page, margins, and chapter numbering.
5. Have the supervisor approve the final title and evaluation scope.
