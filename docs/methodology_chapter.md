# Chapter 3. Methodology

## 3.1 Research Design

This research investigates whether temporal success-aware episodic memory can improve the performance of a Reflexion-based retrieval-augmented generation system. The methodological objective is not limited to answer accuracy alone. It also includes groundedness, hallucination reduction, robustness under iterative reasoning, and the selective reuse of prior reasoning episodes. The proposed method extends a standard retrieval-augmented generation pipeline with two additional mechanisms: an iterative verification and revision loop, and an episodic memory controller that reuses past reasoning traces only when they are relevant, recent, reliable, and low-risk.

The study follows an experimental systems methodology. A complete working architecture is designed, implemented, and evaluated under controlled benchmark conditions. Across the main comparisons, the corpus, retrieval depth, language model setting, and decoding configuration are held constant unless a component is deliberately removed or altered for ablation. This design makes it possible to isolate the contribution of temporal success-aware memory from unrelated implementation factors.

The main research question is whether temporal success-aware episodic memory improves grounded answer quality and reduces hallucination when compared with vanilla retrieval-augmented generation, Reflexion-based generation without memory, and simpler memory reuse baselines.

## 3.2 Overall System Architecture

The proposed architecture consists of four tightly connected layers:

1. Evidence retrieval layer.
2. Answer generation layer.
3. Verification and Reflexion layer.
4. Temporal success-aware episodic memory layer.

At inference time, the system receives a question and optional source-related metadata. The retrieval layer identifies relevant evidence passages. The episodic memory layer selects past reasoning episodes that may be useful for the current query. The answer generation layer produces a candidate response conditioned on the evidence and selected memory guidance. The verification and Reflexion layer then evaluates whether the answer is sufficiently grounded. If the answer is weak or partially unsupported, the system generates guidance and performs another reasoning step. After the episode ends, the final outcome is recorded so that future memory selection can take past success and failure into account.

This architecture differs from conventional single-pass retrieval-augmented generation in two ways. First, it treats answer generation as an iterative reasoning process. Second, it treats prior reasoning episodes as selective knowledge sources rather than universally reusable prompts.

Figure 3.1 presents the overall architecture of the proposed system and shows how the four methodological layers interact during inference.

[Insert Figure 3.1 here: Overall architecture of the proposed temporal success-aware Reflexion RAG system.]

## 3.3 Retrieval Layer

The retrieval layer is responsible for selecting evidence passages that may support the answer. In the proposed system, retrieval can operate either in lexical mode or in hybrid mode. Lexical retrieval is particularly useful for deterministic offline experiments and fixed-context evaluations because it provides stable and easily auditable evidence ranking. Hybrid retrieval combines dense semantic retrieval with lexical matching in order to improve coverage when the wording of the query differs from the wording of the relevant passage.

After the initial candidate set is collected, the system applies reranking. This stage considers semantic relevance, lexical overlap, exact phrase matches, quoted-text matches, and source-related metadata alignment. Additional local-window and document-span reranking improve passage precision in long documents where the correct answer may be embedded inside a much larger context. This layer is methodologically important because later reasoning stages depend directly on the quality of the retrieved evidence. If evidence selection is weak, even a strong generator and critic may fail to produce a grounded answer.

The retrieval stage shown in Figure 3.1 should therefore be interpreted as the evidence foundation of the whole system.

## 3.4 Benchmark-Aware Prompting and Task Profiles

The proposed system is designed to support several benchmark families with different grounding requirements. For that reason, the methodology does not rely on a single universal prompt. Instead, it uses benchmark-aware task profiles that control the role specification, evidence usage policy, citation requirements, and answer constraints.

For example, legal grounded question answering requires conservative use of source text and may require source-local reasoning, whereas conversational question answering benefits from broader contextual synthesis. Similarly, fixed-context hallucination evaluation requires more restrictive prompting than open-ended response generation. Benchmark-aware prompting is therefore part of the methodology rather than a cosmetic prompt-engineering choice, because it helps align system behavior with the evaluation setting and reduces prompt mismatch as a confounding factor.

Prompt construction integrates task instructions, selected evidence, selected memory guidance, and the current iteration’s reflection guidance. This relationship is illustrated within the architecture flow in Figure 3.1.

## 3.5 Reflexion and Verification Mechanism

The proposed method uses an iterative Reflexion loop instead of a single-pass answer generator. Each reasoning cycle contains four main actions: candidate answer generation, adversarial skepticism, critic evaluation, and optional revision.

First, the model generates a draft answer from the currently available evidence and any selected memory guidance. Second, a skeptic attempts to identify unsupported statements, counter-evidence, and false-premise risks. Third, a critic evaluates whether the answer is sufficiently grounded and complete for the current task. If the answer is judged weak, the system produces guidance for another attempt, which may include retrieval-level, answer-level, or self-reflective feedback.

This mechanism is methodologically important for two reasons. It improves answer quality through iterative correction, and it produces interpretable traces that make later analysis possible. By examining the reasoning trace, it becomes possible to determine whether a failure came from missing evidence, weak answer formation, over-aggressive skepticism, or poor memory transfer.

The execution order of the Reflexion workflow is shown in Figure 3.2.

[Insert Figure 3.2 here: Workflow of answer generation, skeptic review, critic evaluation, reflection, and retry.]

## 3.6 Temporal Success-Aware Episodic Memory

The central contribution of the methodology is the temporal success-aware episodic memory mechanism. Its purpose is to reuse prior reasoning episodes only when they are likely to be beneficial for the current question. Each stored episode contains the previous question, the reasoning outcome, the reflection summary, and outcome statistics indicating whether that memory has historically supported successful or unsuccessful reuse.

Unlike similarity-only memory retrieval, the proposed memory controller ranks candidate memories using a composite score that combines semantic relevance, temporal recency, reliability, hallucination-risk suppression, and metadata locality. As a result, a memory is treated as useful not only because it resembles the current question, but because it has also demonstrated practical reliability over time.

The role of episodic memory in the full architecture is shown in Figure 3.1. Its interaction with the runtime workflow is also reflected in Figure 3.2.

## 3.7 Reliability and Risk Estimation

The methodology estimates memory reliability using accumulated success and failure statistics. Each memory begins with prior parameters and is updated whenever it is reused. Reliability is computed as a normalized estimate of past success, which provides a smoother and more stable trust signal than a raw empirical success rate, especially when the number of reuse events is still small.

Hallucination risk is estimated from previous unsuccessful outcomes. Memories with higher failure tendency are downweighted so that superficially relevant but unreliable reasoning traces do not dominate the final prompt. In effect, reliability encourages reuse of memories that have helped in the past, while hallucination-risk suppression protects the system from repeating harmful reasoning patterns.

This logic is reflected in the memory scoring process in Figure 3.1 and can be summarized by the following equations:

```text
reliability = alpha / (alpha + beta)
temporal_weight = exp(-age_days / decay_days)
risk_penalty = max(epsilon, 1 - lambda * hallucination_risk)
```

## 3.8 Memory Filtering and Conservative Reuse

Memory can produce both positive transfer and harmful transfer. For that reason, the proposed methodology uses conservative memory filtering before any memory is inserted into the prompt. The filtering stage considers task compatibility, minimum memory score, availability of useful reflection text, and the quality of the memory’s past outcomes. In stricter grounded settings, stronger filtering constraints are applied so that only low-risk and contextually aligned memories are reused.

This stage is methodologically important because the value of memory does not come from storing more episodes, but from reusing only the right ones. The conservative filtering strategy reduces the probability that unrelated or failure-prone memories will interfere with the current reasoning process.

The filtering and selection sequence is represented in the memory layer of Figure 3.1.

## 3.9 Memory Update Procedure

After an episode has finished, the system records a new memory entry and updates the replay statistics of any reused memories. When the episode is successful, success-related statistics are increased. When the episode is unsuccessful, failure-related statistics are increased. The system also records recency information so that future retrieval can account for temporal decay.

This update step transforms memory into an adaptive mechanism rather than a static repository of past prompts. Over time, the system learns which reasoning patterns are worth reusing and which ones should be suppressed. This adaptive replay logic is shown conceptually in Figure 3.1 and appears as the final stage of the runtime workflow in Figure 3.2.

## 3.10 Experimental Conditions and Baselines

The experimental methodology compares several systems under matched settings. The comparison includes a vanilla retrieval-augmented generation baseline, a simple refinement baseline, a Reflexion-based system without temporal memory weighting, a temporal architecture without active memory reuse, and the full temporal success-aware memory system proposed in this work.

This comparison strategy isolates the contribution of each design choice. Vanilla retrieval-augmented generation measures the value of standard retrieval plus generation. Reflexion-based baselines measure the effect of iterative self-correction. Memory-based baselines measure whether reusing prior reasoning episodes adds value. The proposed method then tests whether selective, reliability-aware memory reuse improves groundedness and hallucination control more effectively than simpler alternatives.

The relationship between the baseline families and the proposed system is summarized in Figure 3.3.

[Insert Figure 3.3 here: Comparative pipeline of baseline systems and the proposed temporal success-aware method.]

## 3.11 Datasets

The evaluation uses multiple benchmark families in order to test different aspects of grounded question answering. Conversational fixed-context data is used to evaluate cross-turn continuity and the reuse of prior reasoning patterns. Fixed-context groundedness benchmarks are used to evaluate unsupported generation and hallucination under controlled evidence conditions. Document-grounded legal question answering is used to evaluate source-sensitive reasoning, metadata locality, and evidence precision in long texts.

These dataset families are complementary because they stress different failure modes. Conversational evaluation emphasizes transfer and context continuity. Groundedness benchmarks emphasize hallucination control. Legal source-grounded evaluation emphasizes precise evidence use and document-specific reasoning.

If required by the thesis format, this section can be accompanied by a dataset summary table rather than an additional figure.

## 3.12 Evaluation Metrics

The methodology uses several groups of evaluation metrics because no single score is sufficient for a grounded question answering system. Answer quality is measured with exact match and token-level F1 where reference answers are available. Groundedness-related evaluation includes hallucination rate, unsupported answer rate, grounded answer rate, and abstention rate. Retrieval quality is measured with retrieval hit rate or a source-hit proxy where full ranking metrics are not available. In addition, the methodology tracks process-level indicators such as the number of reasoning iterations, memory hit count, average selected memory score, average reliability of reused memories, and average hallucination risk of selected memories.

These metrics are used together because a grounded system should not be assessed only by lexical similarity to a gold answer. It should also be evaluated on whether its answer is supported by the evidence and whether the reasoning procedure that produced it is stable and interpretable.

The relationship between datasets, baselines, metrics, and analysis outputs is shown in Figure 3.4.

[Insert Figure 3.4 here: Evaluation framework linking datasets, baselines, metrics, and analysis outputs.]

## 3.13 Ablation Study

To determine which parts of the memory controller are responsible for observed gains, the methodology includes an ablation study. The full method is compared with variants that remove one component at a time, including temporal decay, reliability weighting, hallucination-risk suppression, metadata bonus, and active memory use itself.

This design is necessary because a strong final result alone does not reveal which mechanism is responsible for the improvement. The ablation study makes it possible to distinguish gains from recency modeling, trust estimation, risk suppression, and locality-aware reuse.

In a thesis presentation, this section is usually best supported by a table rather than a separate figure.

## 3.14 Statistical Analysis

For final reporting, the methodology should include uncertainty-aware comparison rather than relying exclusively on single-run point estimates. Recommended practice includes reporting mean values, confidence intervals, and paired comparisons against the strongest baseline on the same evaluation split. If an online sequential evaluation setting is used, repeated runs with different sample orders should also be considered to measure order sensitivity.

This improves the rigor of the conclusions and helps separate exploratory findings from confirmatory evidence.

## 3.15 Failure Analysis

Quantitative evaluation is complemented by manual failure analysis. Representative failures should be categorized into retrieval failure, evidence-ignored failure, unsupported inference, over-abstention, negative memory transfer, and verification error. This analysis is especially important in memory-augmented reasoning systems because some failures arise not from missing evidence alone, but from the reuse of an inappropriate prior reasoning trace.

Failure analysis should therefore explain not only whether the system failed, but why it failed and which layer of the architecture contributed most directly to the error.

## 3.16 Threats to Validity

Several threats to validity must be acknowledged. These include bias in language-model-based judges, benchmark leakage through memory reuse, order sensitivity in online evaluation, dependence on retrieval quality, sensitivity to benchmark-specific prompting, and provider-level variability in model behavior.

The methodology addresses these risks by using matched configurations across baselines, explicit benchmark-aware task profiles, auditable replay updates, and clear separation between exploratory and final evaluation settings.

## 3.17 Reproducibility

The methodology emphasizes reproducibility through explicit configuration control, benchmark-specific memory separation, saved evaluation summaries, and auditable replay logs. These design choices make it possible to reconstruct how a given result was produced and to distinguish methodological improvements from accidental configuration drift.

In thesis terms, reproducibility is part of the methodology because strong empirical claims require not only good performance, but also traceable experimental conditions.

## 3.18 Chapter Summary

This chapter has presented the methodology of a Reflexion-based retrieval-augmented generation architecture enhanced with temporal success-aware episodic memory. The system combines evidence retrieval, benchmark-aware prompting, iterative verification, and adaptive memory reuse in order to improve grounded answer quality and reduce hallucination. The evaluation design then tests this contribution through controlled baseline comparison, ablation analysis, and multi-level metric reporting across conversational, grounded, and source-sensitive question answering settings.

