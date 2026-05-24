# 50-70 Page Expansion Checklist

Use this checklist to turn `thesis_draft.md` into a 50-70 page submission-ready manuscript without filler.

## Expected Word Count

Depending on formatting:

- 50 pages usually requires about 14,000-17,000 words.
- 70 pages usually requires about 19,000-24,000 words.
- The current draft is about 5,858 words, so it needs roughly 9,000-16,000 more words.

## Recommended Expansion by Section

| Section | Current role | Target pages | What to add |
|---|---|---:|---|
| Abstract | Summary | 1 | Keep near 250-350 words; do not expand much |
| Introduction | Research framing | 5-7 | Add motivation, high-stakes examples, research questions, novelty, scope |
| Chapter 1 | Theory | 11-14 | Add equations for RAG, definitions, hallucination taxonomy, examples |
| Chapter 2 | Related work | 10-13 | Add comparison tables, discuss 25-30 cited papers, identify gap |
| Chapter 3 | Method | 12-15 | Add diagrams, module interfaces, prompts, pseudocode, memory examples |
| Chapter 4 | Experiments | 13-17 | Add full result tables, ablation results, propagation metric, examples |
| Conclusion | Closing | 3-4 | Link findings to hypothesis, contributions, limitations, future work |
| References | Bibliography | 3-5 | Use 30+ sources; current draft has 40 |

## Chapter 1 Expansion Prompts

1. Add a formal RAG definition:
   `answer = G(q, R(q, C))`, where `q` is the query, `C` is the corpus, `R` is the retriever, and `G` is the generator.
2. Add a hallucination taxonomy:
   unsupported claim, contradiction, wrong entity, wrong number, false premise, over-answering, citation mismatch.
3. Add one concrete RAG hallucination example from legal QA.
4. Add a subsection-level transition explaining why RAG becomes riskier in agentic settings.
5. Cite Ji et al., Lewis et al., Gao et al., RAGTruth, TruthfulQA, FActScore, SelfCheckGPT.

## Chapter 2 Expansion Prompts

1. Add a related-work comparison table:
   method, core idea, hallucination control, memory support, limitation.
2. Discuss baseline RAG, advanced retrieval, Self-Refine, Reflexion, Self-RAG, RAGAS, ARES, RAGTruth.
3. Add a paragraph after each subsection explaining why the reviewed methods are insufficient for propagation control.
4. End with a sharper research gap:
   existing work reduces final hallucination but rarely measures whether unsupported claims survive across iterations or memory.

## Chapter 3 Expansion Prompts

1. Insert `architecture.drawio` as Figure 3.1.
2. Insert `pipeline.drawio` as Figure 3.2.
3. Add a table mapping implementation functions to architecture modules:
   retrieval, generation, critic, skeptic, reflection, memory scoring, memory update.
4. Add prompt templates for generator, critic, and skeptic as abbreviated examples.
5. Add a worked example of memory scoring.
6. Add a paragraph explaining why temporal decay and hallucination-risk penalty are necessary.
7. Add a full algorithm in formal pseudocode.

## Chapter 4 Expansion Prompts

1. Add dataset statistics:
   number of records, domain, task type, evidence type, gold labels, evaluation limitations.
2. Add result tables for each backend:
   without RAG, vanillaRAG, selfRefineAgentRAG, reflexionAgentRAG, ReflexionTemporalMemorySuccessAgentRAG.
3. Add the missing propagation table:
   unsupported claims, repeated unsupported claims, final propagated claims, memory propagated claims, PER.
4. Add qualitative examples:
   one success, one hallucination corrected by reflection, one failure.
5. Add a strong limitations section:
   detector reliability, fixed-context evaluation, small legal sample, memory effect hidden by independent questions.

## References

The current draft has 40 references. Before final submission:

1. Verify the required citation style from the department.
2. Convert arXiv-only references to conference versions where available.
3. Make sure every reference is cited in the body.
4. Keep only sources actually used in the thesis argument.

## Minimum Work Needed for a Strong Final Version

1. Expand Chapter 1 to at least 4,000 words.
2. Expand Chapter 2 to at least 4,000 words.
3. Expand Chapter 3 to at least 4,500 words.
4. Expand Chapter 4 to at least 5,000 words.
5. Add the propagation metric computation and table.
6. Insert diagrams and captions.
7. Format references consistently.
