# Baseline Run — Full Dataset

**173 cases | 160 successful | 13 failed (7.5%)**
Run timestamp: 2026-03-20

---

## Aggregate Results

| Metric | Value | What it means |
|---|---|---|
| **Hit@1** | 81.3% | The correct article was the #1 ranked result |
| **Recall@1** | 77.2% | Correct article(s) covered in the top result (lower than Hit@1 because multi-article questions dilute this) |
| **Recall@3** | 82.6% | Correct article found somewhere in the top 3 results |
| **Recall@5** | 85.3% | Correct article found somewhere in the top 5 results |
| **MRR** | 0.842 | Average of 1/rank — how far down the list you need to go to find the right article |
| **Precision@3** | 0.300 | Of the 3 returned articles, 30% are relevant |
| **Precision@5** | 0.186 | Of the 5 returned articles, 18.6% are relevant |
| **Judge Correctness** | 0.774 | LLM-scored: does the answer contain the right information? |
| **Judge Completeness** | 0.788 | LLM-scored: does the answer cover all relevant aspects? |
| **Judge Faithfulness** | 0.799 | LLM-scored: does the answer stay grounded in the retrieved sources (no hallucination)? |

**Notable:** 13 failed cases (7.5%) didn't produce a result at all — hard failures worth investigating separately.

The gap between Recall@1 (77.2%) and Recall@5 (85.3%) means ~8% of cases have the right article in the top 5 but not at rank 1 — a reranking opportunity. The remaining ~15% are hard misses not recovered even at rank 5.

---

## GPT Review Prompt

```
You are evaluating the retrieval and answer quality of a RAG system built on the EU Capital Requirements Regulation (CRR – Regulation (EU) No 575/2013).

The system ingests the consolidated CRR HTML, indexes it preserving legal hierarchy (Part → Title → Chapter → Section → Article → Paragraph → Point), and answers regulatory compliance questions with citations via a hybrid retrieval pipeline (BM25 + dense vector embeddings, fused via Reciprocal Rank Fusion, with an AutoMerging retrieval step).

## Baseline Eval Results (full dataset, 173 cases)

- Total cases: 173 | Successful: 160 | Failed: 13 (7.5%)
- Hit@1: 81.3% — correct article ranked #1
- Recall@1: 77.2% — correct article(s) covered in top result
- Recall@3: 82.6% — correct article in top 3
- Recall@5: 85.3% — correct article in top 5
- MRR: 0.842
- Precision@3: 0.300 | Precision@5: 0.186
- Judge Correctness: 0.774
- Judge Completeness: 0.788
- Judge Faithfulness: 0.799

## Context

- Embedding model: bge-small-en-v1.5
- LLM: GPT-4o-mini (synthesis) + GPT-4o (judge)
- Retrieval: hybrid BM25 + Chroma vector store, RRF fusion, AutoMergingRetriever
- Dataset: 173 golden questions across categories (own_funds, capital_ratios), difficulties (easy/medium/hard), and question types (definition, procedural, multi_hop, threshold, false_friend, negative, multi_article)

## Questions for review

1. How do these numbers compare to typical RAG benchmarks for legal/regulatory domains?
2. Which metrics suggest the biggest improvement opportunity — retrieval ranking, retrieval coverage, or answer generation?
3. The gap between Recall@1 (77.2%) and Recall@5 (85.3%) suggests a reranking problem for ~8% of cases. What reranking approaches would you recommend for a legal text domain?
4. Judge Faithfulness (0.799) is the highest of the three judge scores, but Correctness (0.774) is the lowest. What does this pattern indicate?
5. 13 cases failed entirely (no result returned). What are the most likely causes and how should I investigate them?
6. What would you prioritise to move Hit@1 from 81% toward 90%+?
```
