"""
LLM-as-judge for the CRR RAG eval pipeline.

Scores a RAG-generated answer against a reference answer on three dimensions:
  - correctness:   factual accuracy of the answer
  - completeness:  coverage of key points from the reference
  - faithfulness:  absence of hallucinations / unsupported claims

Usage:
    from evals.judge import judge_answer
    scores = judge_answer(question, rag_answer, reference_answer)
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-4o"
JUDGE_METRIC_KEYS = ["judge_correctness", "judge_completeness", "judge_faithfulness"]

_SYSTEM_PROMPT = """\
You are an expert evaluator of regulatory Q&A systems specialised in the EU Capital Requirements Regulation (CRR – Regulation (EU) No 575/2013).

Your task is to score a RAG-generated answer against a reference answer on three dimensions.

Scoring dimensions (each 0.0 – 1.0):
  - correctness:   Is every factual claim in the RAG answer accurate according to the CRR? \
Penalise wrong article numbers, incorrect thresholds, misquoted rules.
  - completeness:  Does the RAG answer cover all key points present in the reference answer? \
Penalise missing requirements, omitted conditions, or dropped sub-questions.
  - faithfulness:  Does the RAG answer avoid hallucinations and claims not supported by the \
retrieved context? Penalise invented rules, numbers, or article references.

Output ONLY a JSON object with exactly these four keys:
  "judge_correctness"   – float in [0.0, 1.0]
  "judge_completeness"  – float in [0.0, 1.0]
  "judge_faithfulness"  – float in [0.0, 1.0]
  "judge_rationale"     – 1–3 sentence explanation of your scores

Do not include any text outside the JSON object.\
"""

_USER_TEMPLATE = """\
QUESTION:
{question}

RAG_ANSWER:
{rag_answer}

REFERENCE_ANSWER:
{reference_answer}
"""


def judge_answer(
    question: str,
    rag_answer: str,
    reference_answer: str,
    model: str = JUDGE_MODEL,
) -> dict:
    """
    Call gpt-4o to score the RAG answer against the reference.

    Returns a dict with keys:
        judge_correctness, judge_completeness, judge_faithfulness  (float | None)
        judge_rationale  (str | None)

    On any error all scores are None and the error is logged.
    """
    _null = {k: None for k in JUDGE_METRIC_KEYS}
    _null["judge_rationale"] = None

    if not rag_answer or not rag_answer.strip():
        logger.warning("judge_answer: rag_answer is empty — returning null scores")
        return _null

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("judge_answer: openai package not installed")
        return _null

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("judge_answer: OPENAI_API_KEY not set")
        return _null

    client = OpenAI(api_key=api_key)

    user_msg = _USER_TEMPLATE.format(
        question=question,
        rag_answer=rag_answer,
        reference_answer=reference_answer or "(no reference answer provided)",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
            timeout=60,
        )
    except Exception as exc:
        logger.error("judge_answer: OpenAI API error: %s", exc)
        return _null

    raw = response.choices[0].message.content or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("judge_answer: JSON parse error: %s — raw=%r", exc, raw[:200])
        return _null

    result: dict = {}
    for key in JUDGE_METRIC_KEYS:
        val = data.get(key)
        if val is not None:
            try:
                val = float(val)
                val = max(0.0, min(1.0, val))   # clamp to [0, 1]
            except (TypeError, ValueError):
                val = None
        result[key] = val

    result["judge_rationale"] = str(data.get("judge_rationale", "")) or None
    return result
