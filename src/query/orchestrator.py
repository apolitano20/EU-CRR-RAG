"""
QueryOrchestrator: centralised query classification and routing.

Sits between api/main.py and QueryEngine. Owns:
- Language detection (langdetect + diacritics fallback)
- Query classification (DEFINITION, DIRECT_ARTICLE, CRR_SPECIFIC, CONVERSATIONAL)
- Routing to the appropriate handler
- Post-retrieval confidence fallback for off-topic queries
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Optional

import openai
from openai import AsyncOpenAI

from src.query.query_engine import (
    QueryEngine,
    QueryResult,
    LLM_MODEL,
    USE_PARAGRAPH_CHUNKING,
    USE_MIXED_CHUNKING,
    _EVAL_KWARGS,
    _LEGAL_QA_TEMPLATE,
    _LEGAL_QA_TEMPLATE_WITH_HISTORY,
    _format_history,
    _rewrite_query_with_history,
    _truncate_context,
    _detect_definition_query,
    _detect_direct_article_lookup,
    _normalise_query,
    _enrich_open_ended_query,
    _generate_hyde_query,
    _generate_sub_queries,
    merge_rrf,
)

logger = logging.getLogger(__name__)


def _json_default(obj: object) -> object:
    """Fallback JSON serializer for numpy scalars (float32, int64, etc.)."""
    try:
        return float(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(obj)


# Post-retrieval confidence threshold: below this → use fallback prompt.
# Raised from 0.35 to 0.40 to reduce false fallbacks on borderline-relevant retrievals.
_LOW_CONFIDENCE_THRESHOLD = 0.40

# Synthesis model for complex multi-hop queries requiring deeper reasoning.
_HARD_QUERY_MODEL = os.getenv("HARD_QUERY_MODEL", "gpt-4o")

# Timeout for the ToC routing LLM call (and the parallel thread/task wrapping it).
# The OpenAI call inside _toc_route uses (_TOC_TIMEOUT - 1s) so the Python-level
# timeout fires first and we get a clean log message instead of an OpenAI exception.
_TOC_TIMEOUT = float(os.getenv("TOC_ROUTING_TIMEOUT_SECONDS", "10"))

# Patterns that indicate a multi-hop or comparative question spanning multiple articles.
_MULTI_HOP_RE = re.compile(
    r"\b(?:"
    r"relationship\s+between"
    r"|difference\s+between"
    r"|compare\s+(?:the\s+)?\w+"
    r"|how\s+(?:does|do)\s+.{1,40}\s+(?:affect|interact|relate\s+to)"
    r"|interaction\s+between"
    r"|under\s+both"
    r"|in\s+conjunction\s+with"
    r")\b",
    re.IGNORECASE,
)

# Conversational patterns: greetings, thanks, short non-regulatory messages
_CONVERSATIONAL_RE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank\s+you|ok|okay|test|bye|"
    r"good\s+(?:morning|afternoon|evening))[!.,\s]*$",
    re.IGNORECASE,
)

# ------------------------------------------------------------------
# Prompt templates for new routes
# ------------------------------------------------------------------

_GENERAL_REGULATORY_TEMPLATE = (
    "You are a regulatory compliance expert specialising in EU prudential banking regulation.\n\n"
    "Answer the following question using your general knowledge. Be accurate and helpful.\n"
    "If unsure about specific details, say so. Do NOT invent article numbers or citations.\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

_FALLBACK_TEMPLATE = (
    "You are a regulatory compliance expert specialising in EU prudential banking regulation "
    "(CRR – Regulation (EU) No 575/2013).\n\n"
    "The following context was retrieved but may not fully answer the question. "
    "Use the context where relevant, and supplement with your general knowledge where the "
    "context is insufficient. Clearly indicate when you are drawing on general knowledge "
    "rather than the CRR text.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

_CONVERSATIONAL_TEMPLATE = (
    "You are a CRR regulatory assistant. The user sent a conversational message. "
    "Respond briefly and helpfully. Mention that you can answer questions about the "
    "EU Capital Requirements Regulation (CRR).\n\n"
    "User message: {query_str}\n\n"
    "Response:"
)

# ------------------------------------------------------------------
# Classification types
# ------------------------------------------------------------------


class QueryType(str, Enum):
    DEFINITION = "definition"
    DIRECT_ARTICLE = "direct_article"
    CRR_SPECIFIC = "crr_specific"
    GENERAL = "general"
    CONVERSATIONAL = "conversational"


@dataclass
class ClassificationResult:
    query_type: QueryType
    language: Optional[str]
    definition_signal: Optional[str] = None  # For DEFINITION: term or "#N"
    article_number: Optional[str] = None     # For DIRECT_ARTICLE


# ------------------------------------------------------------------
# Language detection
# ------------------------------------------------------------------


def _detect_language_heuristic(text: str) -> Optional[str]:
    """Diacritics-based language detection fallback."""
    if set(text) & set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"):
        return "pl"
    if set(text) & set("àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"):
        return "it"
    return None


def detect_language(text: str) -> Optional[str]:
    """Detect query language using langdetect with diacritics fallback.

    Returns ISO language code or None for unknown/unsupported.
    Returns "en" for English (unlike the old diacritics heuristic which returned None),
    enabling the language filter to restrict retrieval to English-only nodes.
    """
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 42  # deterministic results
        lang = detect(text)
        if lang in ("en", "it", "pl"):
            return lang
    except Exception:
        pass

    return _detect_language_heuristic(text) or "en"


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


class QueryOrchestrator:
    """Classifies queries and routes them to the appropriate handler.

    Fixes three concrete failures in the previous ad-hoc routing:
    1. General questions (e.g. "What is Basel III?") now get a useful answer via
       the post-retrieval confidence fallback instead of returning the unhelpful
       "context does not contain sufficient information" message.
    2. English queries now set language="en" (via langdetect) so mixed EN/IT results
       are no longer returned.
    3. Definition fast-path now runs regardless of conversation history (removes the
       previous ``if not history:`` guard that silently skipped the DefinitionsStore
       for follow-up questions).
    """

    def __init__(
        self,
        query_engine: QueryEngine,
        openai_api_key: Optional[str] = None,
        llm_model: str = LLM_MODEL,
    ) -> None:
        self._engine = query_engine
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._model = llm_model

    def load(self) -> None:
        """Delegate to QueryEngine.load()."""
        self._engine.load()

    def is_loaded(self) -> bool:
        return self._engine.is_loaded()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, query: str) -> ClassificationResult:
        """Classify query type and detect language.

        Classification order (first match wins):
        1. CONVERSATIONAL — greetings and short non-regulatory messages
        2. DEFINITION     — Article 4 definition queries
        3. DIRECT_ARTICLE — single-article reference queries
        4. CRR_SPECIFIC   — everything else (may fall back post-retrieval)
        """
        language = detect_language(query)
        norm = _normalise_query(query)

        # 1. Conversational: matches greeting/short-message pattern
        if _CONVERSATIONAL_RE.match(query.strip()):
            return ClassificationResult(
                query_type=QueryType.CONVERSATIONAL,
                language=language,
            )

        # 2. Definition fast-path: specific term or Article 4(N)
        signal = _detect_definition_query(norm)
        if signal is not None:
            return ClassificationResult(
                query_type=QueryType.DEFINITION,
                language=language,
                definition_signal=signal,
            )

        # Generic "Article 4" query → DEFINITION (returns summary)
        if _detect_direct_article_lookup(norm) == "4":
            return ClassificationResult(
                query_type=QueryType.DEFINITION,
                language=language,
                definition_signal=None,
            )

        # 3. Direct single-article lookup
        art_num = _detect_direct_article_lookup(norm)
        if art_num is not None:
            return ClassificationResult(
                query_type=QueryType.DIRECT_ARTICLE,
                language=language,
                article_number=art_num,
            )

        # 4. Default: CRR_SPECIFIC (RAG path; may fall back to general post-retrieval)
        return ClassificationResult(
            query_type=QueryType.CRR_SPECIFIC,
            language=language,
        )

    # ------------------------------------------------------------------
    # Sync query
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        language: Optional[str] = None,
        max_cross_ref_expansions: Optional[int] = None,
        history: Optional[list[dict]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> QueryResult:
        """Classify, route, and return a complete query result."""
        history = history or []

        # Stage 0: Query rewriting for follow-up queries
        effective_query = user_query
        if history:
            effective_query = _rewrite_query_with_history(
                user_query, history, self._api_key, self._model
            )
            if effective_query != user_query:
                logger.info("Query rewritten: '%s' → '%s'", user_query, effective_query)

        # Classify the effective query — always, regardless of history
        classification = self.classify(effective_query)

        # Explicit language preference overrides auto-detected
        lang = language or classification.language

        logger.info(
            "Query classified: type=%s, language=%s, query='%s'",
            classification.query_type,
            lang,
            effective_query[:80],
        )

        if classification.query_type == QueryType.CONVERSATIONAL:
            return self._handle_conversational(effective_query)

        if classification.query_type == QueryType.DEFINITION:
            result = self._engine.lookup_definition(effective_query, lang)
            if result is not None:
                return result
            # Fall through to CRR_SPECIFIC if definition not found in store
            classification = ClassificationResult(
                query_type=QueryType.CRR_SPECIFIC, language=lang
            )

        # CRR_SPECIFIC and DIRECT_ARTICLE both go through retrieval + synthesis.
        # For open-ended CRR_SPECIFIC queries (no article reference): enrich with article hints.
        retrieve_query = effective_query
        is_multi_hop = bool(_MULTI_HOP_RE.search(effective_query))
        _use_hyde   = os.getenv("USE_HYDE", "false").lower() == "true"
        _use_enrich = os.getenv("USE_QUERY_ENRICHMENT", "true").lower() == "true"
        if (
            classification.query_type == QueryType.CRR_SPECIFIC
            and not _detect_direct_article_lookup(_normalise_query(effective_query))
        ):
            if _use_hyde:
                # True HyDE: generate a hypothetical CRR-style passage + article hints.
                # Embeds legal vocabulary rather than the plain-language query, targeting
                # terminology dilution failures where query and article text share few tokens.
                retrieve_query = _generate_hyde_query(effective_query, self._api_key)
            elif _use_enrich:
                # Article-hint enrichment: append predicted article numbers to the query.
                retrieve_query = _enrich_open_ended_query(effective_query, self._api_key)

        # Checkpoint: bail out before expensive retrieval if the client already timed out.
        if cancel is not None and cancel.is_set():
            raise TimeoutError("Query cancelled (client timed out before retrieval).")

        # Multi-hop queries: generate sub-queries and merge retrieval results for broader coverage.
        if is_multi_hop and classification.query_type == QueryType.CRR_SPECIFIC:
            sub_queries = _generate_sub_queries(effective_query, self._api_key)
            try:
                all_nodes, sources, trace_id, norm_query = self._multi_query_retrieve(
                    retrieve_query, sub_queries, lang, max_cross_ref_expansions
                )
            except Exception as exc:
                logger.warning(
                    "Multi-hop retrieval failed (%s: %s) - falling back to single-query retrieval",
                    type(exc).__name__, exc,
                )
                all_nodes, sources, trace_id, norm_query, _eng = self._engine.retrieve(
                    retrieve_query, lang, max_cross_ref_expansions
                )
        else:
            all_nodes, sources, trace_id, norm_query, _eng = self._engine.retrieve(
                retrieve_query, lang, max_cross_ref_expansions
            )

        # Post-retrieval ToC supplement: only fires when retrieval confidence is low.
        # Run_15 showed that universal ToC routing hurts high-confidence categories
        # (liquidity -7.4pp, own_funds -6.5pp) while helping low-confidence ones
        # (diluted_embedding +37.5pp Recall@3, false_friend +7.1pp Hit@1).
        # Gating on max_score targets the gain without the collateral damage.
        _use_toc = (
            os.getenv("USE_TOC_ROUTING", "false").lower() == "true"
            and self._engine.toc_store is not None
            and classification.query_type == QueryType.CRR_SPECIFIC
            and not _detect_direct_article_lookup(_normalise_query(effective_query))
            and not is_multi_hop
        )
        if _use_toc:
            _toc_threshold = float(os.getenv("TOC_CONFIDENCE_THRESHOLD", "0.55"))
            max_score = max((n.score or 0.0) for n in all_nodes) if all_nodes else 0.0
            if max_score < _toc_threshold:
                logger.info(
                    "ToC routing triggered: max_score=%.3f < threshold=%.2f",
                    max_score, _toc_threshold,
                )
                toc_articles = self._toc_route(effective_query, lang)
                if toc_articles:
                    toc_nodes = self._engine.toc_retrieve(toc_articles, retrieve_query, lang)
                    logger.info(
                        "ToC routing: predicted=%s, fetched_nodes=%d",
                        toc_articles, len(toc_nodes),
                    )
                    if toc_nodes:
                        merged_nodes = merge_rrf(all_nodes, toc_nodes)
                        all_nodes = merged_nodes
                        sources = [
                            {
                                "text": n.node.get_content()[:500],
                                "score": float(n.score or 0.0),
                                "metadata": n.node.metadata,
                                "expanded": False,
                            }
                            for n in merged_nodes
                        ]
            else:
                logger.info(
                    "ToC routing skipped: max_score=%.3f >= threshold=%.2f",
                    max_score, _toc_threshold,
                )

        # Checkpoint: bail out before synthesis LLM call if the client already timed out.
        if cancel is not None and cancel.is_set():
            raise TimeoutError("Query cancelled (client timed out before synthesis).")

        context_str = self._build_context(all_nodes)
        prompt = self._select_prompt(
            classification.query_type, sources, norm_query, context_str, history
        )

        # Use the hard model (gpt-4o) for complex multi-hop queries; fast model otherwise.
        synthesis_model = _HARD_QUERY_MODEL if is_multi_hop else self._model

        oai_client = openai.OpenAI(api_key=self._api_key)
        response = oai_client.chat.completions.create(
            model=synthesis_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
            **_EVAL_KWARGS,
        )
        answer = (response.choices[0].message.content or "").strip()
        return QueryResult(answer=answer, sources=sources, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Async streaming query
    # ------------------------------------------------------------------

    async def query_stream(
        self,
        user_query: str,
        language: Optional[str],
        history: list[dict],
        max_cross_ref_expansions: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted events."""
        import asyncio

        # Stage 0: Query rewriting for follow-up queries
        effective_query = user_query
        if history:
            effective_query = await asyncio.to_thread(
                _rewrite_query_with_history,
                user_query,
                history,
                self._api_key,
                self._model,
            )
            if effective_query != user_query:
                logger.info(
                    "Stream query rewritten: '%s' → '%s'", user_query, effective_query
                )

        # Classify the effective query — always, regardless of history
        classification = self.classify(effective_query)

        # Explicit language preference overrides auto-detected
        lang = language or classification.language

        logger.info(
            "Stream query classified: type=%s, language=%s, query='%s'",
            classification.query_type,
            lang,
            effective_query[:80],
        )

        # Conversational: short-circuit before retrieval
        if classification.query_type == QueryType.CONVERSATIONAL:
            async for event in self._stream_conversational(effective_query, lang):
                yield event
            return

        # Definition fast-path: skip retrieval and LLM for Article 4 queries
        if classification.query_type == QueryType.DEFINITION:
            def_result = await asyncio.to_thread(
                self._engine.lookup_definition, effective_query, lang
            )
            if def_result is not None:
                yield f"data: {json.dumps({'type': 'token', 'content': def_result.answer})}\n\n"
                sources_event = {
                    "type": "sources",
                    "sources": def_result.sources,
                    "trace_id": def_result.trace_id,
                    "language": lang,
                    "query_type": QueryType.DEFINITION.value,
                }
                yield f"data: {json.dumps(sources_event, default=_json_default)}\n\n"
                yield 'data: {"type": "done"}\n\n'
                return
            # Fall through to CRR_SPECIFIC
            classification = ClassificationResult(
                query_type=QueryType.CRR_SPECIFIC, language=lang
            )

        # Enrich open-ended CRR_SPECIFIC queries with article number hints
        retrieve_query = effective_query
        is_multi_hop_stream = bool(_MULTI_HOP_RE.search(effective_query))
        is_direct_article_stream = _detect_direct_article_lookup(_normalise_query(effective_query))
        _use_hyde_stream   = os.getenv("USE_HYDE", "false").lower() == "true"
        _use_enrich_stream = os.getenv("USE_QUERY_ENRICHMENT", "true").lower() == "true"
        if (
            classification.query_type == QueryType.CRR_SPECIFIC
            and not is_direct_article_stream
        ):
            if _use_hyde_stream:
                retrieve_query = await asyncio.to_thread(
                    _generate_hyde_query, effective_query, self._api_key
                )
            elif _use_enrich_stream:
                retrieve_query = await asyncio.to_thread(
                    _enrich_open_ended_query, effective_query, self._api_key
                )

        # Run retrieval first; ToC supplement fires post-retrieval if confidence is low.
        all_nodes, sources, trace_id, norm_query, _eng = await asyncio.to_thread(
            self._engine.retrieve,
            retrieve_query,
            lang,
            max_cross_ref_expansions,
        )

        # Post-retrieval ToC supplement (mirrors sync query() logic).
        _use_toc_stream = (
            os.getenv("USE_TOC_ROUTING", "false").lower() == "true"
            and self._engine.toc_store is not None
            and classification.query_type == QueryType.CRR_SPECIFIC
            and not is_direct_article_stream
            and not is_multi_hop_stream
        )
        if _use_toc_stream:
            _toc_threshold_stream = float(os.getenv("TOC_CONFIDENCE_THRESHOLD", "0.55"))
            max_score_stream = max((n.score or 0.0) for n in all_nodes) if all_nodes else 0.0
            if max_score_stream < _toc_threshold_stream:
                logger.info(
                    "ToC routing (stream) triggered: max_score=%.3f < threshold=%.2f",
                    max_score_stream, _toc_threshold_stream,
                )
                try:
                    toc_articles = await asyncio.wait_for(
                        asyncio.to_thread(self._toc_route, effective_query, lang),
                        timeout=_TOC_TIMEOUT,
                    )
                except Exception as exc:
                    logger.warning("ToC routing (stream) timed out or failed: %s", exc)
                    toc_articles = []
                if toc_articles:
                    toc_nodes = await asyncio.to_thread(
                        self._engine.toc_retrieve, toc_articles, retrieve_query, lang
                    )
                    logger.info(
                        "ToC routing (stream): predicted=%s, fetched_nodes=%d",
                        toc_articles, len(toc_nodes),
                    )
                    if toc_nodes:
                        merged_nodes = merge_rrf(all_nodes, toc_nodes)
                        all_nodes = merged_nodes
                        sources = [
                            {
                                "text": n.node.get_content()[:500],
                                "score": float(n.score or 0.0),
                                "metadata": n.node.metadata,
                                "expanded": False,
                            }
                            for n in merged_nodes
                        ]
            else:
                logger.info(
                    "ToC routing (stream) skipped: max_score=%.3f >= threshold=%.2f",
                    max_score_stream, _toc_threshold_stream,
                )

        context_str = self._build_context(all_nodes)
        prompt = self._select_prompt(
            classification.query_type, sources, norm_query, context_str, history
        )

        query_type_val = classification.query_type.value
        synthesis_model_stream = _HARD_QUERY_MODEL if is_multi_hop_stream else self._model

        client = AsyncOpenAI(api_key=self._api_key)
        stream = await client.chat.completions.create(
            model=synthesis_model_stream,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=120.0,
            **_EVAL_KWARGS,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        sources_event = {
            "type": "sources",
            "sources": sources,
            "trace_id": trace_id,
            "language": lang,
            "query_type": query_type_val,
        }
        yield f"data: {json.dumps(sources_event, default=_json_default)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _multi_query_retrieve(
        self,
        main_query: str,
        sub_queries: list[str],
        lang: Optional[str],
        max_expansions: Optional[int],
    ) -> tuple[list, list, str, str]:
        """Retrieve nodes for main query + sub-queries; merge by score-based deduplication.

        All queries run retrieval independently. Results are merged by keeping the highest
        score per node_id and sorting descending. This broadens coverage for multi-hop
        questions that span multiple CRR articles.

        Returns:
            (merged_nodes, sources, trace_id, norm_query)
        """
        import uuid as _uuid

        merged: dict[str, object] = {}   # node_id -> NodeWithScore (highest score wins)
        trace_id = str(_uuid.uuid4())
        norm_query = _normalise_query(main_query)

        for i, q in enumerate([main_query] + sub_queries):
            try:
                nodes, _sources, tid, nq, _ = self._engine.retrieve(q, lang, max_expansions)
                if i == 0:
                    trace_id = tid
                    norm_query = nq
                for node in nodes:
                    nid = node.node.node_id
                    existing_score = (merged[nid].score or 0.0) if nid in merged else -1.0
                    if (node.score or 0.0) > existing_score:
                        merged[nid] = node
            except Exception as exc:
                logger.warning("Sub-query retrieve failed for '%s...': %s", q[:50], exc)

        merged_nodes = sorted(merged.values(), key=lambda n: n.score or 0.0, reverse=True)
        sources = [
            {
                "text": n.node.get_content()[:500],
                "score": float(round(n.score or 0.0, 4)),
                "metadata": n.node.metadata,
                "expanded": False,
            }
            for n in merged_nodes
        ]
        logger.info(
            "Multi-query retrieval: %d queries → %d unique nodes",
            1 + len(sub_queries),
            len(merged_nodes),
        )
        return list(merged_nodes), sources, trace_id, norm_query

    def _toc_route(self, query: str, language: Optional[str]) -> list[str]:
        """Ask GPT-4o-mini to predict relevant CRR articles from the Table of Contents.

        Runs in parallel with vector retrieval. Returns up to 6 article numbers,
        or [] on any failure (timeout, API error, JSON parse error).

        The returned article numbers are validated to match r'^\\d+[a-z]*$' so
        that Roman numeral annex IDs and free-text values are naturally filtered out.
        """
        toc = self._engine.toc_store
        lang = language or "en"
        if toc is None or not toc.is_loaded(lang):
            return []

        toc_text = toc.format_for_prompt(lang)
        if not toc_text:
            return []

        prompt = (
            "You are a legal expert on the EU CRR (Regulation (EU) No 575/2013).\n\n"
            "Below is the Table of Contents. Each entry shows article number, title, "
            "hierarchy, and key terms from the article text.\n\n"
            "<TOC>\n"
            f"{toc_text}\n"
            "</TOC>\n\n"
            "Given the user's question, identify which articles most likely contain the answer.\n\n"
            "Think step by step:\n"
            "1. What regulatory concepts does the question involve?\n"
            "2. Which parts/titles/chapters cover those?\n"
            "3. Which specific articles are relevant?\n\n"
            'Return JSON: {"reasoning": "...", "articles": ["92", "93"]}\n'
            "Maximum 6 articles, ordered by relevance. "
            "Only include articles that appear in the ToC above.\n\n"
            f"Question: {query}"
        )

        # Use _TOC_TIMEOUT - 1s so the Python-level thread/asyncio timeout fires
        # before the OpenAI SDK raises its own exception, giving cleaner log messages.
        openai_timeout = max(2.0, _TOC_TIMEOUT - 1.0)
        try:
            import openai as _openai
            client = _openai.OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200,
                timeout=openai_timeout,
                **_EVAL_KWARGS,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("ToC routing LLM call failed: %s", exc)
            return []

        # Parse JSON response
        _ARTICLE_NUM_RE = re.compile(r"^\d+[a-z]*$")
        try:
            data = json.loads(raw)
            articles = data.get("articles", [])
            validated = [
                str(a) for a in articles
                if _ARTICLE_NUM_RE.match(str(a))
            ][:6]
            logger.info("ToC routing predicted articles: %s", validated)
            return validated
        except (json.JSONDecodeError, Exception):
            # Regex fallback: extract bare article numbers from raw text
            fallback = re.findall(r"\b(\d+[a-z]*)\b", raw)
            validated = [a for a in fallback if _ARTICLE_NUM_RE.match(a)][:6]
            logger.warning(
                "ToC routing: JSON parse failed, regex fallback found: %s", validated
            )
            return validated

    def _build_context(self, all_nodes: list) -> str:
        """Build article-labelled context string from retrieved nodes."""
        context_parts = []
        _needs_parent_fetch = USE_PARAGRAPH_CHUNKING or USE_MIXED_CHUNKING
        if _needs_parent_fetch:
            # Any PARAGRAPH chunks in results — fetch parent ARTICLE docs for full synthesis context.
            # Mixed mode: deduplicator ensured 1 chunk per article; some may already be ARTICLE.
            article_order: list[str] = []
            _seen_arts: set[str] = set()
            node_language: Optional[str] = None
            for node in all_nodes:
                meta = node.node.metadata
                art = meta.get("article", "")
                if not node_language:
                    node_language = meta.get("language")
                if art and art not in _seen_arts:
                    _seen_arts.add(art)
                    article_order.append(art)
            for art in article_order:
                # In mixed mode: winning chunk may already be ARTICLE — use directly.
                winning_node = next(
                    (n for n in all_nodes if n.node.metadata.get("article") == art),
                    None,
                )
                winning_type = (winning_node.node.metadata.get("chunk_type", "") if winning_node else "")
                if winning_type == "ARTICLE":
                    meta = winning_node.node.metadata  # type: ignore[union-attr]
                    art_title = meta.get("article_title", "")
                    header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                    content = meta.get("display_text") or winning_node.node.get_content()  # type: ignore[union-attr]
                    context_parts.append(f"{header}\n\n{content}")
                else:
                    fetch_conds: list[tuple[str, str]] = [("article", art), ("chunk_type", "ARTICLE")]
                    if node_language:
                        fetch_conds.append(("language", node_language))
                    art_nodes = self._engine._fetch_nodes_direct(fetch_conds, top_k=1)
                    if art_nodes:
                        art_node = art_nodes[0].node
                        art_title = art_node.metadata.get("article_title", "")
                        header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                        content = art_node.metadata.get("display_text") or art_node.get_content()
                        context_parts.append(f"{header}\n\n{content}")
                    else:
                        # Fallback: use paragraph chunk text directly if ARTICLE doc not found
                        if winning_node is not None:
                            meta = winning_node.node.metadata
                            art_title = meta.get("article_title", "")
                            header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                            context_parts.append(f"{header}\n\n{winning_node.node.get_content()}")
        else:
            for node in all_nodes:
                meta = node.node.metadata
                art = meta.get("article", "")
                art_title = meta.get("article_title", "")
                header = f"Article {art}" + (f" — {art_title}" if art_title else "")
                context_parts.append(f"{header}\n\n{node.node.get_content()}")
        return _truncate_context("\n\n---\n\n".join(context_parts))

    def _select_prompt(
        self,
        query_type: QueryType,
        sources: list[dict],
        norm_query: str,
        context_str: str,
        history: list[dict],
    ) -> str:
        """Select the appropriate prompt based on query type and retrieval confidence."""
        # Post-retrieval confidence check for CRR_SPECIFIC queries
        if query_type == QueryType.CRR_SPECIFIC:
            max_score = max(
                (s["score"] for s in sources if not s.get("expanded")), default=0.0
            )
            has_article_ref = _detect_direct_article_lookup(norm_query) is not None
            if max_score < _LOW_CONFIDENCE_THRESHOLD and not has_article_ref:
                logger.info(
                    "Low retrieval confidence (%.2f) — using fallback prompt", max_score
                )
                return _FALLBACK_TEMPLATE.format(
                    context_str=context_str, query_str=norm_query
                )

        history_str = _format_history(history)
        if history_str:
            return _LEGAL_QA_TEMPLATE_WITH_HISTORY.format(
                history_str=history_str,
                context_str=context_str,
                query_str=norm_query,
            )
        return _LEGAL_QA_TEMPLATE.format(
            context_str=context_str, query_str=norm_query
        )

    def _handle_conversational(self, query: str) -> QueryResult:
        """Handle conversational greetings/short messages via direct LLM call."""
        prompt = _CONVERSATIONAL_TEMPLATE.format(query_str=query)
        oai_client = openai.OpenAI(api_key=self._api_key)
        response = oai_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30.0,
            **_EVAL_KWARGS,
        )
        answer = (response.choices[0].message.content or "").strip()
        return QueryResult(answer=answer, sources=[], trace_id=str(uuid.uuid4()))

    async def _stream_conversational(
        self, query: str, lang: Optional[str]
    ) -> AsyncGenerator[str, None]:
        """Stream a conversational response."""
        trace_id = str(uuid.uuid4())
        prompt = _CONVERSATIONAL_TEMPLATE.format(query_str=query)
        client = AsyncOpenAI(api_key=self._api_key)
        stream = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            timeout=30.0,
            **_EVAL_KWARGS,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"
        sources_event = {
            "type": "sources",
            "sources": [],
            "trace_id": trace_id,
            "language": lang,
            "query_type": QueryType.CONVERSATIONAL.value,
        }
        yield f"data: {json.dumps(sources_event, default=_json_default)}\n\n"
        yield 'data: {"type": "done"}\n\n'
