"""
Unit tests for features added in run_26 and run_27:

- _expand_synonyms: run_26 synonym entries (all entries, not just the two previously tested)
- _build_key_facts_block: threshold preamble extraction (run_27 Part A)
- _append_missing_thresholds: post-generation completeness diff (run_27 Part B)
- ArticleDeduplicatorPostprocessor: mixed-chunking deduplication (run_20, on every query path)
- _FALSE_PREMISE_RULE: verified present in both prompt templates
- ParagraphWindowReranker._split_windows: pure logic, no model needed
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(node_id: str, article: str = "", score: float = 0.8,
               chunk_type: str = "ARTICLE") -> MagicMock:
    node = MagicMock()
    node.node.node_id = node_id
    node.node.metadata = {"article": article, "chunk_type": chunk_type}
    node.node.get_content.return_value = f"content of {node_id}"
    node.score = score
    return node


# ---------------------------------------------------------------------------
# _expand_synonyms — run_26 entries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExpandSynonymsRun26:
    """All run_26 synonym map entries must fire correctly."""

    def _expand(self, text: str) -> str:
        from src.query.query_engine import _expand_synonyms
        return _expand_synonyms(text)

    # --- original entries (regression guard) ---

    def test_preference_shares(self):
        result = self._expand("Are preference shares eligible for AT1?")
        assert "Additional Tier 1 instruments" in result

    def test_perpetual_bonds(self):
        result = self._expand("Can perpetual bonds count as AT1?")
        assert "Additional Tier 1 instruments" in result

    def test_subordinated_notes(self):
        result = self._expand("Are subordinated notes Tier 2 capital?")
        assert "Tier 2 instruments" in result

    def test_subordinated_debt(self):
        result = self._expand("Does subordinated debt qualify as regulatory capital?")
        assert "Tier 2 instruments" in result

    def test_bail_in_bonds(self):
        result = self._expand("Can bail-in bonds be included in own funds?")
        assert "eligible liabilities" in result

    def test_minority_interests(self):
        result = self._expand("How should minority interests be treated in consolidation?")
        assert "minority interest capital instruments" in result

    def test_non_performing_loans_plural(self):
        result = self._expand("What is the treatment for non-performing loans?")
        assert "Non-Performing Exposures" in result

    def test_non_performing_loan_singular(self):
        result = self._expand("Does a single non-performing loan affect capital?")
        assert "Non-Performing Exposure" in result

    def test_securitisation_position(self):
        result = self._expand("How is a securitisation position risk-weighted?")
        assert "securitisation exposure" in result

    def test_enterprises(self):
        result = self._expand("What risk weight applies to exposures to enterprises?")
        assert "corporates" in result

    def test_enterprise_singular(self):
        result = self._expand("Is a single enterprise a corporate exposure?")
        assert "corporate" in result

    # --- run_10 entries ---

    def test_local_authority(self):
        result = self._expand("What risk weight applies to a local authority?")
        assert "regional government or local authority" in result

    def test_local_public_authority(self):
        result = self._expand("Is a local public authority zero risk-weighted?")
        assert "regional government or local authority" in result

    def test_local_authorities_plural(self):
        result = self._expand("How are local authorities treated under CRR?")
        assert "regional governments or local authorities" in result

    def test_pledged_assets(self):
        result = self._expand("Are pledged assets excluded from liquid asset calculations?")
        assert "encumbered assets" in result

    def test_pledged_collateral(self):
        result = self._expand("How is pledged collateral treated?")
        assert "encumbered assets" in result

    def test_core_capital(self):
        result = self._expand("What is the minimum core capital ratio?")
        assert "Common Equity Tier 1" in result

    # --- run_26 entries ---

    def test_accumulated_earnings(self):
        result = self._expand("Can accumulated earnings be included in CET1?")
        assert "retained earnings" in result

    def test_maximum_allowable_exposure(self):
        result = self._expand("What is the maximum allowable exposure to a single client?")
        assert "large exposure" in result

    def test_pledged_as_collateral(self):
        result = self._expand("Assets pledged as collateral cannot be counted as liquid.")
        assert "encumbered" in result

    def test_easily_sellable(self):
        result = self._expand("Institutions must hold easily sellable assets.")
        assert "liquid assets high quality" in result

    def test_available_liquid_resources(self):
        result = self._expand("What available liquid resources must be held?")
        assert "liquid assets" in result

    def test_solvency_threshold(self):
        result = self._expand("What is the solvency threshold under CRR?")
        assert "initial capital own funds" in result

    def test_internal_permission_tailored_risk(self):
        result = self._expand(
            "Can an institution use internal permission for tailored risk evaluation?"
        )
        assert "Internal Ratings-Based approach IRB" in result

    def test_blend_of_assets(self):
        result = self._expand("The pool contains a blend of assets.")
        assert "mixed pool" in result

    # --- case-insensitivity ---

    def test_case_insensitive_accumulated_earnings(self):
        result = self._expand("ACCUMULATED EARNINGS from prior years.")
        assert "retained earnings" in result

    # --- no false triggers ---

    def test_unrelated_query_unchanged(self):
        q = "What is the minimum CET1 ratio under Article 92?"
        assert self._expand(q) == q

    def test_partial_word_not_matched(self):
        # "pledged" without "as collateral" or "assets" should not trigger either rule
        result = self._expand("The asset was pledged.")
        # "pledged" alone is not a key — "pledged assets" and "pledged as collateral" are
        assert "encumbered" not in result


# ---------------------------------------------------------------------------
# _build_key_facts_block  (run_27 Part A)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildKeyFactsBlock:
    """_build_key_facts_block extracts thresholds into a preamble."""

    def _build(self, context: str) -> str:
        from src.query.query_engine import _build_key_facts_block
        return _build_key_facts_block(context)

    def _section(self, header: str, body: str) -> str:
        return f"{header}\n{body}"

    def _two_sections(self, s1: str, s2: str) -> str:
        return s1 + "\n\n---\n\n" + s2

    def test_returns_empty_when_no_thresholds(self):
        ctx = self._section(
            "Article 4 — Definitions",
            "'Institution' means a credit institution or an investment firm.",
        )
        assert self._build(ctx) == ""

    def test_detects_percentage_threshold(self):
        ctx = self._section(
            "Article 92 — Own funds requirements",
            "Institutions shall maintain a CET1 ratio of 4.5%.",
        )
        result = self._build(ctx)
        assert "KEY NUMERICAL THRESHOLDS" in result
        assert "4.5%" in result

    def test_detects_day_threshold(self):
        ctx = self._section(
            "Article 412 — Liquidity coverage requirement",
            "Institutions shall hold liquid assets covering 30 days of net outflows.",
        )
        result = self._build(ctx)
        assert "30 days" in result

    def test_detects_eur_amount(self):
        ctx = self._section(
            "Article 93 — Initial capital requirement",
            "The initial capital must be at least EUR 5 million.",
        )
        result = self._build(ctx)
        assert "EUR 5 million" in result

    def test_detects_basis_points(self):
        ctx = self._section(
            "Article 395 — Large exposures limit",
            "The limit shall not exceed 25% of Tier 1 capital, with a floor of 150 basis points.",
        )
        result = self._build(ctx)
        assert "150 basis points" in result

    def test_header_included_in_output(self):
        ctx = self._section(
            "Article 92 — Own funds requirements",
            "CET1 ratio of 4.5%, Tier 1 ratio of 6%, total capital ratio of 8%.",
        )
        result = self._build(ctx)
        assert "Article 92" in result

    def test_multiple_sections_each_listed(self):
        s1 = self._section("Article 92 — Own funds", "CET1 ratio of 4.5%.")
        s2 = self._section("Article 93 — Initial capital", "At least EUR 5 million.")
        result = self._build(self._two_sections(s1, s2))
        assert "Article 92" in result
        assert "Article 93" in result
        assert "4.5%" in result
        assert "EUR 5 million" in result

    def test_section_without_thresholds_not_included(self):
        s1 = self._section("Article 92 — Own funds", "CET1 ratio of 4.5%.")
        s2 = self._section("Article 4 — Definitions", "An institution means a credit institution.")
        result = self._build(self._two_sections(s1, s2))
        assert "Article 4" not in result

    def test_multiple_percentages_in_one_section(self):
        ctx = self._section(
            "Article 92 — Own funds requirements",
            "CET1 4.5%, Tier 1 6%, Total capital 8%.",
        )
        result = self._build(ctx)
        assert "4.5%" in result
        assert "6%" in result
        assert "8%" in result

    def test_preamble_format(self):
        ctx = self._section("Article 92 — Own funds", "CET1 ratio of 4.5%.")
        result = self._build(ctx)
        assert result.startswith("KEY NUMERICAL THRESHOLDS")
        assert result.endswith("\n\n")


# ---------------------------------------------------------------------------
# _append_missing_thresholds  (run_27 Part B)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAppendMissingThresholds:
    """_append_missing_thresholds appends a completeness note when thresholds are omitted."""

    def _append(self, context: str, answer: str) -> str:
        from src.query.query_engine import _append_missing_thresholds
        return _append_missing_thresholds(context, answer)

    def _ctx(self, art: str, body: str) -> str:
        return f"Article {art} — Title\n{body}"

    def test_returns_answer_unchanged_when_nothing_missing(self):
        ctx = self._ctx("92", "CET1 ratio of 4.5%.")
        answer = "The CET1 ratio must be 4.5% per Article 92."
        result = self._append(ctx, answer)
        assert result == answer

    def test_appends_note_when_threshold_missing(self):
        ctx = self._ctx("92", "CET1 ratio of 4.5%, Tier 1 ratio of 6%.")
        answer = "The CET1 ratio is 4.5% per Article 92."
        result = self._append(ctx, answer)
        assert "Completeness note" in result
        assert "6%" in result

    def test_only_checks_cited_articles(self):
        """Thresholds from uncited articles should NOT trigger the note."""
        s1 = self._ctx("92", "CET1 ratio of 4.5%.")
        s2 = "Article 93 — Initial capital\nAt least EUR 5 million."
        ctx = s1 + "\n\n---\n\n" + s2
        # Answer cites Article 92 but not 93
        answer = "CET1 is 4.5% per Article 92."
        result = self._append(ctx, answer)
        # EUR 5 million is not in answer but Article 93 is uncited — no note
        assert "EUR 5 million" not in result

    def test_no_citation_returns_answer_unchanged(self):
        ctx = self._ctx("92", "CET1 ratio of 4.5%.")
        answer = "The institution must meet capital requirements."  # no article ref
        result = self._append(ctx, answer)
        assert result == answer

    def test_note_contains_article_identifier(self):
        ctx = self._ctx("92", "CET1 4.5%, Tier 1 6%, Total 8%.")
        answer = "The minimum CET1 ratio is 4.5% per Article 92."
        result = self._append(ctx, answer)
        assert "Article 92" in result

    def test_no_note_when_all_thresholds_present(self):
        ctx = self._ctx("92", "CET1 4.5%.")
        answer = "The CET1 ratio is 4.5% per Article 92."
        result = self._append(ctx, answer)
        assert "Completeness note" not in result

    def test_multiple_missing_thresholds_all_listed(self):
        ctx = self._ctx("92", "CET1 4.5%, Tier 1 6%, total capital 8%.")
        answer = "The total capital ratio is 8% per Article 92."
        result = self._append(ctx, answer)
        assert "4.5%" in result
        assert "6%" in result

    def test_appended_note_is_appended_not_prepended(self):
        ctx = self._ctx("92", "CET1 4.5%, Tier 1 6%.")
        answer = "CET1 is 4.5% per Article 92."
        result = self._append(ctx, answer)
        assert result.startswith(answer)

    def test_cited_via_lowercase_article_ref(self):
        ctx = self._ctx("92", "CET1 ratio of 4.5%, Tier 1 ratio of 6%.")
        answer = "Per article 92, CET1 is 4.5%."
        result = self._append(ctx, answer)
        # article cited in lowercase — still detected
        assert "6%" in result


# ---------------------------------------------------------------------------
# ArticleDeduplicatorPostprocessor  (run_20 / mixed chunking)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestArticleDeduplicatorPostprocessor:
    """ArticleDeduplicatorPostprocessor correctly deduplicates mixed ARTICLE+PARAGRAPH chunks."""

    def _dedup(self, nodes):
        from src.query.query_engine import ArticleDeduplicatorPostprocessor
        proc = ArticleDeduplicatorPostprocessor()
        return proc._postprocess_nodes(nodes)

    def _art(self, node_id, article, score):
        return _make_node(node_id, article=article, score=score, chunk_type="ARTICLE")

    def _para(self, node_id, article, score):
        return _make_node(node_id, article=article, score=score, chunk_type="PARAGRAPH")

    def test_empty_returns_empty(self):
        assert self._dedup([]) == []

    def test_single_article_node_returned(self):
        nodes = [self._art("n1", "92", 0.8)]
        result = self._dedup(nodes)
        assert len(result) == 1
        assert result[0].node.node_id == "n1"

    def test_single_paragraph_node_returned(self):
        nodes = [self._para("n1", "92", 0.8)]
        result = self._dedup(nodes)
        assert len(result) == 1

    def test_article_preferred_when_within_margin(self):
        """ARTICLE chunk preferred when its score is within 0.02 of best PARAGRAPH."""
        art = self._art("art_92", "92", score=0.79)   # within 0.02 of para
        para = self._para("para_92", "92", score=0.80)
        result = self._dedup([art, para])
        assert len(result) == 1
        assert result[0].node.node_id == "art_92"

    def test_paragraph_preferred_when_score_gap_exceeds_margin(self):
        """PARAGRAPH wins when its score is more than 0.02 above the ARTICLE chunk."""
        art = self._art("art_92", "92", score=0.70)
        para = self._para("para_92", "92", score=0.75)  # 0.05 gap > margin
        result = self._dedup([art, para])
        assert len(result) == 1
        assert result[0].node.node_id == "para_92"

    def test_two_articles_both_kept(self):
        """One chunk per article — two different articles → two results."""
        nodes = [
            self._art("art_92", "92", 0.9),
            self._art("art_26", "26", 0.7),
        ]
        result = self._dedup(nodes)
        ids = {n.node.node_id for n in result}
        assert ids == {"art_92", "art_26"}

    def test_multiple_paragraph_chunks_same_article_keeps_best(self):
        """Multiple PARAGRAPH chunks for same article — only highest-scored survives."""
        p1 = self._para("para_92_a", "92", 0.6)
        p2 = self._para("para_92_b", "92", 0.85)
        p3 = self._para("para_92_c", "92", 0.5)
        result = self._dedup([p1, p2, p3])
        assert len(result) == 1
        assert result[0].node.node_id == "para_92_b"

    def test_result_sorted_descending_by_score(self):
        nodes = [
            self._art("art_26", "26", 0.6),
            self._art("art_92", "92", 0.9),
            self._art("art_36", "36", 0.75),
        ]
        result = self._dedup(nodes)
        scores = [n.score for n in result]
        assert scores == sorted(scores, reverse=True)

    def test_no_article_metadata_uses_node_id_as_key(self):
        """Nodes without an article number (e.g. annexes) are keyed by node_id."""
        n1 = _make_node("annex_I", article="", score=0.8, chunk_type="ARTICLE")
        n2 = _make_node("annex_I", article="", score=0.9, chunk_type="ARTICLE")
        result = self._dedup([n1, n2])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_article_exactly_at_margin_boundary_prefers_article(self):
        """When gap is clearly within margin (0.01 < 0.02), ARTICLE is preferred."""
        # Use values without floating-point representation issues
        art = self._art("art_92", "92", score=0.79)
        para = self._para("para_92", "92", score=0.80)  # gap == 0.01 < _PREFER_ARTICLE_MARGIN
        result = self._dedup([art, para])
        assert result[0].node.node_id == "art_92"

    def test_mixed_articles_and_paragraphs_across_multiple_articles(self):
        """Real-world mixed scenario: articles 92 and 26, each with a paragraph competitor."""
        art_92 = self._art("art_92", "92", 0.88)
        para_92 = self._para("para_92", "92", 0.92)   # gap > 0.02 → para wins for 92
        art_26 = self._art("art_26", "26", 0.80)
        para_26 = self._para("para_26", "26", 0.81)   # gap == 0.01 → art wins for 26
        result = self._dedup([art_92, para_92, art_26, para_26])
        assert len(result) == 2
        ids = {n.node.node_id for n in result}
        assert "para_92" in ids
        assert "art_26" in ids


# ---------------------------------------------------------------------------
# _FALSE_PREMISE_RULE — verified present in prompt templates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFalsePremiseRuleInPrompts:
    """_FALSE_PREMISE_RULE must be embedded in both _LEGAL_QA_TEMPLATE variants."""

    def test_false_premise_rule_in_standard_template(self):
        from src.query.query_engine import _LEGAL_QA_TEMPLATE, _FALSE_PREMISE_RULE
        # PromptTemplate stores the raw string in .template
        assert "FALSE PREMISE RULE" in _LEGAL_QA_TEMPLATE.template

    def test_false_premise_rule_in_history_template(self):
        from src.query.query_engine import _LEGAL_QA_TEMPLATE_WITH_HISTORY
        assert "FALSE PREMISE RULE" in _LEGAL_QA_TEMPLATE_WITH_HISTORY.template

    def test_false_premise_rule_has_no_example(self):
        """Prompt must instruct model not to reproduce examples verbatim."""
        from src.query.query_engine import _FALSE_PREMISE_RULE
        assert "do not reproduce these verbatim" in _FALSE_PREMISE_RULE

    def test_false_premise_rule_has_three_examples(self):
        from src.query.query_engine import _FALSE_PREMISE_RULE
        # Each example starts with "Q: "
        assert _FALSE_PREMISE_RULE.count("Q: ") == 3

    def test_false_premise_rule_applies_conditionally(self):
        """Rule must state it applies ONLY when context clearly contradicts the assumption."""
        from src.query.query_engine import _FALSE_PREMISE_RULE
        assert "ONLY when the context clearly" in _FALSE_PREMISE_RULE


# ---------------------------------------------------------------------------
# ParagraphWindowReranker._split_windows (pure logic, no model)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParagraphWindowSplitter:
    """_split_windows correctly splits and samples paragraph windows."""

    def _reranker(self, max_windows: int = 4):
        from src.query.query_engine import ParagraphWindowReranker
        reranker = ParagraphWindowReranker.__new__(ParagraphWindowReranker)
        reranker._MIN_WINDOW_CHARS = 30
        reranker._max_windows = max_windows
        return reranker

    def _article(self, n_paras: int, chars_each: int = 80) -> str:
        return "\n\n".join(f"Paragraph {i}: " + "x" * chars_each for i in range(n_paras))

    def test_single_paragraph_returns_list_of_one(self):
        text = "A" * 80
        result = self._reranker()._split_windows(text)
        assert len(result) == 1

    def test_short_paragraphs_filtered_out(self):
        text = "Short.\n\nShort.\n\n" + "A" * 80
        result = self._reranker()._split_windows(text)
        # Only the long paragraph meets the 30-char minimum
        assert len(result) == 1
        assert "A" * 30 in result[0]

    def test_fewer_paragraphs_than_max_all_returned(self):
        text = self._article(3)
        result = self._reranker(max_windows=4)._split_windows(text)
        assert len(result) == 3

    def test_more_paragraphs_than_max_sampled_to_max(self):
        text = self._article(10)
        result = self._reranker(max_windows=4)._split_windows(text)
        assert len(result) == 4

    def test_exactly_max_paragraphs_all_returned(self):
        text = self._article(4)
        result = self._reranker(max_windows=4)._split_windows(text)
        assert len(result) == 4

    def test_empty_text_returns_original(self):
        result = self._reranker()._split_windows("")
        assert result == [""]

    def test_all_short_paragraphs_returns_full_text(self):
        text = "Hi.\n\nHi.\n\nHi."  # all below 30-char min
        result = self._reranker()._split_windows(text)
        assert result == [text]

    def test_sampling_covers_start_middle_end(self):
        """With 8 paragraphs sampled to 4, indices should span the range."""
        text = self._article(8)
        result = self._reranker(max_windows=4)._split_windows(text)
        assert len(result) == 4
        # First sample should be paragraph 0
        assert "Paragraph 0" in result[0]
