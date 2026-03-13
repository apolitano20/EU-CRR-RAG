"""
Unit tests for src/models/document.py — DocumentNode and NodeLevel.
"""
import pytest

from src.models.document import DocumentNode, NodeLevel


# ---------------------------------------------------------------------------
# NodeLevel
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNodeLevel:
    EXPECTED_VALUES = {"PART", "TITLE", "CHAPTER", "SECTION", "ARTICLE", "ANNEX"}

    def test_all_levels_defined(self):
        actual = {member.value for member in NodeLevel}
        assert actual == self.EXPECTED_VALUES

    def test_paragraph_and_point_removed(self):
        actual = {member.value for member in NodeLevel}
        assert "PARAGRAPH" not in actual
        assert "POINT" not in actual

    def test_annex_level_defined(self):
        assert NodeLevel.ANNEX == "ANNEX"

    def test_is_string_enum(self):
        assert NodeLevel.ARTICLE == "ARTICLE"

    def test_str_comparison(self):
        assert NodeLevel.PART == "PART"


# ---------------------------------------------------------------------------
# DocumentNode.to_metadata()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToMetadata:
    def _node(self, **kwargs) -> DocumentNode:
        defaults = dict(node_id="n1", level=NodeLevel.ARTICLE, text="Some text.")
        return DocumentNode(**{**defaults, **kwargs})

    def test_returns_dict(self):
        assert isinstance(self._node().to_metadata(), dict)

    def test_includes_node_id(self):
        meta = self._node(node_id="abc").to_metadata()
        assert meta["node_id"] == "abc"

    def test_includes_level_value(self):
        meta = self._node(level=NodeLevel.CHAPTER).to_metadata()
        assert meta["level"] == "CHAPTER"

    def test_none_fields_become_empty_string(self):
        node = self._node(part=None, title=None, chapter=None, section=None, article=None)
        meta = node.to_metadata()
        assert meta["part"] == ""
        assert meta["title"] == ""
        assert meta["chapter"] == ""
        assert meta["section"] == ""
        assert meta["article"] == ""

    def test_populated_fields_preserved(self):
        node = self._node(part="ONE", title="I", chapter="2", section="1", article="92")
        meta = node.to_metadata()
        assert meta["part"] == "ONE"
        assert meta["title"] == "I"
        assert meta["chapter"] == "2"
        assert meta["section"] == "1"
        assert meta["article"] == "92"

    def test_new_fields_in_metadata(self):
        node = self._node(
            article_title="Own funds requirements",
            referenced_articles="26,36",
            referenced_external="Directive 2013/36/EU",
            has_table=True,
            has_formula=False,
        )
        meta = node.to_metadata()
        assert meta["article_title"] == "Own funds requirements"
        assert meta["referenced_articles"] == "26,36"
        assert meta["referenced_external"] == "Directive 2013/36/EU"
        assert meta["has_table"] is True
        assert meta["has_formula"] is False

    def test_annex_fields_in_metadata(self):
        node = DocumentNode(
            node_id="anx_I_en",
            level=NodeLevel.ANNEX,
            text="Annex content.",
            annex_id="I",
            annex_title="List of activities",
        )
        meta = node.to_metadata()
        assert meta["annex_id"] == "I"
        assert meta["annex_title"] == "List of activities"
        assert meta["level"] == "ANNEX"

    def test_none_article_title_becomes_empty_string(self):
        node = self._node(article_title=None)
        meta = node.to_metadata()
        assert meta["article_title"] == ""

    def test_none_annex_id_becomes_empty_string(self):
        node = self._node(annex_id=None)
        meta = node.to_metadata()
        assert meta["annex_id"] == ""

    def test_referenced_articles_default_empty_string(self):
        node = self._node()
        meta = node.to_metadata()
        assert meta["referenced_articles"] == ""

    def test_has_table_default_false(self):
        node = self._node()
        meta = node.to_metadata()
        assert meta["has_table"] is False

    def test_extra_metadata_merged(self):
        node = self._node(metadata={"prev_node_id": "n0", "next_node_id": "n2"})
        meta = node.to_metadata()
        assert meta["prev_node_id"] == "n0"
        assert meta["next_node_id"] == "n2"

    def test_extra_metadata_does_not_override_standard_fields(self):
        """Extra metadata should NOT silently overwrite node_id/level etc."""
        node = self._node(node_id="real-id", metadata={"node_id": "injected"})
        meta = node.to_metadata()
        assert "node_id" in meta


# ---------------------------------------------------------------------------
# DocumentNode.citation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCitation:
    def _node(self, **kwargs) -> DocumentNode:
        defaults = dict(node_id="n1", level=NodeLevel.ARTICLE, text="text")
        return DocumentNode(**{**defaults, **kwargs})

    def test_full_citation(self):
        node = self._node(part="ONE", title="I", chapter="2", section="1", article="92")
        assert node.citation == "Part ONE, Title I, Chapter 2, Section 1, Article 92"

    def test_article_only(self):
        node = self._node(article="92")
        assert node.citation == "Article 92"

    def test_part_and_title(self):
        node = self._node(part="III", title="II")
        assert node.citation == "Part III, Title II"

    def test_no_fields_falls_back_to_node_id(self):
        node = self._node(node_id="fallback-id")
        assert node.citation == "fallback-id"

    def test_citation_is_string(self):
        assert isinstance(self._node(article="1").citation, str)

    def test_citation_omits_none_fields(self):
        node = self._node(part="ONE", title=None, article="5")
        citation = node.citation
        assert "Title" not in citation
        assert "Part ONE" in citation
        assert "Article 5" in citation

    def test_annex_citation(self):
        node = DocumentNode(
            node_id="anx_I_en",
            level=NodeLevel.ANNEX,
            text="text",
            annex_id="I",
        )
        assert node.citation == "Annex I"

    def test_annex_citation_takes_priority(self):
        """When annex_id is set, citation shows only the annex, not Part/Title."""
        node = DocumentNode(
            node_id="anx_II_en",
            level=NodeLevel.ANNEX,
            text="text",
            annex_id="II",
            part="THREE",  # should be ignored
        )
        assert node.citation == "Annex II"
        assert "Part" not in node.citation
