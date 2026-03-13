"""
Unit tests for src/ingestion/language_config.py
"""
import pytest

from src.ingestion.language_config import (
    DEFAULT_CELEX,
    LANGUAGE_CONFIGS,
    SUPPORTED_LANGUAGES,
    LanguageConfig,
    get_config,
)


# ---------------------------------------------------------------------------
# get_config()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetConfig:
    def test_returns_language_config_instance(self):
        assert isinstance(get_config("en"), LanguageConfig)

    def test_en_config(self):
        cfg = get_config("en")
        assert cfg.language == "en"
        assert cfg.article_keyword == "Article"

    def test_it_config(self):
        cfg = get_config("it")
        assert cfg.language == "it"
        assert cfg.article_keyword == "Articolo"

    def test_pl_config(self):
        cfg = get_config("pl")
        assert cfg.language == "pl"
        assert cfg.article_keyword == "Artykuł"

    def test_unknown_language_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_config("de")

    def test_unknown_language_error_message_lists_supported(self):
        with pytest.raises(ValueError, match="en"):
            get_config("xx")

    def test_case_sensitive_unknown(self):
        """Language codes are lowercase; 'EN' should raise."""
        with pytest.raises(ValueError):
            get_config("EN")

    def test_all_supported_languages_resolvable(self):
        for lang in SUPPORTED_LANGUAGES:
            cfg = get_config(lang)
            assert cfg.language == lang


# ---------------------------------------------------------------------------
# LanguageConfig.build_url()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildUrl:
    def test_en_url_contains_celex(self):
        cfg = get_config("en")
        url = cfg.build_url()
        assert DEFAULT_CELEX in url

    def test_en_url_contains_language_segment(self):
        cfg = get_config("en")
        url = cfg.build_url()
        assert "/EN/" in url

    def test_it_url_contains_language_segment(self):
        cfg = get_config("it")
        url = cfg.build_url()
        assert "/IT/" in url

    def test_pl_url_contains_language_segment(self):
        cfg = get_config("pl")
        url = cfg.build_url()
        assert "/PL/" in url

    def test_custom_celex_substitution(self):
        cfg = get_config("en")
        custom = "02013R0575-20240101"
        url = cfg.build_url(celex=custom)
        assert custom in url
        assert DEFAULT_CELEX not in url

    def test_url_starts_with_https(self):
        for lang in SUPPORTED_LANGUAGES:
            url = get_config(lang).build_url()
            assert url.startswith("https://")

    def test_url_points_to_eurlex(self):
        for lang in SUPPORTED_LANGUAGES:
            url = get_config(lang).build_url()
            assert "eur-lex.europa.eu" in url


# ---------------------------------------------------------------------------
# heading_keywords structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHeadingKeywords:
    REQUIRED_LEVELS = {"PART", "TITLE", "CHAPTER", "SECTION"}

    def test_all_required_levels_present_en(self):
        cfg = get_config("en")
        assert self.REQUIRED_LEVELS <= set(cfg.heading_keywords)

    def test_all_required_levels_present_it(self):
        cfg = get_config("it")
        assert self.REQUIRED_LEVELS <= set(cfg.heading_keywords)

    def test_all_required_levels_present_pl(self):
        cfg = get_config("pl")
        assert self.REQUIRED_LEVELS <= set(cfg.heading_keywords)

    def test_each_level_has_at_least_one_keyword(self):
        for lang in SUPPORTED_LANGUAGES:
            cfg = get_config(lang)
            for level, keywords in cfg.heading_keywords.items():
                assert keywords, f"Empty keyword list for {lang}/{level}"

    def test_section_has_subsection_variant_before_section(self):
        """SUB-SECTION / SOTTOSEZIONE must appear before SECTION / SEZIONE
        so that the regex matching short-circuits correctly."""
        for lang in SUPPORTED_LANGUAGES:
            cfg = get_config(lang)
            kws = cfg.heading_keywords.get("SECTION", [])
            if len(kws) >= 2:
                # The first keyword should be the longer sub-section variant
                assert len(kws[0]) >= len(kws[-1]), (
                    f"{lang}: SECTION keywords should list sub-section variant first; got {kws}"
                )

    def test_configs_are_frozen(self):
        """LanguageConfig is frozen=True — mutation should raise."""
        cfg = get_config("en")
        with pytest.raises((TypeError, AttributeError)):
            cfg.language = "de"  # type: ignore[misc]
