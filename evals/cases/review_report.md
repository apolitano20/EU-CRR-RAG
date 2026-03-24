# Golden Dataset — Automated Review Report

| | |
|---|---|
| Total cases | 173 |
| LLM-checked (article text available) | 173 |
| **Cases needing review** | **24** |
| — High severity | 12 |
| — Medium severity | 13 |
| — Low severity (FYI) | 67 |

## Missing from `expected_articles` (10 cases) — ✅ 8 FIXED, 2 FALSE POSITIVES

8 cases patched in golden_dataset.jsonl (see below). case_123 and case_127 were false positives — `429(7)` and `429a(7)` are paragraph refs already covered by `429`/`429a` in expected_articles.

- **case_011** (Art. 36) `[HIGH]`
  Possibly missing from expected_articles: ['47b','47c']. Article 47c is relevant for understanding the derogation mentioned in Article 36(5). 47b is referenced in condition (c) of the specialised debt restructurer definition — purchased NPEs that are refinanced or renegotiated must qualify as forbearance measures under Art. 47b for the 5% threshold to be met.
- **case_013** (Art. 52) `[HIGH]`
  Possibly missing from expected_articles: ['77','78']. Article 77 sets the general conditions that must always be met before any call, redemption or repurchase — these are a necessary precondition regardless of timing. Article 78(4) is about the specific exception that permits redemption before the five-year minimum — directly answers the question. This is referenced in the answer but missing from expected_articles.
- **case_021** (Art. 71) `[HIGH]`
  Possibly missing from expected_articles: ['66', '79']. Articles 66 and 79 are referenced in Article 71 for deductions and adjustments to Tier 2 capital.
- **case_022** (Art. 71) `[HIGH]`
  Possibly missing from expected_articles: ['66', '79']. Articles 66 and 79 are referenced in Article 71 and are necessary for a complete understanding of the Tier 2 capital determination process.
- **case_045** (Art. 96) `[HIGH]`
  Possibly missing from expected_articles: ['92', '97']. Articles 92 and 97 are referenced in the calculation method but are missing from expected_articles.
- **case_108** (Art. 416) `[HIGH]`
  Possibly missing from expected_articles: ['132', '418']. Articles 132 and 418 are referenced in the answer but missing from expected_articles.
- **case_109** (Art. 416) `[HIGH]`
  Possibly missing from expected_articles: ['460','509']. Article 509 is referenced in the context of criteria for identifying high and extremely high liquidity and credit quality. Article 460 Defines the endpoint — it is the pending uniform definition that triggers the interim regime. Without it, the process described in the question applies. Directly frames the "before a uniform definition is specified" part of the question.
- **case_121** (Art. 428) `[HIGH]`
  Possibly missing from expected_articles: ['427']. The reference answer mentions Article 427(2) for the presentation of items, which is not included in the expected_articles.
- **case_123** (Art. 429) `[HIGH]`
  Possibly missing from expected_articles: ['429(7)']. The reference answer specifically cites Article 429(7)(a)-(b), which is crucial for the answer.
- **case_127** (Art. 429a) `[HIGH]`
  Possibly missing from expected_articles: ['429a(7)']. The reference answer relies on Article 429a(7) for the adjusted leverage ratio requirement, which is not listed in expected_articles.

## Numerical accuracy issues — ✅ RESOLVED

All flags in this section were false positives. The CRR text uses `x %` (space before `%`) while reference answers use `x%`; the reviewer incorrectly treated these as discrepancies. Reviewer prompt updated to treat the two formats as identical.

- **case_011**, **case_084**: false positives (spacing artefact — values are correct)
- **case_161**: genuine error — reference answer has been corrected (Art. 121(3) Table 1: 50% short-term / 75% non-short-term for Grade B institutions vs. 100% flat for unrated corporates under Art. 122(2))

## Reference answer completeness gaps — ✅ ALL RESOLVED

- **case_011** (Art. 36) `[REVIEWED]` Added internal decision-making process requirement (Art 36(5))
- **case_026** (Art. 73) `[REVIEWED]` Added Art 73(2) three conditions for CA permission + resolution authority consultation
- **case_031** (Art. 78) `[REVIEWED]` Added Art 78(4) early redemption conditions (regulatory reclassification / tax treatment change)
- **case_036** (Art. 93) `[REVIEWED]` Added Art 93(5) merger scenario and Art 93(6) CA solvency override
- **case_047** (Art. 97) `[REVIEWED]` Added Art 97(3) provision for firms <1 year (projected overheads from business plan)
- **case_053** (Art. 132) `[REVIEWED]` Added Art 132(3) MDB derogation for sustainable development CIUs
- **case_074** (Art. 395) `[REVIEWED]` Added Art 395(1) reasonable limit cap (≤ 100% T1) when EUR 150m > 25% T1
- **case_083** (Art. 400) `[REVIEWED]` Expanded from 1 to 8 exemptions under Art 400(1), including intl orgs, MDBs, cash/certificate deposits
- **case_092** (Art. 403) `[REVIEWED]` Added Art 403(2) currency mismatch, maturity mismatch and partial coverage treatment
- **case_104** (Art. 414) `[REVIEWED]` Added daily reporting obligation until compliance restored + CA monitoring
- **case_148** (Art. 411, 412) `[REVIEWED]` Reference answer revised to stay strictly grounded in GT; inferred unencumbered requirement removed
- **case_168** (Art. 242) `[REVIEWED]` Added Art 244(2) SRT conditions; expected_articles updated to ['242', '244']

## `citation_type` mismatch (1 cases)

- **case_156** (Art. 132) `[MEDIUM]`
  citation_type is 'article_cited' but no article number found in question.

## Difficulty calibration (67 cases)

- **case_003** (Art. 26) `[LOW]`
  Suggested difficulty: medium. The task involves understanding a single article with multiple subparagraphs, which is more aligned with a medium difficulty level.
- **case_004** (Art. 27) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and cross-referencing Articles 28 and 29.
- **case_006** (Art. 27) `[LOW]`
  Suggested difficulty: medium. The answer is based on a single article lookup without requiring multiple cross-references.
- **case_007** (Art. 28) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and exceptions, making it more complex than 'easy'.
- **case_010** (Art. 36) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple sub-points within Article 36(1), which is more complex than a simple lookup.
- **case_016** (Art. 62) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and cross-references within Article 62, making it more complex than 'easy'.
- **case_019** (Art. 62) `[LOW]`
  Suggested difficulty: medium. The task involves understanding a single exclusion clause in Article 62, which is not overly complex.
- **case_020** (Art. 71, 62, 66, 79) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple articles and their interrelations, which is more complex than 'easy'.
- **case_022** (Art. 71) `[LOW]`
  Suggested difficulty: medium. The task involves understanding cross-references to two additional articles, which is more complex than a single article lookup but not extremely difficult.
- **case_026** (Art. 73) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions for competent authority permission, making it more complex than 'easy'.
- **case_028** (Art. 73) `[LOW]`
  Suggested difficulty: medium. The answer is based on a single article (Article 73) without requiring multiple cross-references.
- **case_030** (Art. 78) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and exceptions within Article 78, which is more complex than 'easy'.
- **case_034** (Art. 92) `[LOW]`
  Suggested difficulty: medium. The answer primarily relies on a single article (Article 92) with references to external directives, making it less complex than a 'hard' difficulty.
- **case_037** (Art. 93) `[LOW]`
  Suggested difficulty: medium. The question involves straightforward application of Article 93's paragraphs without complex cross-referencing.
- **case_038** (Art. 94) `[LOW]`
  Suggested difficulty: medium. The question requires understanding multiple conditions and periodic assessment, which is more complex than a simple lookup.
- **case_039** (Art. 94) `[LOW]`
  Suggested difficulty: medium. The answer involves straightforward procedural steps from a single article, making it less complex than 'hard'.
- **case_041** (Art. 95) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding the specific calculation method in Article 95(2), which involves multiple cross-references to other articles.
- **case_044** (Art. 96) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding specific conditions and cross-referencing with Directive 2013/36/EU, which adds complexity.
- **case_046** (Art. 96) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding Article 96(3) and its reference to Directive 2013/36/EU, which is a straightforward lookup.
- **case_049** (Art. 97) `[LOW]`
  Suggested difficulty: medium. The answer is based on a single article (Article 97) without requiring multiple cross-references.
- **case_051** (Art. 98) `[LOW]`
  Suggested difficulty: medium. The task involves understanding and comparing two specific sub-articles within Article 98, which is less complex than a typical 'hard' difficulty multi-hop question.
- **case_056** (Art. 132c) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding specific conditions in paragraph 3, making it more complex than a simple lookup.
- **case_064** (Art. 157) `[LOW]`
  Suggested difficulty: medium. The task involves understanding a single article with multiple paragraphs, which is more aligned with a medium difficulty level.
- **case_067** (Art. 392) `[LOW]`
  Suggested difficulty: easy. The task involves a straightforward lookup of a single article.
- **case_070** (Art. 393) `[LOW]`
  Suggested difficulty: easy. The answer is a straightforward application of Article 393, which is a single sentence lookup.
- **case_071** (Art. 394) `[LOW]`
  Suggested difficulty: medium. The question involves understanding multiple reporting requirements and exemptions, which is more complex than a simple definition lookup.
- **case_073** (Art. 394) `[LOW]`
  Suggested difficulty: medium. The task involves understanding and summarizing a single article with clear sub-sections, which is less complex than a 'hard' difficulty level.
- **case_078** (Art. 396) `[LOW]`
  Suggested difficulty: medium. The task involves understanding multiple conditions and guidelines, which is more complex than a simple lookup.
- **case_079** (Art. 396) `[LOW]`
  Suggested difficulty: medium. The question requires understanding the application of waivers under Articles 7(1) and 9, which involves cross-referencing multiple articles.
- **case_080** (Art. 397) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding and applying multiple parts of Article 397, including specific-risk requirements and additional own funds requirements, which is more complex than a simple definition.
- **case_081** (Art. 397) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding the application of specific-risk requirements and cross-referencing multiple parts of the regulation.
- **case_082** (Art. 397) `[LOW]`
  Suggested difficulty: medium. The task involves a straightforward lookup of Article 397 and understanding a single table, which is less complex than 'hard'.
- **case_089** (Art. 402) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and cross-references, which is more complex than 'easy'.
- **case_091** (Art. 402) `[LOW]`
  Suggested difficulty: medium. The question involves comparing two sections of the same article, which is less complex than a typical 'hard' difficulty multi-hop question.
- **case_092** (Art. 403) `[LOW]`
  Suggested difficulty: medium. The question involves understanding multiple conditions and cross-references within Article 403, making it more complex than 'easy'.
- **case_094** (Art. 403) `[LOW]`
  Suggested difficulty: medium. The answer is based on a single article (Article 403) without requiring multiple cross-references.
- **case_097** (Art. 411) `[LOW]`
  Suggested difficulty: medium. The answer primarily relies on definitions from Article 411 without requiring complex cross-referencing.
- **case_100** (Art. 412) `[LOW]`
  Suggested difficulty: medium. The question primarily involves understanding Article 412 and its interaction with Article 460, which is a two-step process rather than a complex multi-hop.
- **case_104** (Art. 414) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and ongoing reporting requirements, which is more complex than a simple lookup.
- **case_106** (Art. 414) `[LOW]`
  Suggested difficulty: medium. The task involves understanding a single article (Article 414) without requiring multiple cross-references.
- **case_107** (Art. 416) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple conditions and exceptions within Article 416(2), which is more complex than a simple lookup.
- **case_112** (Art. 417) `[LOW]`
  Suggested difficulty: medium. The answer is based on a single condition from Article 417, making it less complex than 'hard'.
- **case_116** (Art. 425) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding multiple exceptions and conditions, which is more complex than a simple lookup.
- **case_118** (Art. 425) `[LOW]`
  Suggested difficulty: medium. The procedural steps and conditions are detailed in a single article (Article 425), making it less complex than 'hard'.
- **case_119** (Art. 428) `[LOW]`
  Suggested difficulty: medium. The question requires identifying specific sub-points within Article 428, which involves more than a simple lookup.
- **case_126** (Art. 429a) `[LOW]`
  Suggested difficulty: medium. The definition involves multiple conditions and cross-references, making it more complex than 'easy'.
- **case_129** (Art. 429b) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding and applying conditions from a single article (429b), which is not overly complex.
- **case_130** (Art. 429b) `[LOW]`
  Suggested difficulty: medium. The definition requires understanding conditions in multiple paragraphs, making it more complex than a simple lookup.
- **case_131** (Art. 430) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding exemptions in Article 6(5) and their application to Article 430, which involves cross-referencing.
- **case_133** (Art. 430) `[LOW]`
  Suggested difficulty: medium. The answer primarily relies on information from Article 430(6) and 430(8), which are directly provided in the text, making it less complex than 'hard'.
- **case_135** (Art. 52) `[LOW]`
  Suggested difficulty: medium. The question requires understanding multiple conditions from a single article, which is more aligned with a medium difficulty level.
- **case_136** (Art. 26, 62) `[LOW]`
  Suggested difficulty: medium. The question involves understanding specific conditions from two articles, which is more aligned with a medium difficulty level.
- **case_137** (Art. 26, 62, 92) `[LOW]`
  Suggested difficulty: medium. The question involves identifying components of capital tiers, which is a straightforward lookup in the provided articles.
- **case_139** (Art. 92, 93) `[LOW]`
  Suggested difficulty: medium. The question involves understanding two articles and their implications, but it does not require extensive cross-referencing or complex interpretation.
- **case_142** (Art. 93) `[LOW]`
  Suggested difficulty: medium. The question requires understanding a single article (Article 93) without needing complex cross-references.
- **case_144** (Art. 412, 413) `[LOW]`
  Suggested difficulty: medium. The question involves distinguishing between two specific articles and their requirements, which is more straightforward than a 'hard' difficulty level.
- **case_145** (Art. 412) `[LOW]`
  Suggested difficulty: easy. The answer is directly supported by a single sentence in Article 412(2), making it straightforward.
- **case_147** (Art. 411) `[LOW]`
  Suggested difficulty: medium. The question requires understanding the definitions in Article 411, which is not overly complex.
- **case_149** (Art. 392) `[LOW]`
  Suggested difficulty: easy. The answer is a straightforward lookup of a single article and threshold.
- **case_150** (Art. 395, 396) `[LOW]`
  Suggested difficulty: medium. The question involves understanding exceptions to a rule, which is not as complex as a 'hard' difficulty level suggests.
- **case_152** (Art. 393) `[LOW]`
  Suggested difficulty: easy. The answer is a direct lookup from Article 393, which is straightforward.
- **case_153** (Art. 396) `[LOW]`
  Suggested difficulty: medium. The answer requires understanding a single article (Article 396) rather than multiple cross-references.
- **case_154** (Art. 132, 132a) `[LOW]`
  Suggested difficulty: medium. The question primarily involves understanding a specific condition in Article 132(3)(c)(iii) and does not require extensive cross-referencing.
- **case_159** (Art. 115) `[LOW]`
  Suggested difficulty: medium. The answer is a straightforward lookup from Article 115(1) and does not require complex cross-referencing.
- **case_160** (Art. 121) `[LOW]`
  Suggested difficulty: medium. The task involves understanding a specific condition from a single article, which is less complex than 'hard'.
- **case_162** (Art. 115) `[LOW]`
  Suggested difficulty: medium. The question involves understanding specific conditions under Article 115, which is not overly complex.
- **case_164** (Art. 258, 242) `[LOW]`
  Suggested difficulty: medium. The question requires understanding the threshold condition from Article 258(1)(a) and the definition of 'mixed pool' from Article 242(8), which is a two-step process but not overly complex.

## Coverage distribution

**By category**

| Category | n |
|---|---|
| large_exposures | 35 |
| own_funds | 31 |
| capital_ratios | 31 |
| liquidity | 27 |
| known_failures | 12 |
| leverage | 12 |
| liquidity_lcr | 5 |
| ciu_treatment | 5 |
| credit_risk_sa | 5 |
| securitisation_methods | 3 |
| leverage_ratio_total_exposure | 3 |
| STS_vs_nonSTS | 1 |
| significant_risk_transfer | 1 |
| leverage_ratio_derivatives | 1 |
| leverage_ratio_cash_pooling | 1 |

**By difficulty**

| Difficulty | n |
|---|---|
| easy | 47 |
| medium | 44 |
| hard | 82 |

**By question type**

| Question type | n |
|---|---|
| multi_hop | 48 |
| procedural | 30 |
| threshold | 28 |
| definition | 26 |
| negative | 16 |
| false_friend | 14 |
| diluted_embedding | 6 |
| multi_article | 3 |
| ambiguous | 2 |

**By citation type**

| Citation type | n |
|---|---|
| article_cited | 100 |
| open_ended | 73 |
