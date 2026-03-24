# Golden Dataset — Manual Review Priorities

Goal: sanity-check the reference answers for correctness and completeness.
Risk areas: wrong numbers, missing details, incomplete synthesis, miscalibrated difficulty labels.

## Already Updated
- **case_003** (Art 26(3)) — three-month EBA deviation detail added; plain prose ✅
- **case_031** (Art 78) — quantitative limits, renewal clause added; plain prose ✅

---

## Highest Risk: Reference Answer Likely Incomplete or Wrong

### Hard multi_hop — synthesise multiple paragraphs, most error-prone

| Case | Article | Specific Risk |
|---|---|---|
| case_003 | 26(3) | 5 subparagraphs synthesized — **already fixed** ✅ |
| case_011 | 36(5) | 6-point specialised debt restructurer definition |
| case_031 | 78 | 5 paragraphs; lots of cross-refs — **already fixed** ✅ |
| case_034 | 92(1a) | G-SII buffer formula — number-sensitive |
| case_037 | 93 | 4 scenarios across paragraphs 2, 4, 5, 6 |
| case_039 | 94 | 3-month / 6-of-12 months logic |
| case_055 | 132 | Multi-level CIU — `known_failures` + hard, likely auto-generated |
| case_061 | 153 | Specialised lending + EBA RTS — `known_failures` + hard |
| case_064 | 157 | Full dilution risk process — `known_failures` + hard |
| case_082 | 397 | Table 1 factors (200%→900%) — easy to transcribe wrong |
| case_091 | 402 | Residential vs commercial property — easy to flip conditions |
| case_094 | 403(3) | 5 conditions for tri-party agent treatment |
| case_097 | 411 | Level 1/2 assets + liquidity buffer — 3 definitions cross-referenced |

### Threshold / number precision — exact figures easy to copy wrong

| Case | Article | Check |
|---|---|---|
| case_029 | 78 | 3% of issue / 10% of surplus CET1 — both figures |
| case_033 | 92 | 72.5% output floor / TREA formula |
| case_038 | 94 | 5% of total assets AND EUR 50m — both thresholds |
| case_056 | 132c | 20% conversion factor |
| case_059 | 153 | ×1.25 correlation multiplier |
| case_074 | 395 | 25% OR EUR 150m |
| case_076 | 395(5) | 500% (≤10 days) / 600% (aggregate) |

---

## `expected_articles` Completeness

Cases where the reference answer cites articles not listed in `expected_articles` — the field may be too narrow for multi-article questions:

| Case | Listed | Also referenced in answer |
|---|---|---|
| case_021, case_022 | ["71"] | Arts 66 and 79 |
| case_031 | ["78"] | Arts 77, 79 |
| case_088 | ["401", "399", "403"] | Already correct ✅ |

Worth scanning all hard/multi_hop cases for similar gaps.

---

## Lower Priority (spot check only)

Easy `definition` and single-paragraph cases — typically direct quotes, unlikely to be wrong:
- case_001, case_020, case_023, case_032, case_065, case_068, case_072, case_098
