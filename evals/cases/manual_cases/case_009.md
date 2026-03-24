# Case 009

## Query

An institution has an off-balance-sheet exposure that may convert into units of a collective investment undertaking (CIU). The exposure value (as calculated under the general exposure framework) is €10 million.

Two cases:

Case A: The exposure qualifies for one of the specialised CIU approaches

Case B: The exposure does not qualify for any specialised approach

Question:
Explain how the risk-weighted exposure amount should be calculated in each case, highlighting:

What inputs are required

How the treatment differs conceptually

What role the risk weight plays in each case

## Answer (AI)

**Direct Answer**  
In Case A, where the exposure qualifies for a specialized CIU approach, the risk-weighted exposure amount is calculated using either the look-through approach or the mandate-based approach, applying the appropriate risk weight based on the underlying exposures of the CIU. In Case B, where the exposure does not qualify for any specialized approach, the institution must use the standard treatment, assigning a risk weight of 1,250% to the exposure.

**Key Provisions**  
- Article 132(1): Institutions calculate the risk-weighted exposure amount for exposures in the form of units or shares in a CIU by multiplying the risk-weighted exposure amount of the CIU by the percentage of units or shares held.  
- Article 132(2): If conditions are met, institutions may apply the look-through approach or the mandate-based approach.  
- Article 132(3): Standard treatment applies when an institution does not qualify for the specialized approaches, resulting in a risk weight of 1,250%.  

**Conditions, Exceptions & Definitions**  
- Case A applies if the CIU meets the criteria set out in Article 132(3).  
- Case B results when the CIU does not meet the specified conditions for specialized approaches.

**Article References**  
Article 132, Article 113

## Sources

- **Article 132c** — Treatment of off-balance-sheet exposures to CIUs (score: 1.000)
- **Article 152** — Treatment of exposures in the form of units or shares in CIUs (score: 0.674)
- **Article 132a** — Approaches for calculating risk-weighted exposure amounts of CIUs (score: 0.498)
- **Article 132** — Own funds requirements for exposures in the form of units or shares in CIUs (score: 0.480)
- **Article 235** — Calculating risk-weighted exposure amounts under the substitution approach where the guaranteed exposure is treated under the Standardised Approach (score: 0.424)
- **Article 113** — Calculation of risk-weighted exposure amounts (score: 0.326)
- **Article 3** — Application of stricter requirements by institutions (score: 0.000)
- **Article 22** — Sub-consolidation in the case of entities in third countries (score: 0.000)
- **Article 50** — Common Equity Tier 1 capital (score: 0.000)

## Article Viewed in Viewer

**Article 132c – Treatment of off-balance-sheet exposures to CIUs**

1. Institutions shall calculate the risk-weighted exposure amount for their off-balance-sheet items with the potential to be converted into exposures in the form of units or shares in a CIU by multiplying the exposure values of those exposures calculated in accordance with Article 111, with the following risk weight:
  (a) for all exposures for which institutions use one of the approaches set out in Article 132a: where:
  (b) for all other exposures, .
2.
Institutions shall calculate the risk-weighted exposure amount for off-balance-sheet exposures arising from minimum value commitments that meet all the conditions set out in paragraph 3 of this Article by multiplying the exposure value of those exposures by a conversion factor of 20 % and the risk weight derived under Article 132 or 152.
3. Institutions shall determine the risk-weighted exposure amount for off-balance-sheet exposures arising from minimum value commitments in accordance with paragraph 2 where all the following conditions are met:
  (a) the off-balance-sheet exposure of the institution is a minimum value commitment for an investment into units or shares of one or more CIUs under which the institution is only obliged to pay out under the minimum value commitment where the market value of the underlying exposures of the CIU or CIUs is below a predetermined threshold at one or more points in time, as specified in the contract;
  (b) the CIU is any of the following:
  (i) a UCITS as defined in Directive 2009/65/EC; or
  (ii) an AIF as defined in point (a) of Article 4(1) of Directive 2011/61/EU which solely invests in transferable securities or in other liquid financial assets referred to in Article 50(1) of Directive 2009/65/EC, where the mandate of the AIF does not allow a leverage higher than that allowed under Article 51(3) of Directive 2009/65/EC;
  (c) the current market value of the underlying exposures of the CIU underlying the minimum value commitment without considering the effect of the off-balance-sheet minimum value commitments covers or exceeds the present value of the threshold specified in the minimum value commitment;
  (d) when the excess of the market value of the underlying exposures of the CIU or CIUs over the present value of the minimum value commitment declines, the institution, or another undertaking in so far as it is covered by the supervision on a consolidated basis to which the institution itself is subject in accordance with this Regulation and Directive 2013/36/EU or Directive 2002/87/EC, can influence the composition of the underlying exposures of the CIU or CIUs or limit the potential for a further reduction of the excess in other ways;
  (e) the ultimate direct or indirect beneficiary of the minimum value commitment is typically a retail client as defined in point (11) of Article 4(1) of Directive 2014/65/EU.;

## Feedback / Notes

# Evaluation Feedback — Q2

## Verdict
Partially Correct

## Summary
The answer correctly identifies the existence of different treatments depending on whether a specialised CIU approach applies. However, it introduces a critical error by assigning a specific risk weight (1,250%) that is not supported by the provided provision and drifts beyond the scope of the question.

## What is correct
- Correct identification of **branching logic**:
  - Case A: specialised CIU approach applies
  - Case B: no specialised approach applies
- Correct high-level understanding that:
  - Exposure value is the starting point
  - A risk weight is applied to derive the risk-weighted exposure amount

## Critical Issues

### 1. Scope Drift
- The question concerns **off-balance-sheet exposures potentially converting into CIU units**
- The answer shifts to **general CIU exposure treatment**, introducing concepts not strictly grounded in the provided text

### 2. Unsupported Precision (Hallucination)
- The statement:
  > "assigning a risk weight of 1,250%"
  is **not supported by the provided provision**
- This introduces external knowledge not explicitly grounded in the text

## Why This Matters
In regulatory contexts:
- Answers must remain **strictly within the scope of the provided legal basis**
- Introducing external assumptions can lead to **incorrect or non-compliant interpretations**

## Suggested Improvement
Replace:
> "assigning a risk weight of 1,250%"

With:
> "applying the risk weight applicable to exposures that do not qualify for the specialised CIU approaches"

Additionally, clarify that:
- The provision specifies the **mechanics (exposure value × risk weight)**
- It does **not explicitly define the fallback risk weight**

## Key Insight
This answer reveals a common failure mode in legal RAG systems:
> When uncertain, the model fills gaps using general domain knowledge instead of staying within the provided text.

The correct behavior should be:
> If a detail is not specified in the provision, explicitly state that it is not specified.
