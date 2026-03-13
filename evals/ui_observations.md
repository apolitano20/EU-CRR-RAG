# UI Observations

This document logs UI / UX issues discovered while manually testing the CRR RAG system.

The goal is to capture problems that affect usability, trust, and navigation of the regulation.


---

# Navigation

### Potentially Irrelevant sources appear in the "Sources" section (IMPORTANT: CHECK HOW THE SOURCES ARE ATTRIBUTED. I AM NOT SURE THEY ARE INDEED IRRELEVANT.)

Example (Case 001):

Sources listed:
- Art. 272 — Definizioni
- Art. 325r — Sensibilità al rischio delta
- Art. 73 — Distributions on instruments
- Art. 10 — Waiver for credit institutions permanently affiliated to a central body

Only **Article 73** is relevant to the question.

**Impact**
- Reduces trust in the system
- Makes the user think the system is citing unrelated provisions

**Suggested improvement**
- Limit displayed sources to the **top 1–3 most relevant articles**
- Introduce a reranker to filter irrelevant retrieved articles


### Sources in mixed languages 
Sources are listed in a mix of Italian and English:
- Art. 272 — Definizioni
- Art. 325r — Sensibilità al rischio delta
- Art. 73 — Distributions on instruments
- Art. 10 — Waiver for credit institutions permanently affiliated to a central body

---

# Article Viewer

### Formatting in the Article viewer (IMPORTANT!)
- The right panel with the article viewer is a huge block of text. The formatting is totally lost (no bullet points, indentation etc.). This makes reading the article incredibly challenging. The formatting in the Article viewer should be preserved. 

### Relevant paragraphs are not highlighted

When the user clicks a cited article, the full article appears in the viewer, but the relevant paragraph is not highlighted.

Example:

For Article 73, the system refers mainly to paragraphs:
- 73(1)
- 73(2)

But the viewer shows the entire article with no indication of where the relevant information is.

**Impact**
- Slows down verification
- Makes the user scan the entire article manually

**Suggested improvement**
- Highlight the paragraph(s) used in the answer
- Scroll automatically to the relevant paragraph


---

# Citations

### Article title sometimes slightly incorrect

Example:

The UI shows:

"Distributions on instruments"

Actual CRR title:

"Distributions on own funds instruments"

**Impact**
- Small legal imprecision
- May reduce confidence in the system

**Suggested improvement**
- Always display the **exact article title from the CRR text**
- Avoid paraphrasing titles

### Order of citations is counterintuitive
- When I ask a question e.g. about Article 93, the first Source mentioned is Article 92a. This is counterintuitive, the first source mentioned should be Article 93 itself. 

---

# Source Display

### Source labeling is unclear

Current UI shows:

"Article References Article 73"

This wording is confusing.

**Suggested improvement**

Use clearer labels such as:

- Primary Source
- Supporting Articles
- Cross References

### Every new query clean the window. 
- When I ask a second question, I see the right panel static (still opened to the previous article) but the left panel shows a totally new answer. Ideally the system should show previous answers to, something like a modern LLM (e.g. GPT or Claude) interface/ 

---

# Cross-reference navigation

### Article references inside the text are not interactive

Example references such as:

"Article 4(1)"

are not clickable in the article viewer.

**Impact**
- Harder to navigate the regulation
- Users must manually search for referenced provisions

**Suggested improvement**
- Convert all article references to clickable links
- Opening them should display the article in the viewer

### Irrelevant Hyperlinks
The Article of the CRR mentions Articles of EXTERNAL regulations. For instance, the ending of  Article 73 reads: "Power is delegated to the Commission to adopt the regulatory technical standards referred to in the first subparagraph in accordance with Articles 10 to 14 of Regulation (EU) No 1093/2010.". In this example, "Article 10" is clickable hyperlink, while "Article 14" is not. Either ways, the issue is that these two Articles (10 and 14) are not of the CRR, but of (EU) No 1093/2010, which is a different regulation. 

### Hierarchical sections not navigable
- On top of the right panel I see CRR > Part II > Title I > Chapter 
 > Article 73. If I click Chapter 6, I would expect the system to show me all Articles under Chapter 6. However, the button is not clickable, so I can't navigate back and forth in the interface. 
---

# Preview behavior

### No preview when hovering over references

Users must click references to see the article.

**Suggested improvement**

Add hover previews for references such as:

Article 4(1)

Hover should show:
- article title
- first paragraph
- quick "Open article" button


---

# Formatting

### Bullet lists in answers sometimes inconsistent

Some responses produce uneven bullet formatting or inconsistent indentation.

**Impact**
- Slightly harder to read long summaries

**Suggested improvement**
- Standardize bullet formatting in the answer template


---

# Observations

### The article viewer + chat layout works well

The two-panel layout (answer + article viewer) allows immediate verification of the AI output.

This is **very helpful for legal content** and should be preserved.

Benefits observed:
- Easy comparison between summary and legal text
- Fast detection of hallucinations
- Good transparency of sources


---

# Future UX Improvements

Ideas worth exploring later:

- Highlight referenced paragraphs in the article viewer
- Hover preview for article references
- Better source ranking (top sources first)
- Clickable cross-reference navigation
- Optional article summaries for chapters/titles