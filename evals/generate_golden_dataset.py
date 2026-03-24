"""
Automated Golden Dataset Generator for CRR RAG eval pipeline.

Extracts articles from Qdrant, calls GPT to generate Q&A pairs, outputs JSONL.

Usage:
    python -m evals.generate_golden_dataset                      # full run
    python -m evals.generate_golden_dataset --dry-run            # no GPT calls
    python -m evals.generate_golden_dataset --pass 1             # article-anchored only
    python -m evals.generate_golden_dataset --pass 2             # adversarial only
    python -m evals.generate_golden_dataset --max-articles 5     # limit for testing
    python -m evals.generate_golden_dataset --model gpt-4.1
    python -m evals.generate_golden_dataset --output evals/v2.jsonl
    python -m evals.generate_golden_dataset --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — priority articles per category
# ---------------------------------------------------------------------------

PRIORITY_ARTICLES: dict[str, list[str]] = {
    "own_funds":      ["26", "27", "28", "36", "52", "62", "71", "72", "73", "78"],
    "capital_ratios": ["92", "93", "94", "95", "96", "97", "98"],
    "liquidity":      ["411", "412", "413", "414", "416", "417", "422", "425", "428"],
    "large_exposures":["392", "393", "394", "395", "396", "397", "400", "401", "402", "403"],
    "leverage":       ["429", "429a", "429b", "430"],
    "known_failures": ["132", "132c", "153", "157"],
}

# Article 4 skipped — 60k chars, covered by definitions fast-path
SKIP_ARTICLES = {"4"}

# Adversarial batches — grouped for multi-hop / false-friend questions
ADVERSARIAL_BATCHES: list[dict] = [
    {
        "id": "capital_stack",
        "articles": ["26", "52", "62", "71", "92"],
        "focus": "Distinguish CET1 / AT1 / T2 eligibility conditions and their interaction with the 8 % total capital ratio.",
    },
    {
        "id": "capital_ratios",
        "articles": ["92", "93", "94", "95"],
        "focus": "Stress thresholds, floor mechanisms, and output floor under Article 92(3) vs 93.",
    },
    {
        "id": "liquidity_lcr",
        "articles": ["411", "412", "413", "414"],
        "focus": "Liquidity coverage requirement: inflows, outflows, HQLA definitions and netting.",
    },
    {
        "id": "large_exposures",
        "articles": ["392", "393", "395", "396"],
        "focus": "25 % large exposure limit: scope, exemptions, and what counts as connected clients.",
    },
    {
        "id": "ciu_treatment",
        "articles": ["132", "132a", "132c", "152"],
        "focus": "CIU look-through vs mandate-based vs fallback approach — when each applies.",
    },
    {
        "id": "credit_risk_sa",
        "articles": ["113", "114", "115", "121", "122"],
        "focus": "Standardised approach risk weights: institutions, corporates, retail — conditions for each.",
    },
    {
        "id": "securitisation",
        "articles": ["242", "243", "254", "258"],
        "focus": "STS vs non-STS securitisation treatment, significant risk transfer, and IRB approach.",
    },
    {
        "id": "leverage_ratio",
        "articles": ["429", "429a", "429b", "429c"],
        "focus": "Leverage ratio total exposure measure: on-balance, off-balance, SFT netting, derivatives.",
    },
]

DEFAULT_OUTPUT = Path("evals/cases/golden_dataset.jsonl")
DEFAULT_MODEL_ENV = "GOLDEN_DATASET_MODEL"
DEFAULT_MODEL = "gpt-4.1"
MAX_ARTICLE_CHARS = 12_000


# ---------------------------------------------------------------------------
# Stage 1: Article Extraction
# ---------------------------------------------------------------------------

def extract_text_from_payload(payload: dict) -> str:
    """Extract text from a Qdrant payload, handling both top-level and LlamaIndex blob."""
    text = payload.get("text", "") or ""
    if not text:
        raw = payload.get("_node_content", "")
        if raw:
            try:
                text = json.loads(raw).get("text", "") or ""
            except Exception:
                pass
    return text


def extract_articles(language: str = "en", verbose: bool = False) -> dict[str, dict]:
    """
    Scroll all Qdrant payloads and group by article number.

    Returns:
        article_num → {
            "article": str,
            "text": str,           # concatenated text of all chunks
            "title": str,
            "part": str,
            "title_num": str,
            "chapter": str,
            "section": str,
        }
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.indexing.vector_store import VectorStore

    vs = VectorStore()
    vs.connect()
    logger.info("Scrolling Qdrant payloads (language=%s)…", language)
    payloads = vs.scroll_payloads(language=language)
    logger.info("  Retrieved %d payloads.", len(payloads))

    # Group chunks by article
    article_chunks: dict[str, list[dict]] = defaultdict(list)
    for p in payloads:
        art = str(p.get("article", "")).strip()
        if not art or art in SKIP_ARTICLES:
            continue
        text = extract_text_from_payload(p)
        if not text:
            continue
        article_chunks[art].append(
            {
                "text": text,
                "part": p.get("part", ""),
                "title_num": p.get("title", ""),
                "chapter": p.get("chapter", ""),
                "section": p.get("section", ""),
                "article_title": p.get("article_title", ""),
            }
        )

    # Merge chunks per article
    articles: dict[str, dict] = {}
    for art, chunks in article_chunks.items():
        combined_text = "\n\n".join(c["text"] for c in chunks if c["text"])
        # Pick metadata from first chunk (most complete)
        ref = chunks[0]
        articles[art] = {
            "article": art,
            "text": combined_text,
            "title": ref.get("article_title", ""),
            "part": ref.get("part", ""),
            "title_num": ref.get("title_num", ""),
            "chapter": ref.get("chapter", ""),
            "section": ref.get("section", ""),
        }

    if verbose:
        logger.info("Extracted %d unique articles:", len(articles))
        for art in sorted(articles, key=_sort_key):
            entry = articles[art]
            char_count = len(entry["text"])
            logger.info("  Article %s — %d chars — %s", art, char_count, entry.get("title", ""))

    return articles


def _sort_key(art: str) -> tuple:
    """Sort article numbers numerically (handles '429a', '132c' etc.)."""
    m = re.match(r"^(\d+)([a-z]*)$", art.lower())
    if m:
        return (int(m.group(1)), m.group(2))
    return (999999, art)


def select_priority_articles(
    all_articles: dict[str, dict],
    max_articles: Optional[int] = None,
) -> list[dict]:
    """Return the subset of articles matching PRIORITY_ARTICLES, with category attached."""
    # Build reverse map: article_num → category
    art_to_cat: dict[str, str] = {}
    for cat, arts in PRIORITY_ARTICLES.items():
        for a in arts:
            art_to_cat[a] = cat

    selected = []
    for art_num in sorted(art_to_cat, key=_sort_key):
        if art_num not in all_articles:
            logger.debug("Priority article %s not found in Qdrant — skipping.", art_num)
            continue
        entry = dict(all_articles[art_num])
        entry["category"] = art_to_cat[art_num]
        selected.append(entry)

    if max_articles:
        selected = selected[:max_articles]

    logger.info("Selected %d priority articles for Pass 1.", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Stage 2: GPT client with retry + JSON parsing
# ---------------------------------------------------------------------------

class GPTClient:
    """Thin wrapper around OpenAI chat completions with retry + JSON extraction."""

    def __init__(self, model: str) -> None:
        self.model = model
        import openai
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """Call GPT with exponential backoff. Returns raw text or None on failure."""
        delays = [2, 4, 8, 16, 64]
        last_error: Optional[Exception] = None
        for attempt, delay in enumerate(delays, 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                jitter = random.uniform(0, delay * 0.25)
                wait = delay + jitter
                logger.warning(
                    "GPT call attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt, len(delays), exc, wait,
                )
                time.sleep(wait)
        logger.error("GPT call exhausted retries. Last error: %s", last_error)
        return None

    def call_with_json_fix(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """Call GPT, and if JSON parsing fails, ask it to fix the output once."""
        raw = self.call(system_prompt, user_prompt, temperature, max_tokens)
        if raw is None:
            return None
        # Quick check: does it parse?
        parsed = _parse_json_response(raw)
        if parsed is not None:
            return raw  # already good
        # Ask GPT to fix it
        fix_prompt = (
            "Your previous response was not valid JSON. "
            "Please return ONLY a valid JSON array, no markdown, no commentary.\n\n"
            f"Original response:\n{raw}"
        )
        fixed = self.call(system_prompt, fix_prompt, temperature=0.0, max_tokens=max_tokens)
        return fixed


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()
    # Remove opening fence with optional language tag
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_json_response(raw: str) -> Optional[list[dict]]:
    """
    Parse GPT JSON output robustly.

    Strategy:
    1. Strip markdown code fences.
    2. Try direct json.loads.
    3. Fallback: extract individual {...} objects via regex.
    """
    if not raw:
        return None

    cleaned = _strip_code_fences(raw)

    # Attempt 1: direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Attempt 2: find JSON array with regex
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Attempt 3: extract individual objects
    objects = []
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", cleaned, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    if objects:
        return objects

    return None


# ---------------------------------------------------------------------------
# Stage 3: Pass 1 — Article-Anchored Generation
# ---------------------------------------------------------------------------

PASS1_SYSTEM_PROMPT = """\
You are a senior regulatory expert specialising in EU banking law, specifically \
the Capital Requirements Regulation (CRR — Regulation (EU) No 575/2013).

Your task is to generate evaluation Q&A pairs for a RAG system that answers questions \
about the CRR. Each Q&A pair will be used to test whether the RAG system retrieves \
the correct article and produces an accurate answer.

Requirements:
- Questions must be specific enough that the correct answer can ONLY come from the \
  article provided.
- Vary difficulty: easy (direct lookup), medium (requires reading multiple paragraphs), \
  hard (requires synthesising sub-points or understanding cross-references).
- Vary question_type: use one of: threshold, definition, procedural, multi_hop, \
  multi_article, negative, false_friend, ambiguous, diluted_embedding.
- reference_answer must be factually accurate and cite specific paragraphs/points.
- expected_articles is a JSON array of article number strings (e.g. ["92"] or ["92","93"]).
- notes should mention the specific paragraph(s) that contain the answer.

Output ONLY a JSON array — no markdown, no commentary. Each element:
{
  "question": "...",
  "expected_articles": ["92"],
  "reference_answer": "...",
  "category": "own_funds",
  "difficulty": "easy|medium|hard",
  "question_type": "threshold|definition|procedural|multi_hop|multi_article|negative|false_friend|ambiguous|diluted_embedding",
  "language": "en",
  "notes": "..."
}
"""

def _build_pass1_user_prompt(article: dict) -> str:
    text = article["text"]
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS] + "\n\n[TEXT TRUNCATED]"

    location_parts = []
    if article.get("part"):
        location_parts.append(f"Part: {article['part']}")
    if article.get("title_num"):
        location_parts.append(f"Title: {article['title_num']}")
    if article.get("chapter"):
        location_parts.append(f"Chapter: {article['chapter']}")
    if article.get("section"):
        location_parts.append(f"Section: {article['section']}")
    location_str = " | ".join(location_parts) if location_parts else "Unknown"

    return f"""\
Article number: {article['article']}
Article title: {article.get('title', 'Unknown')}
Location: {location_str}
Category: {article.get('category', 'unknown')}

Full article text:
{text}

Generate 2–3 Q&A pairs covering different aspects of this article at varying difficulty levels. \
Include at least one "hard" question that requires synthesising multiple paragraphs or points. \
Use category="{article.get('category', 'unknown')}" in all entries.
"""


def run_pass1(
    gpt: GPTClient,
    priority_articles: list[dict],
    existing_cases: list[dict],
    output_path: Path,
    id_counter: list[int],  # mutable counter [n]
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Generate article-anchored Q&A pairs. Returns number of new cases added."""
    # Track which articles already have cases (by article number)
    covered_articles: set[str] = set()
    for case in existing_cases:
        for art in case.get("expected_articles", []):
            covered_articles.add(str(art))

    new_cases = 0
    total = len(priority_articles)
    for idx, article in enumerate(priority_articles, 1):
        art_num = article["article"]
        logger.info("[Pass 1] %d/%d — Article %s (%s)", idx, total, art_num, article.get("title", ""))

        if art_num in covered_articles:
            logger.info("  → Already covered, skipping.")
            continue

        if dry_run:
            logger.info("  → [dry-run] Would call GPT for Article %s.", art_num)
            continue

        user_prompt = _build_pass1_user_prompt(article)
        raw = gpt.call_with_json_fix(PASS1_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            logger.warning("  → GPT call failed for Article %s, skipping.", art_num)
            continue

        cases = _parse_json_response(raw)
        if not cases:
            logger.warning("  → Could not parse GPT response for Article %s.", art_num)
            if verbose:
                logger.debug("Raw response:\n%s", raw)
            continue

        added = _append_cases(cases, output_path, id_counter, existing_cases, art_num)
        new_cases += added
        logger.info("  → Added %d case(s) (total so far: %d).", added, id_counter[0] - 1)

    return new_cases


# ---------------------------------------------------------------------------
# Stage 4: Pass 2 — Adversarial Generation
# ---------------------------------------------------------------------------

PASS2_SYSTEM_PROMPT = """\
You are a senior regulatory expert specialising in EU banking law (CRR). \
Your task is to generate adversarial evaluation Q&A pairs designed to stress-test \
a RAG system's ability to distinguish between closely related articles.

Requirements:
- Focus on the failure modes described in the batch focus.
- Generate questions that would be EASY to answer incorrectly if the wrong article \
  is retrieved (false friends, similar terminology, adjacent concepts).
- For multi_hop: the answer requires synthesising information from 2+ articles.
- For negative: the question contains a plausible-sounding but wrong premise.
- For false_friend: terminology from one article is easily confused with another.
- For ambiguous: the question wording could plausibly refer to multiple articles.
- For diluted_embedding: rephrase so key regulatory terms are replaced with \
  plain-language synonyms that would score lower in vector similarity.
- reference_answer must be accurate and cite specific paragraph(s).
- expected_articles lists ALL articles needed to answer (can be multiple).

Output ONLY a JSON array — no markdown, no commentary. Each element:
{
  "question": "...",
  "expected_articles": ["92", "93"],
  "reference_answer": "...",
  "category": "capital_ratios",
  "difficulty": "hard",
  "question_type": "multi_hop|negative|false_friend|ambiguous|diluted_embedding",
  "language": "en",
  "notes": "..."
}
"""


def _build_pass2_user_prompt(batch: dict, article_texts: dict[str, dict]) -> str:
    parts = []
    total_chars = 0
    # Distribute char budget evenly across articles in batch
    per_article_limit = MAX_ARTICLE_CHARS // max(len(batch["articles"]), 1)

    for art_num in batch["articles"]:
        entry = article_texts.get(art_num)
        if not entry:
            parts.append(f"--- Article {art_num} ---\n[NOT FOUND IN INDEX]\n")
            continue
        text = entry["text"]
        if len(text) > per_article_limit:
            text = text[:per_article_limit] + "\n[TRUNCATED]"
        total_chars += len(text)
        title = entry.get("title", "")
        parts.append(f"--- Article {art_num}: {title} ---\n{text}\n")

    articles_joined = "\n".join(parts)
    article_nums_str = ", ".join(batch["articles"])

    return f"""\
Batch ID: {batch['id']}
Articles: {article_nums_str}
Focus / failure modes to target: {batch['focus']}

{articles_joined}

Generate 3–5 adversarial Q&A pairs targeting the failure modes described above. \
At least one must be question_type="multi_hop", at least one "false_friend" or "negative". \
Use difficulty="hard" for all.
"""


def run_pass2(
    gpt: GPTClient,
    batches: list[dict],
    all_articles: dict[str, dict],
    existing_cases: list[dict],
    output_path: Path,
    id_counter: list[int],
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Generate adversarial Q&A pairs. Returns number of new cases added."""
    covered_batches: set[str] = {
        case.get("notes", "").split("batch=")[-1].split()[0]
        for case in existing_cases
        if "batch=" in case.get("notes", "")
    }

    new_cases = 0
    total = len(batches)
    for idx, batch in enumerate(batches, 1):
        batch_id = batch["id"]
        logger.info("[Pass 2] %d/%d — Batch '%s' (%s)", idx, total, batch_id, ", ".join(batch["articles"]))

        if batch_id in covered_batches:
            logger.info("  → Already covered, skipping.")
            continue

        if dry_run:
            logger.info("  → [dry-run] Would call GPT for batch '%s'.", batch_id)
            continue

        user_prompt = _build_pass2_user_prompt(batch, all_articles)
        raw = gpt.call_with_json_fix(PASS2_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            logger.warning("  → GPT call failed for batch '%s', skipping.", batch_id)
            continue

        cases = _parse_json_response(raw)
        if not cases:
            logger.warning("  → Could not parse GPT response for batch '%s'.", batch_id)
            if verbose:
                logger.debug("Raw response:\n%s", raw)
            continue

        # Tag each case with the batch ID in notes for resumability
        for case in cases:
            existing_notes = case.get("notes", "")
            batch_tag = f"batch={batch_id}"
            if batch_tag not in existing_notes:
                case["notes"] = f"{existing_notes} {batch_tag}".strip()

        added = _append_cases(cases, output_path, id_counter, existing_cases, batch_id)
        new_cases += added
        logger.info("  → Added %d case(s) (total so far: %d).", added, id_counter[0] - 1)

    return new_cases


# ---------------------------------------------------------------------------
# Shared helpers: deduplication + JSONL I/O
# ---------------------------------------------------------------------------

def _normalise_question(q: str) -> str:
    """Lowercase + collapse whitespace for dedup comparison."""
    return re.sub(r"\s+", " ", q.lower().strip())


def load_existing_cases(output_path: Path) -> list[dict]:
    """Load existing JSONL cases from output_path (returns empty list if missing)."""
    if not output_path.exists():
        return []
    cases = []
    with open(output_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line: %s", line[:80])
    logger.info("Loaded %d existing cases from %s.", len(cases), output_path)
    return cases


def _append_cases(
    new_cases: list[dict],
    output_path: Path,
    id_counter: list[int],
    existing_cases: list[dict],
    source_label: str,
) -> int:
    """Validate, deduplicate, assign IDs, and append to JSONL. Returns count added."""
    existing_questions = {
        _normalise_question(c.get("question", ""))
        for c in existing_cases
    }

    required_fields = {"question", "expected_articles", "reference_answer", "category", "difficulty", "question_type"}
    added = 0

    with open(output_path, "a", encoding="utf-8") as fh:
        for case in new_cases:
            # Basic validation
            missing = required_fields - set(case.keys())
            if missing:
                logger.warning("Skipping case (missing fields %s) from %s.", missing, source_label)
                continue

            if not case.get("question"):
                logger.warning("Skipping case with empty question from %s.", source_label)
                continue

            # Deduplication
            norm_q = _normalise_question(case["question"])
            if norm_q in existing_questions:
                logger.debug("Skipping duplicate question: %s", case["question"][:80])
                continue

            # Ensure expected_articles is a list of strings
            arts = case.get("expected_articles", [])
            if isinstance(arts, str):
                arts = [arts]
            case["expected_articles"] = [str(a) for a in arts]

            # Assign ID
            case_id = f"case_{id_counter[0]:03d}"
            id_counter[0] += 1
            case["id"] = case_id

            # Ensure language field
            case.setdefault("language", "en")

            # Write immediately (crash-safe)
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
            fh.flush()

            existing_cases.append(case)
            existing_questions.add(norm_q)
            added += 1

    return added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a golden evaluation dataset for the CRR RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL),
        help=f"OpenAI model to use (default: {DEFAULT_MODEL}, env: {DEFAULT_MODEL_ENV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pass",
        dest="pass_num",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only Pass 1 (article-anchored) or Pass 2 (adversarial). Default: both.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Limit Pass 1 to first N priority articles (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract articles and log the plan — no GPT calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language filter for Qdrant scroll (default: en).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CRR Golden Dataset Generator")
    logger.info("  model   : %s", args.model)
    logger.info("  output  : %s", args.output)
    logger.info("  pass    : %s", args.pass_num or "1+2")
    logger.info("  dry-run : %s", args.dry_run)
    logger.info("=" * 60)

    # --- Stage 1: Extract articles ---
    all_articles = extract_articles(language=args.language, verbose=args.verbose)
    logger.info("Total articles extracted: %d", len(all_articles))

    # --- Stage 2: Select priority articles ---
    priority_articles = select_priority_articles(all_articles, max_articles=args.max_articles)

    # Dry-run: print plan and exit
    if args.dry_run:
        logger.info("")
        logger.info("=== DRY RUN — PASS 1 PLAN ===")
        for a in priority_articles:
            logger.info(
                "  Article %s (%s) — %d chars — %s",
                a["article"], a["category"], len(a["text"]), a.get("title", ""),
            )
        logger.info("")
        logger.info("=== DRY RUN — PASS 2 PLAN ===")
        for b in ADVERSARIAL_BATCHES:
            missing = [a for a in b["articles"] if a not in all_articles]
            logger.info(
                "  Batch %-20s  articles: %-30s  missing: %s",
                b["id"],
                ", ".join(b["articles"]),
                missing or "none",
            )
        logger.info("")
        logger.info("Dry run complete. No GPT calls made.")
        return

    # --- Load existing cases (resumability) ---
    existing_cases = load_existing_cases(args.output)
    # ID counter starts at max existing ID + 1
    if existing_cases:
        max_id = max(
            int(c["id"].replace("case_", ""))
            for c in existing_cases
            if c.get("id", "").startswith("case_")
        )
        id_counter = [max_id + 1]
    else:
        id_counter = [1]
    logger.info("ID counter starts at: %d", id_counter[0])

    # --- GPT client ---
    gpt = GPTClient(model=args.model)

    # --- Pass 1 ---
    if args.pass_num in (None, 1):
        logger.info("")
        logger.info("--- Starting Pass 1: Article-Anchored Generation ---")
        n1 = run_pass1(
            gpt=gpt,
            priority_articles=priority_articles,
            existing_cases=existing_cases,
            output_path=args.output,
            id_counter=id_counter,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        logger.info("Pass 1 complete. New cases: %d", n1)

    # --- Pass 2 ---
    if args.pass_num in (None, 2):
        logger.info("")
        logger.info("--- Starting Pass 2: Adversarial Generation ---")
        n2 = run_pass2(
            gpt=gpt,
            batches=ADVERSARIAL_BATCHES,
            all_articles=all_articles,
            existing_cases=existing_cases,
            output_path=args.output,
            id_counter=id_counter,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        logger.info("Pass 2 complete. New cases: %d", n2)

    # --- Summary ---
    final_cases = load_existing_cases(args.output)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Generation complete.")
    logger.info("  Total cases in %s: %d", args.output, len(final_cases))

    # Breakdown by category
    cat_counts: dict[str, int] = defaultdict(int)
    qt_counts: dict[str, int] = defaultdict(int)
    diff_counts: dict[str, int] = defaultdict(int)
    for c in final_cases:
        cat_counts[c.get("category", "unknown")] += 1
        qt_counts[c.get("question_type", "unknown")] += 1
        diff_counts[c.get("difficulty", "unknown")] += 1

    logger.info("  By category:      %s", dict(sorted(cat_counts.items())))
    logger.info("  By question_type: %s", dict(sorted(qt_counts.items())))
    logger.info("  By difficulty:    %s", dict(sorted(diff_counts.items())))
    logger.info("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
