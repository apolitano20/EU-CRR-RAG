// Matches "Article 92" or "Articles 92 and 93" — captures the keyword and the full number run
export const ART_RUN_RE = /\b(Articles?)\s+(\d[\w]*(?:\s*(?:,|and|or)\s+\d[\w]*)*)/gi;

// Text following an Article ref that signals an external regulation (not CRR)
export const EXTERNAL_CONTEXT_RE = /^\s*(?:to\s+\d[\w]*\s+)?of\s+(?:Regulation|Directive|Decision|Delegated|Implementing)/i;

// Lettered/roman list items: "(a) text", "(i) text", optionally with leading whitespace
export const GRID_ITEM_RE = /^\s*\([a-z]+\)\s/i;

// Roman numeral sub-items: "(i) text", "(ii) text" — deeper nesting
export const ROMAN_ITEM_RE = /^\s*\((?:i{1,3}|iv|vi{0,3})\)\s/i;

// Numbered paragraphs: "1. text", "2. text", or bare label "7." on its own line
export const NUMBERED_PARA_RE = /^\d+\.(\s|$)/;

// Split inline list items like "(a) ...; (b) ...; (i) ..." onto separate lines.
// Only matches single letters (a)-(z) or short roman numerals (i)-(viii) preceded
// by a semicolon/colon or sentence-ending punctuation to avoid false splits.
export const INLINE_ITEM_RE = /(?:;\s*|:\s+)(\([a-z]{1,4}\)\s)/gi;

export function splitInlineItems(text: string): string {
  return text.replace(INLINE_ITEM_RE, ";\n$1");
}

export type TextRun = { kind: "text"; value: string };
export type ArticleRun = { kind: "article"; num: string };
export type ParsedRun = TextRun | ArticleRun;

export function parseTextRuns(text: string): ParsedRun[] {
  const runs: ParsedRun[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  ART_RUN_RE.lastIndex = 0;
  while ((match = ART_RUN_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      runs.push({ kind: "text", value: text.slice(lastIndex, match.index) });
    }
    const [fullMatch, word, body] = match;
    const textAfter = text.slice(match.index + fullMatch.length);

    if (EXTERNAL_CONTEXT_RE.test(textAfter)) {
      runs.push({ kind: "text", value: fullMatch });
    } else {
      runs.push({ kind: "text", value: `${word} ` });
      const numRe = /\d[\w]*/g;
      let numMatch: RegExpExecArray | null;
      let bodyLastIndex = 0;
      while ((numMatch = numRe.exec(body)) !== null) {
        if (numMatch.index > bodyLastIndex) {
          runs.push({ kind: "text", value: body.slice(bodyLastIndex, numMatch.index) });
        }
        runs.push({ kind: "article", num: numMatch[0] });
        bodyLastIndex = numMatch.index + numMatch[0].length;
      }
      if (bodyLastIndex < body.length) {
        runs.push({ kind: "text", value: body.slice(bodyLastIndex) });
      }
    }
    lastIndex = match.index + fullMatch.length;
  }

  if (lastIndex < text.length) {
    runs.push({ kind: "text", value: text.slice(lastIndex) });
  }

  return runs;
}
