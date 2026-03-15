"use client";

import type { ReactNode } from "react";

// Matches "Article 92" or "Articles 92" — captures the word and number separately
const ART_REF_RE = /\b(Articles?)\s+(\d[\w]*)/g;

// Text following an Article ref that signals an external regulation (not CRR)
const EXTERNAL_CONTEXT_RE = /^\s*(?:to\s+\d[\w]*\s+)?of\s+(?:Regulation|Directive|Decision|Delegated|Implementing)/i;

// Lettered/roman list items: "(a) text", "(i) text", optionally with leading whitespace
const GRID_ITEM_RE = /^\s*\([a-z]+\)\s/i;

// Roman numeral sub-items: "(i) text", "(ii) text" — deeper nesting
const ROMAN_ITEM_RE = /^\s*\((?:i{1,3}|iv|vi{0,3})\)\s/i;

// Numbered paragraphs: "1. text", "2. text", or bare label "7." on its own line
const NUMBERED_PARA_RE = /^\d+\.(\s|$)/;

// Split inline list items like "(a) ...; (b) ...; (i) ..." onto separate lines.
// Only matches single letters (a)-(z) or short roman numerals (i)-(viii) preceded
// by a semicolon/colon or sentence-ending punctuation to avoid false splits.
const INLINE_ITEM_RE = /(?:;\s*|:\s+)(\([a-z]{1,4}\)\s)/gi;

function splitInlineItems(text: string): string {
  return text.replace(INLINE_ITEM_RE, ";\n$1");
}

interface ProvisionTextProps {
  text: string;
  onArticleRef: (articleId: string) => void;
}

function parseRefs(text: string, onArticleRef: (id: string) => void): ReactNode[] {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  ART_REF_RE.lastIndex = 0;
  while ((match = ART_REF_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const [fullMatch, word, num] = match;
    const textAfter = text.slice(match.index + fullMatch.length);

    // If followed by "of Regulation/Directive/...", it's an external ref — render as plain text
    if (EXTERNAL_CONTEXT_RE.test(textAfter)) {
      parts.push(`${word} ${num}`);
    } else {
      parts.push(
        <button
          key={`art-${num}-${match.index}`}
          onClick={() => onArticleRef(num)}
          className="text-[#003399] hover:text-[#002277] hover:underline font-medium cursor-pointer"
        >
          {word} {num}
        </button>
      );
    }
    lastIndex = match.index + fullMatch.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

export default function ProvisionText({ text, onArticleRef }: ProvisionTextProps) {
  // Split on double newlines first (node chunk boundaries from get_article join),
  // then split inline list items onto their own lines.
  const chunks = text.split(/\n\n+/);

  return (
    <div className="space-y-4 text-sm leading-relaxed text-slate-700">
      {chunks.map((chunk, ci) => {
        // Split inline (a)/(b)/(i) items onto separate lines
        const expanded = splitInlineItems(chunk);
        const lines = expanded.split("\n").filter((l) => l.trim());
        if (lines.length === 0) return null;

        return (
          <div key={ci} className="space-y-1.5">
            {lines.map((line, li) => {
              const trimmed = line.trim();

              // Roman numeral sub-item — deeper indent: (i), (ii), (iii)
              if (ROMAN_ITEM_RE.test(trimmed)) {
                return (
                  <div
                    key={li}
                    className="ml-10 pl-4 border-l-2 border-slate-200 py-0.5 text-slate-600"
                  >
                    {parseRefs(trimmed, onArticleRef)}
                  </div>
                );
              }

              // Lettered list item: (a), (b), (c), (d), (e)
              if (GRID_ITEM_RE.test(trimmed)) {
                return (
                  <div
                    key={li}
                    className="ml-5 pl-4 border-l-2 border-slate-200 py-0.5 text-slate-600"
                  >
                    {parseRefs(trimmed, onArticleRef)}
                  </div>
                );
              }

              // Numbered paragraph — "1. text", "2. text", or bare "7." label
              if (NUMBERED_PARA_RE.test(trimmed)) {
                const spaceIdx = trimmed.indexOf(" ");
                const numLabel = spaceIdx === -1 ? trimmed : trimmed.slice(0, spaceIdx);
                const body = spaceIdx === -1 ? "" : trimmed.slice(spaceIdx + 1);
                return (
                  <p key={li}>
                    <span className="font-semibold text-slate-800 mr-1.5">{numLabel}</span>
                    {body ? parseRefs(body, onArticleRef) : null}
                  </p>
                );
              }

              return <p key={li}>{parseRefs(trimmed, onArticleRef)}</p>;
            })}
          </div>
        );
      })}
    </div>
  );
}
