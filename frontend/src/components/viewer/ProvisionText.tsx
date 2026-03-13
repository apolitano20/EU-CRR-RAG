"use client";

import type { ReactNode } from "react";

// Matches "Article 92" or "Articles 92" — captures the word and number separately
const ART_REF_RE = /\b(Articles?)\s+(\d[\w]*)/g;

// Grid-list items stored by ingester with leading whitespace: "  (a) text", "  (i) text"
const GRID_ITEM_RE = /^\s+\(/;

// Numbered paragraphs stored by ingester as "1. text", "2. text"
const NUMBERED_PARA_RE = /^\d+\.\s+/;

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
    const [, word, num] = match;
    parts.push(
      <button
        key={`art-${num}-${match.index}`}
        onClick={() => onArticleRef(num)}
        className="text-[#003399] hover:text-[#002277] hover:underline font-medium cursor-pointer"
      >
        {word} {num}
      </button>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

export default function ProvisionText({ text, onArticleRef }: ProvisionTextProps) {
  // Split on double newlines first (node chunk boundaries from get_article join),
  // then on single newlines within each chunk (ingest-level line structure).
  const chunks = text.split(/\n\n+/);

  return (
    <div className="space-y-4 text-sm leading-relaxed text-slate-700">
      {chunks.map((chunk, ci) => {
        const lines = chunk.split("\n").filter((l) => l.trim());
        if (lines.length === 0) return null;

        return (
          <div key={ci} className="space-y-1.5">
            {lines.map((line, li) => {
              const trimmed = line.trim();

              // Lettered/numbered list item — stored with leading spaces by ingester
              if (GRID_ITEM_RE.test(line)) {
                return (
                  <div
                    key={li}
                    className="pl-5 border-l-2 border-slate-200 py-0.5 text-slate-600"
                  >
                    {parseRefs(trimmed, onArticleRef)}
                  </div>
                );
              }

              // Numbered paragraph — "1. text", "2. text"
              if (NUMBERED_PARA_RE.test(trimmed)) {
                const numEnd = trimmed.indexOf(" ");
                const numLabel = trimmed.slice(0, numEnd);
                const body = trimmed.slice(numEnd + 1);
                return (
                  <p key={li}>
                    <span className="font-semibold text-slate-800 mr-1.5">{numLabel}</span>
                    {parseRefs(body, onArticleRef)}
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
