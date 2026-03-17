"use client";

import type { ReactNode } from "react";
import {
  ART_RUN_RE,
  EXTERNAL_CONTEXT_RE,
  GRID_ITEM_RE,
  ROMAN_ITEM_RE,
  NUMBERED_PARA_RE,
  INLINE_ITEM_RE,
  splitInlineItems,
  parseTextRuns,
  type ParsedRun,
} from "@/lib/legal-text-parser";

interface ProvisionTextProps {
  text: string;
  onArticleRef: (articleId: string) => void;
}

function renderRuns(runs: ParsedRun[], onArticleRef: (id: string) => void): ReactNode[] {
  return runs.map((run, i) => {
    if (run.kind === "article") {
      return (
        <button
          key={`art-${run.num}-${i}`}
          onClick={() => onArticleRef(run.num)}
          className="text-[#003399] hover:text-[#002277] hover:underline font-medium cursor-pointer"
        >
          {run.num}
        </button>
      );
    }
    return run.value;
  });
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
                    {renderRuns(parseTextRuns(trimmed), onArticleRef)}
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
                    {renderRuns(parseTextRuns(trimmed), onArticleRef)}
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
                    {body ? renderRuns(parseTextRuns(body), onArticleRef) : null}
                  </p>
                );
              }

              return <p key={li}>{renderRuns(parseTextRuns(trimmed), onArticleRef)}</p>;
            })}
          </div>
        );
      })}
    </div>
  );
}
