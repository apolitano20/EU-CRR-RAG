"use client";

import type { SourceNode } from "@/lib/types";

interface SourceChipListProps {
  sources: SourceNode[];
  queryLanguage?: string;
  onSourceClick: (articleId: string, language?: string) => void;
}

export default function SourceChipList({ sources, queryLanguage, onSourceClick }: SourceChipListProps) {
  // Filter by query language when available, then deduplicate by article
  const languageFiltered = queryLanguage
    ? sources.filter((s) => !s.metadata.language || s.metadata.language === queryLanguage)
    : sources;

  const seen = new Set<string>();
  const unique = languageFiltered.filter((s) => {
    if (!s.metadata.article) return false;
    const key = s.metadata.article;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  if (unique.length === 0) return null;

  return (
    <div className="border-t border-slate-100 pt-3">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-2">
        Sources
      </p>
      <div className="flex flex-wrap gap-2">
        {unique.map((source, i) => {
          const artId = source.metadata.article!;
          const lang = source.metadata.language;
          const label = source.metadata.article_title
            ? `Art. ${artId} — ${source.metadata.article_title}`
            : `Article ${artId}`;

          return (
            <button
              key={`${artId}-${lang}-${i}`}
              onClick={() => onSourceClick(artId, lang)}
              title={source.text.slice(0, 200)}
              className={`
                inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
                transition-colors cursor-pointer
                ${
                  source.expanded
                    ? "bg-amber-50 text-amber-700 border border-amber-200 hover:bg-amber-100"
                    : "bg-blue-50 text-[#003399] border border-blue-200 hover:bg-blue-100"
                }
              `}
            >
              {source.expanded && <span className="opacity-70">↗</span>}
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
