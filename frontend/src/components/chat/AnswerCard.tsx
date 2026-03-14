"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { SourceNode } from "@/lib/types";
import SourceChipList from "./SourceChipList";

const markdownComponents: Components = {
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-slate-800 mt-4 mb-2 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-bold text-slate-700 mt-3 mb-1.5">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-700 mt-2 mb-1">{children}</h3>
  ),
  p: ({ children }) => (
    <p className="text-sm text-slate-700 leading-relaxed mb-2 last:mb-0">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>
  ),
  li: ({ children }) => <li className="text-sm text-slate-700">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-800">{children}</strong>
  ),
  code: ({ children }) => (
    <code className="px-1 py-0.5 bg-slate-100 text-slate-700 rounded text-xs font-mono">
      {children}
    </code>
  ),
};

interface AnswerCardProps {
  answer: string;
  sources: SourceNode[];
  queryLanguage?: string;
  onSourceClick: (articleId: string, language?: string) => void;
}

export default function AnswerCard({ answer, sources, queryLanguage, onSourceClick }: AnswerCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const sourcesMarkdown = sources
      .map((s) => `- **${s.metadata.article || "unknown"}** (score: ${s.score.toFixed(3)})`)
      .join("\n");
    const md = `${answer}\n\n### Cross-References\n${sourcesMarkdown}`;
    await navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
          title="Copy as Markdown"
        >
          {copied ? (
            <>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              Copied
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              Copy .md
            </>
          )}
        </button>
      </div>
      <div>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {answer}
        </ReactMarkdown>
      </div>
      <SourceChipList sources={sources} queryLanguage={queryLanguage} onSourceClick={onSourceClick} />
    </div>
  );
}
