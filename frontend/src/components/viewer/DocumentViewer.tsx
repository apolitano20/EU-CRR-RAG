"use client";

import { useState } from "react";
import { getArticle, ArticleNotFoundError } from "@/lib/api";
import type { ArticleResponse } from "@/lib/types";
import DocumentBreadcrumb from "./DocumentBreadcrumb";
import ProvisionHeader from "./ProvisionHeader";
import ProvisionText from "./ProvisionText";

interface DocumentViewerProps {
  article: ArticleResponse | null;
  viewerError: string | null;
  onArticleSelect: (article: ArticleResponse) => void;
  onArticleNotFound: (articleId: string) => void;
}

export default function DocumentViewer({ article, viewerError, onArticleSelect, onArticleNotFound }: DocumentViewerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!article) return;
    const breadcrumb = [article.part, article.title, article.chapter, article.section]
      .filter(Boolean)
      .join(" > ");
    const md = `## ${article.article} – ${article.article_title}\n\n${breadcrumb ? `_${breadcrumb}_\n\n` : ""}${article.text}`;
    await navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleArticleRef = async (articleId: string) => {
    try {
      const data = await getArticle(articleId, article?.language);
      onArticleSelect(data);
    } catch (err) {
      if (err instanceof ArticleNotFoundError) {
        onArticleNotFound(err.articleId);
      } else {
        console.error("Failed to navigate to article:", err);
      }
    }
  };

  if (!article) {
    if (viewerError) {
      return (
        <div className="h-full flex flex-col items-center justify-center bg-slate-50 text-center select-none px-8">
          <div className="text-4xl mb-3">🔍</div>
          <p className="text-sm font-medium text-slate-600">Article not found</p>
          <p className="text-xs mt-2 text-slate-400 max-w-xs leading-relaxed">{viewerError}</p>
        </div>
      );
    }
    return (
      <div className="h-full flex flex-col items-center justify-center bg-slate-50 text-center select-none">
        <div className="text-4xl mb-3">📄</div>
        <p className="text-sm font-medium text-slate-500">No article selected</p>
        <p className="text-xs mt-2 text-slate-400 max-w-xs leading-relaxed">
          Click a source chip in the Q&amp;A panel to open an article here
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="flex-none px-6 py-3 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
        <DocumentBreadcrumb article={article} />
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 text-xs text-slate-500 hover:text-slate-700 hover:bg-slate-200/50 rounded transition-colors flex-none"
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
      <div className="flex-1 overflow-y-auto px-6 py-5">
        <ProvisionHeader article={article} />
        <ProvisionText text={article.text} onArticleRef={handleArticleRef} />
        {article.referenced_external && article.referenced_external.length > 0 && (
          <div className="mt-6 pt-4 border-t border-slate-100">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">
              Referenced Regulations &amp; Directives
            </p>
            <div className="flex flex-wrap gap-1.5">
              {article.referenced_external.map((ref) => (
                <span key={ref}
                  className="inline-block px-2 py-0.5 text-xs rounded bg-amber-50 text-amber-700 border border-amber-200"
                >{ref}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
