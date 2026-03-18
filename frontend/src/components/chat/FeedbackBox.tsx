"use client";

import { useState } from "react";
import { submitFeedback } from "@/lib/api";
import type { ArticleResponse, SourceNode } from "@/lib/types";

interface FeedbackBoxProps {
  query: string;
  answer: string;
  sources: SourceNode[];
  viewedArticle: ArticleResponse | null;
}

type SubmitState = "idle" | "loading" | "done" | "error";

export default function FeedbackBox({
  query,
  answer,
  sources,
  viewedArticle,
}: FeedbackBoxProps) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [state, setState] = useState<SubmitState>("idle");
  const [savedFile, setSavedFile] = useState("");

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setState("loading");
    try {
      const res = await submitFeedback({
        query,
        answer,
        feedback: text.trim(),
        sources,
        viewed_article: viewedArticle,
      });
      setSavedFile(res.filename);
      setState("done");
      setText("");
    } catch {
      setState("error");
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mt-1 flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 transition-colors"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
        Add eval feedback
      </button>
    );
  }

  return (
    <div className="mt-3 border border-amber-200 rounded-lg bg-amber-50 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-amber-700 uppercase tracking-wide">
          Eval Feedback
        </span>
        <button
          onClick={() => { setOpen(false); setState("idle"); setText(""); }}
          className="text-xs text-slate-400 hover:text-slate-600"
        >
          ✕
        </button>
      </div>

      {viewedArticle && (
        <p className="text-xs text-amber-600">
          Article in viewer:{" "}
          <span className="font-medium">
            Art. {viewedArticle.article} — {viewedArticle.article_title}
          </span>
        </p>
      )}

      <textarea
        value={text}
        onChange={(e) => { setText(e.target.value); setState("idle"); }}
        placeholder="What did you observe? Any retrieval issues, hallucinations, missing citations…"
        rows={4}
        className="w-full text-xs rounded border border-amber-300 bg-white px-2.5 py-2 text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-amber-400 resize-none"
      />

      <div className="flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={state === "loading" || !text.trim()}
          className="px-3 py-1.5 text-xs font-medium rounded bg-amber-600 text-white hover:bg-amber-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {state === "loading" ? "Saving…" : "Submit feedback"}
        </button>

        {state === "done" && (
          <span className="text-xs text-green-600 font-medium">
            ✓ Saved as <code className="font-mono">{savedFile}</code>
          </span>
        )}
        {state === "error" && (
          <span className="text-xs text-red-500">Failed to save — is the API running?</span>
        )}
      </div>
    </div>
  );
}
