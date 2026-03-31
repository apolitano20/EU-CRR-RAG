"use client";

import { useState, useRef, useEffect } from "react";
import { useQuery } from "@/hooks/useQuery";
import QuestionInput from "./QuestionInput";
import AnswerCard from "./AnswerCard";
import FeedbackBox from "./FeedbackBox";
import { getArticle, ArticleNotFoundError } from "@/lib/api";
import type { ArticleResponse, HistoryTurn, QueryResponse } from "@/lib/types";

interface Message {
  question: string;
  result: QueryResponse;
}

interface ChatPanelProps {
  onArticleSelect: (article: ArticleResponse) => void;
  onArticleNotFound: (articleId: string) => void;
  selectedArticle: ArticleResponse | null;
}

export default function ChatPanel({ onArticleSelect, onArticleNotFound, selectedArticle }: ChatPanelProps) {
  const { isLoading, error, streamingAnswer, submitQuery } = useQuery();
  const [messages, setMessages] = useState<Message[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const MAX_HISTORY_TURNS = 5;

  const handleSubmit = async (query: string) => {
    setPendingQuestion(query);
    const history: HistoryTurn[] = messages
      .slice(-MAX_HISTORY_TURNS)
      .map((m) => ({ question: m.question, answer: m.result.answer }));
    const result = await submitQuery(query, undefined, history);
    if (result) {
      setMessages((prev) => [...prev, { question: query, result }]);
    }
    setPendingQuestion("");
  };

  const handleSourceClick = async (articleId: string, language?: string) => {
    try {
      const article = await getArticle(articleId, language);
      onArticleSelect(article);
    } catch (err) {
      if (err instanceof ArticleNotFoundError) {
        onArticleNotFound(err.articleId);
      } else {
        console.error("Failed to load article:", err);
      }
    }
  };

  return (
    <div className="h-full flex flex-col bg-white border-r border-slate-200">
      <div className="flex-none px-4 py-3 border-b border-slate-100">
        <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
          Compliance Q&amp;A
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !pendingQuestion && !isLoading && !error && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16 select-none">
            <div className="text-4xl mb-3">⚖️</div>
            <p className="text-sm font-medium text-slate-500">
              Ask a compliance question
            </p>
            <p className="text-xs mt-2 text-slate-400 max-w-xs leading-relaxed">
              e.g. &ldquo;What are the own funds requirements?&rdquo; or
              &ldquo;Explain Article 92&rdquo;
            </p>
          </div>
        )}

        <div className="space-y-8">
          {messages.map((msg, i) => (
            <div key={i} className="space-y-3">
              <div className="px-3 py-2.5 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-sm font-medium text-slate-700">{msg.question}</p>
              </div>
              <AnswerCard
                answer={msg.result.answer}
                sources={msg.result.sources}
                queryLanguage={msg.result.language ?? undefined}
                onSourceClick={handleSourceClick}
              />
              <FeedbackBox
                query={msg.question}
                answer={msg.result.answer}
                sources={msg.result.sources}
                viewedArticle={selectedArticle}
              />
            </div>
          ))}

          {pendingQuestion && (
            <div className="px-3 py-2.5 bg-slate-50 rounded-lg border border-slate-200">
              <p className="text-sm font-medium text-slate-700">{pendingQuestion}</p>
            </div>
          )}

          {isLoading && streamingAnswer && (
            <div className="prose prose-sm max-w-none text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {streamingAnswer}
              <span className="inline-block w-0.5 h-4 bg-slate-400 animate-pulse align-text-bottom ml-0.5" />
            </div>
          )}

          {isLoading && !streamingAnswer && (
            <div className="mt-1 space-y-3">
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-slate-300 border-t-slate-500 animate-spin" />
                Retrieving CRR content…
              </div>
              <div className="space-y-2.5 animate-pulse">
                {[72, 55, 83, 40, 68].map((w, i) => (
                  <div
                    key={i}
                    className="h-3 bg-slate-100 rounded"
                    style={{ width: `${w}%` }}
                  />
                ))}
              </div>
            </div>
          )}

          {error && !isLoading && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
              {error}
            </div>
          )}
        </div>

        <div ref={bottomRef} />
      </div>

      <div className="flex-none p-4 border-t border-slate-100 bg-white">
        <QuestionInput onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
}
