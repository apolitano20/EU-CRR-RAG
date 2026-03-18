import type { ArticleResponse, QueryResponse, SourceNode } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ArticleNotFoundError extends Error {
  constructor(public articleId: string) {
    super(`Article ${articleId} not found in the CRR index`);
  }
}

export async function postQuery(
  query: string,
  language?: string
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, preferred_language: language ?? null }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function submitFeedback(payload: {
  query: string;
  answer: string;
  feedback: string;
  sources: SourceNode[];
  viewed_article?: ArticleResponse | null;
}): Promise<{ status: string; filename: string }> {
  const res = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...payload,
      viewed_article: payload.viewed_article ?? null,
    }),
  });
  if (!res.ok) throw new Error(`Feedback submission failed: ${res.status}`);
  return res.json();
}

export async function getArticle(
  articleId: string,
  language?: string
): Promise<ArticleResponse> {
  const params = language ? `?language=${encodeURIComponent(language)}` : "";
  const res = await fetch(`${API_BASE}/api/article/${encodeURIComponent(articleId)}${params}`);
  if (res.status === 404) throw new ArticleNotFoundError(articleId);
  if (!res.ok) throw new Error(`Article fetch failed: ${res.status} ${res.statusText}`);
  return res.json();
}
