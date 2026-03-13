import type { ArticleResponse, QueryResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export async function getArticle(
  articleId: string,
  language?: string
): Promise<ArticleResponse> {
  const params = language ? `?language=${encodeURIComponent(language)}` : "";
  const res = await fetch(`${API_BASE}/api/article/${encodeURIComponent(articleId)}${params}`);
  if (!res.ok)
    throw new Error(`Article fetch failed: ${res.status} ${res.statusText}`);
  return res.json();
}
