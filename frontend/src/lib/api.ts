import type { ArticleResponse, HistoryTurn, QueryResponse, SourceNode } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ArticleNotFoundError extends Error {
  constructor(public articleId: string) {
    super(`Article ${articleId} not found in the CRR index`);
  }
}

export async function postQueryStream(
  query: string,
  language: string | undefined,
  history: HistoryTurn[],
  onToken: (token: string) => void,
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, preferred_language: language ?? null, history }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status} ${res.statusText}`);

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: QueryResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.type === "token") {
        onToken(payload.content);
      } else if (payload.type === "sources") {
        finalResult = {
          answer: "", // filled in by the caller from accumulated tokens
          sources: payload.sources,
          trace_id: payload.trace_id,
          language: payload.language ?? null,
        };
      }
    }
  }

  if (!finalResult) throw new Error("Stream ended without sources event");
  return finalResult;
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
