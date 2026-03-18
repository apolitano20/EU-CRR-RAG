"use client";

import { useState } from "react";
import { postQueryStream } from "@/lib/api";
import type { HistoryTurn, QueryResponse } from "@/lib/types";

export function useQuery() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState("");

  const submitQuery = async (
    query: string,
    language?: string,
    history?: HistoryTurn[],
  ): Promise<QueryResponse | null> => {
    setIsLoading(true);
    setError(null);
    setStreamingAnswer("");

    let accumulated = "";

    try {
      const result = await postQueryStream(query, language, history ?? [], (token) => {
        accumulated += token;
        setStreamingAnswer(accumulated);
      });
      return { ...result, answer: accumulated };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      return null;
    } finally {
      setIsLoading(false);
      setStreamingAnswer("");
    }
  };

  return { isLoading, error, streamingAnswer, submitQuery };
}
