"use client";

import { useState } from "react";
import { postQuery } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";

export function useQuery() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const submitQuery = async (query: string): Promise<QueryResponse | null> => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await postQuery(query);
      setResult(data);
      return data;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { isLoading, error, result, submitQuery };
}
