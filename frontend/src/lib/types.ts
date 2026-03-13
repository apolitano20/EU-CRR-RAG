export interface SourceNode {
  text: string;
  score: number;
  metadata: {
    article?: string;
    article_title?: string;
    part?: string;
    title?: string;
    chapter?: string;
    section?: string;
    language?: string;
    referenced_articles?: string;
    [key: string]: unknown;
  };
  expanded: boolean;
}

export interface QueryResponse {
  answer: string;
  sources: SourceNode[];
  trace_id: string;
}

export interface ArticleResponse {
  article: string;
  article_title: string;
  text: string;
  part?: string;
  title?: string;
  chapter?: string;
  section?: string;
  referenced_articles: string[];
  language: string;
}
