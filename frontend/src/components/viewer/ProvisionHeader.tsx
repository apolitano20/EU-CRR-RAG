import type { ArticleResponse } from "@/lib/types";

interface ProvisionHeaderProps {
  article: ArticleResponse;
}

export default function ProvisionHeader({ article }: ProvisionHeaderProps) {
  return (
    <div className="mb-5 pb-4 border-b border-slate-100">
      <h1 className="text-xl font-bold text-slate-800">Article {article.article}</h1>
      {article.article_title && (
        <h2 className="text-base font-medium text-slate-600 mt-1">
          {article.article_title}
        </h2>
      )}
      {article.language && article.language !== "en" && (
        <span className="inline-block mt-2 px-2 py-0.5 text-xs bg-slate-100 text-slate-500 rounded-full uppercase">
          {article.language}
        </span>
      )}
    </div>
  );
}
