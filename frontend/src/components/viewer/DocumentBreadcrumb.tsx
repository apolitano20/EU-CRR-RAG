import type { ArticleResponse } from "@/lib/types";

interface DocumentBreadcrumbProps {
  article: ArticleResponse;
}

export default function DocumentBreadcrumb({ article }: DocumentBreadcrumbProps) {
  const crumbs = [
    article.part && `Part ${article.part}`,
    article.title && `Title ${article.title}`,
    article.chapter && `Chapter ${article.chapter}`,
    article.section && `Section ${article.section}`,
    `Article ${article.article}`,
  ].filter(Boolean) as string[];

  return (
    <nav className="flex items-center flex-wrap gap-0 text-xs text-slate-500">
      <span className="font-semibold text-[#003399]">CRR</span>
      {crumbs.map((crumb, i) => (
        <span key={i} className="flex items-center">
          <span className="mx-1.5 text-slate-300">›</span>
          <span className={i === crumbs.length - 1 ? "font-medium text-slate-700" : ""}>
            {crumb}
          </span>
        </span>
      ))}
    </nav>
  );
}
