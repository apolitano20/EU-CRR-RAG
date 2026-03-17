import type { ArticleResponse } from "@/lib/types";

const LEVEL_LABELS: Record<string, Record<string, string>> = {
  it: { part: "Parte", title: "Titolo", chapter: "Capo", section: "Sezione", article: "Articolo" },
};
const DEFAULT_LABELS = { part: "Part", title: "Title", chapter: "Chapter", section: "Section", article: "Article" };

interface DocumentBreadcrumbProps {
  article: ArticleResponse;
}

export default function DocumentBreadcrumb({ article }: DocumentBreadcrumbProps) {
  const labels = LEVEL_LABELS[article.language ?? "en"] ?? DEFAULT_LABELS;
  const crumbs = [
    article.part    && `${labels.part} ${article.part}`,
    article.title   && `${labels.title} ${article.title}`,
    article.chapter && `${labels.chapter} ${article.chapter}`,
    article.section && `${labels.section} ${article.section}`,
    `${labels.article} ${article.article}`,
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
