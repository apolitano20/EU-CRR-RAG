"use client";

import { useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import ChatPanel from "@/components/chat/ChatPanel";
import DocumentViewer from "@/components/viewer/DocumentViewer";
import type { ArticleResponse } from "@/lib/types";

export default function AppLayout() {
  const [selectedArticle, setSelectedArticle] = useState<ArticleResponse | null>(null);

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <header className="flex-none h-12 bg-[#003399] text-white flex items-center px-5 shadow-sm z-10">
        <span className="text-base font-bold tracking-tight">EU CRR</span>
        <span className="ml-2 text-slate-300 text-sm font-normal">
          Legal Research Assistant
        </span>
        <span className="ml-auto text-xs text-slate-400 font-mono">
          Regulation (EU) No 575/2013
        </span>
      </header>

      <div className="flex-1 min-h-0">
        <PanelGroup direction="horizontal" className="h-full">
          <Panel defaultSize={42} minSize={25}>
            <ChatPanel onArticleSelect={setSelectedArticle} />
          </Panel>

          <PanelResizeHandle className="w-px bg-slate-200 hover:bg-[#003399] hover:w-0.5 transition-all cursor-col-resize" />

          <Panel defaultSize={58} minSize={30}>
            <DocumentViewer
              article={selectedArticle}
              onArticleSelect={setSelectedArticle}
            />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
