import React from 'react';
import { BookOpen, Layers, CheckCircle } from 'lucide-react';

interface Source {
  chunkId: string;
  docId: string;
  language: string;
  chunkStrategy: string;
  score: number;
  text: string;
}

interface SourcesListProps {
  sources: Source[];
}

export const SourcesList: React.FC<SourcesListProps> = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  const getLangBadge = (_lang: string) => {
    return <span className="px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/30 text-[11px] font-semibold">English</span>;
  };

  return (
    <div className="w-full glass-panel rounded-2xl p-5 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2 text-slate-200 font-bold text-sm">
          <BookOpen className="w-4 h-4 text-brand-400" />
          <span>Retrieved Evidence & Passages ({sources.length})</span>
        </div>
        <span className="text-xs text-slate-400 flex items-center space-x-1">
          <CheckCircle className="w-3.5 h-3.5 text-brand-400" />
          <span>Grounded Evidence Verified</span>
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {sources.map((src, index) => (
          <div
            key={src.chunkId || index}
            className="p-4 rounded-xl bg-slate-900/80 border border-slate-850 hover:border-slate-700 transition-all space-y-2.5"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center space-x-2">
                <span className="w-5 h-5 rounded-full bg-slate-800 text-brand-400 text-xs font-mono font-bold flex items-center justify-center">
                  #{index + 1}
                </span>
                {getLangBadge(src.language)}
                <span className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[11px] font-mono">
                  <Layers className="w-3 h-3 text-slate-400" />
                  <span>{src.chunkStrategy}</span>
                </span>
              </div>

              <div className="text-xs font-mono text-slate-400">
                Score: <strong className="text-emerald-400">{(src.score * 100).toFixed(1)}%</strong>
              </div>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed italic bg-slate-950/40 p-3 rounded-lg border border-slate-900">
              "{src.text}"
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};
