import React from 'react';
import { Mic, Globe, Brain, Search, Cpu, Sparkles, ShieldCheck, CheckCircle2, Loader2 } from 'lucide-react';

interface Stage {
  id: string;
  name: string;
  icon: React.ElementType;
}

const PIPELINE_STAGES: Stage[] = [
  { id: 'stt', name: 'Sarvam Voice STT', icon: Mic },
  { id: 'queryAnalysis', name: 'Lang & Code-Mix Analysis', icon: Globe },
  { id: 'routing', name: 'Adaptive Chunk Routing', icon: Brain },
  { id: 'search', name: 'Parallel Qdrant + BM25', icon: Search },
  { id: 'reranking', name: 'FlashRank Cross-Encoder', icon: Cpu },
  { id: 'generation', name: 'Gemini Grounded Gen', icon: Sparkles },
  { id: 'grounding', name: 'Factuality Guardrail', icon: ShieldCheck },
];

interface LivePipelineProps {
  isLoading: boolean;
  latencies?: Record<string, number>;
}

export const LivePipeline: React.FC<LivePipelineProps> = ({
  isLoading,
  latencies = {}
}) => {
  return (
    <div className="w-full glass-panel rounded-2xl p-5 mb-6 border border-slate-800">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold tracking-wider uppercase text-slate-400 flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
          <span>Real-time Pipeline Progression</span>
        </h3>
        {isLoading && (
          <span className="text-xs font-semibold text-brand-400 flex items-center space-x-1.5 animate-pulse">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            <span>Processing Pipeline...</span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5">
        {PIPELINE_STAGES.map((stage) => {
          const Icon = stage.icon;
          const isDone = !isLoading && Object.keys(latencies).length > 0;
          const stageLatency = latencies[stage.id];

          return (
            <div
              key={stage.id}
              className={`p-3 rounded-xl border flex flex-col items-center text-center transition-all duration-300 ${
                isDone
                  ? 'bg-slate-900/90 border-brand-500/40 text-brand-300'
                  : isLoading
                  ? 'bg-slate-900/40 border-slate-800 text-slate-400'
                  : 'bg-slate-950/40 border-slate-800/60 text-slate-400'
              }`}
            >
              <div className="mb-2 p-2 rounded-lg bg-slate-800/80 text-brand-400">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-brand-400" />
                ) : isLoading ? (
                  <Icon className="w-4 h-4 animate-pulse text-brand-400" />
                ) : (
                  <Icon className="w-4 h-4" />
                )}
              </div>
              <span className="text-[11px] font-semibold leading-tight line-clamp-1 mb-1">{stage.name}</span>
              <span className="text-[10px] font-mono text-slate-400">
                {stageLatency !== undefined ? `${stageLatency}ms` : 'Ready'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
