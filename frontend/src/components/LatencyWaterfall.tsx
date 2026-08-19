import React from 'react';
import { Zap, Clock, AlertTriangle, CheckCircle } from 'lucide-react';

interface LatencyWaterfallProps {
  latencyData?: {
    stages?: Record<string, number>;
    ragLatencyMs?: number;
    totalLatencyMs?: number;
  };
}

export const LatencyWaterfall: React.FC<LatencyWaterfallProps> = ({ latencyData }) => {
  if (!latencyData || !latencyData.stages) {
    return null;
  }

  const stages = latencyData.stages;
  const ragTotal = latencyData.ragLatencyMs || 0;
  const grandTotal = latencyData.totalLatencyMs || 0;
  const TARGET_BUDGET_MS = 200;
  const isWithinBudget = ragTotal <= TARGET_BUDGET_MS;

  const stageKeys = [
    { key: 'stt', label: 'Sarvam Voice STT', color: 'bg-indigo-500' },
    { key: 'inputValidation', label: 'Security & Sanitization', color: 'bg-emerald-500' },
    { key: 'queryAnalysis', label: 'Language & Query Routing', color: 'bg-cyan-500' },
    { key: 'vectorSearch', label: 'Qdrant Vector Retrieval', color: 'bg-blue-500' },
    { key: 'bm25Search', label: 'Indic BM25 Lexical Search', color: 'bg-teal-500' },
    { key: 'fusion', label: 'Reciprocal Rank Fusion (RRF)', color: 'bg-amber-500' },
    { key: 'reranking', label: 'FlashRank Cross-Encoder', color: 'bg-orange-500' },
    { key: 'generation', label: 'Gemini Grounded Gen', color: 'bg-fuchsia-500' },
    { key: 'grounding', label: 'Factuality & Guardrails', color: 'bg-rose-500' },
  ];

  return (
    <div className="w-full glass-panel rounded-2xl p-5 border border-slate-800">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Zap className="w-4 h-4 text-brand-400" />
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
            Latency Breakdown & 200ms Benchmark Target
          </h3>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <span className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-900 border border-slate-700">
            <Clock className="w-3.5 h-3.5 text-brand-400" />
            <span className="text-slate-400">RAG Latency:</span>
            <strong className={`font-mono ${isWithinBudget ? 'text-emerald-400' : 'text-amber-400'}`}>
              {ragTotal} ms
            </strong>
          </span>

          <span className={`flex items-center space-x-1 px-3 py-1 rounded-full border text-xs font-semibold ${
            isWithinBudget
              ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
              : 'bg-amber-950/60 border-amber-500/40 text-amber-300'
          }`}>
            {isWithinBudget ? (
              <>
                <CheckCircle className="w-3.5 h-3.5" />
                <span>&lt;200ms Target Passed</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Exceeds Target</span>
              </>
            )}
          </span>
        </div>
      </div>

      {/* Stacked Visual Bar */}
      <div className="w-full bg-slate-900 rounded-xl h-6 flex overflow-hidden p-0.5 border border-slate-800 mb-4">
        {stageKeys.map(({ key, color }) => {
          const val = stages[key] || 0;
          if (val <= 0) return null;
          const pct = Math.max(2, (val / Math.max(1, grandTotal)) * 100);
          return (
            <div
              key={key}
              style={{ width: `${pct}%` }}
              className={`h-full ${color} transition-all duration-500 relative group cursor-pointer`}
              title={`${key}: ${val}ms`}
            />
          );
        })}
      </div>

      {/* Individual Waterfall Items */}
      <div className="space-y-2 text-xs">
        {stageKeys.map(({ key, label, color }) => {
          const val = stages[key];
          if (val === undefined) return null;
          const barPct = Math.min(100, (val / TARGET_BUDGET_MS) * 100);

          return (
            <div key={key} className="flex items-center justify-between py-1 px-2 rounded-lg hover:bg-slate-900/60 transition-colors">
              <div className="flex items-center space-x-2.5 w-1/3">
                <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
                <span className="text-slate-300 font-medium truncate">{label}</span>
              </div>

              <div className="w-1/2 bg-slate-900 rounded-full h-2 overflow-hidden mx-4">
                <div
                  style={{ width: `${barPct}%` }}
                  className={`h-full ${color} rounded-full transition-all duration-300`}
                />
              </div>

              <div className="w-20 text-right font-mono font-semibold text-slate-200">
                {val} ms
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
