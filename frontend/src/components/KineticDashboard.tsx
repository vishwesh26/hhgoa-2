import React, { useState, useRef } from 'react';
import { VoiceRecorder } from './VoiceRecorder';

interface KineticDashboardProps {
  onQuerySubmit: (query: string) => void;
  onAudioSubmit: (blob: Blob, filename: string) => void;
  loading: boolean;
  result: any;
  error: string | null;
}

export const KineticDashboard: React.FC<KineticDashboardProps> = ({
  onQuerySubmit,
  onAudioSubmit,
  loading,
  result,
  error,
}) => {
  const [inputText, setInputText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputText.trim() && !loading) {
      onQuerySubmit(inputText.trim());
      setInputText('');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onAudioSubmit(file, file.name);
    }
  };

  return (
    <div className="flex-1 flex flex-col p-6 md:p-10 overflow-y-auto max-w-6xl mx-auto w-full gap-8">
      {/* Top Banner Status */}
      <div className="flex justify-between items-center bg-white border-4 border-black p-4 hard-shadow">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 bg-neon-green border-2 border-black rounded-full animate-pulse"></div>
          <span className="font-mono text-sm font-bold tracking-wider">SYSTEM STATUS: OPTIMAL & READY</span>
        </div>
        <div className="hidden sm:flex items-center gap-3 font-mono text-xs text-slate-700">
          <span className="bg-surface-container px-2 py-1 border-2 border-black">STT: Sarvam Saarika 2.5</span>
          <span className="bg-surface-container px-2 py-1 border-2 border-black">LLM: Gemini 2.5 Flash</span>
          <span className="bg-surface-container px-2 py-1 border-2 border-black">RAG: Qdrant + BM25</span>
        </div>
      </div>

      {/* Main Interaction Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left: Interaction Controls & Voice Box */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col items-center gap-6 text-center">
            <h3 className="font-headline font-black text-xl tracking-tight uppercase">Voice Query Capture</h3>
            <p className="font-mono text-xs text-slate-600">
              Speak naturally in English or type your question below.
            </p>

            {/* Voice Recorder Component */}
            <VoiceRecorder onAudioSubmit={onAudioSubmit} disabled={loading} />

            {/* Audio File Upload Alternative */}
            <div className="w-full pt-4 border-t-2 border-black flex flex-col items-center gap-2">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="audio/*"
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                className="w-full py-2 bg-surface border-2 border-black font-mono text-xs font-bold hover:bg-neon-yellow transition-colors hard-shadow-sm flex items-center justify-center gap-2"
              >
                <span className="material-symbols-outlined text-base">upload_file</span>
                Upload Pre-Recorded Audio (.wav / .webm / .mp3)
              </button>
            </div>
          </div>

          {/* Text Input Form */}
          <form onSubmit={handleSubmit} className="bg-white border-4 border-black p-4 hard-shadow flex flex-col gap-3">
            <label className="font-mono text-xs font-bold uppercase tracking-wider flex items-center gap-1">
              <span className="material-symbols-outlined text-sm">keyboard</span> Or Type Your Question:
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask about BJ's food stamps, Corporation, Photosynthesis..."
                className="flex-1 p-3 border-2 border-black font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !inputText.trim()}
                className="px-5 bg-primary text-white font-mono text-xs font-bold border-2 border-black hover:bg-primary-container hard-shadow-sm disabled:opacity-50"
              >
                SEND
              </button>
            </div>

            {/* Quick Sample Questions for Testing */}
            <div className="flex flex-col gap-1.5 mt-2 pt-2 border-t-2 border-slate-200">
              <span className="font-mono text-[11px] font-bold text-slate-700 uppercase flex items-center gap-1">
                <span className="material-symbols-outlined text-xs text-primary">lightbulb</span>
                Sample Evaluation Prompts (Click to test):
              </span>
              <div className="flex flex-wrap gap-1.5">
                {[
                  { text: "Does BJ's accept food stamps?", label: "BJ's Food Stamps" },
                  { text: 'What is the definition of a corporation?', label: 'Corporation Definition' },
                  { text: 'What are foods low in potassium?', label: 'Low Potassium Foods' },
                  { text: 'What does laches mean in legal terms?', label: 'Laches Legal Term' },
                  { text: 'Who won the 2026 World Cup?', label: '🛡 Guardrail Refusal' },
                ].map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    disabled={loading}
                    onClick={() => onQuerySubmit(item.text)}
                    className="px-2.5 py-1 bg-surface-container border border-black font-mono text-[11px] font-bold hover:bg-neon-yellow transition-colors hard-shadow-xs text-left"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          </form>
        </div>

        {/* Right: Live Response & Telemetry Canvas */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          {/* Main Response Box */}
          <div className="bg-white border-4 border-black p-6 hard-shadow min-h-[380px] flex flex-col">
            
            {/* Header with Badges */}
            <div className="flex flex-wrap justify-between items-center border-b-2 border-black pb-4 mb-4 gap-2">
              <span className="font-mono text-xs font-bold tracking-wider text-slate-800 flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm text-primary">psychology</span>
                GROUNDED RESPONSE & TELEMETRY
              </span>
              {result && (
                <div className="flex items-center gap-2">
                  <span className="bg-black text-white font-mono text-xs px-2 py-0.5 border border-black font-bold">
                    ENGLISH (EN)
                  </span>
                  <span className={`font-mono text-xs px-2 py-0.5 border-2 border-black font-bold ${
                    result.confidence > 0.6 ? 'bg-neon-green text-black' : 'bg-neon-yellow text-black'
                  }`}>
                    CONF: {Math.round((result.confidence || 0) * 100)}%
                  </span>
                  <span className="bg-primary text-white font-mono text-xs px-2 py-0.5 border border-black font-bold">
                    {result.latency?.totalLatencyMs ? `${Math.round(result.latency.totalLatencyMs)}ms` : '<200ms'}
                  </span>
                </div>
              )}
            </div>

            {/* Content Area */}
            {loading ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center my-auto">
                <div className="w-12 h-12 border-4 border-black border-t-primary rounded-full animate-spin"></div>
                <div className="font-mono text-sm font-bold tracking-wider animate-pulse">
                  EXECUTING VOICE RAG PIPELINE...
                </div>
                <p className="font-mono text-xs text-slate-500">
                  STT ➔ Query Analyzer ➔ Parallel Hybrid Search ➔ FlashRank ➔ Grounded LLM
                </p>
              </div>
            ) : error ? (
              <div className="p-4 bg-red-50 border-3 border-red-500 text-red-900 font-mono text-sm">
                <div className="font-bold flex items-center gap-1 mb-1">
                  <span className="material-symbols-outlined text-base">error</span> Error Encountered:
                </div>
                {error}
              </div>
            ) : result ? (
              <div className="flex-1 flex flex-col gap-4">
                
                {/* Transcript */}
                {result.transcript && (
                  <div className="bg-surface-container p-3 border-2 border-black">
                    <span className="font-mono text-xs font-bold text-slate-600 block mb-1 uppercase tracking-wider">
                      Transcribed Query:
                    </span>
                    <p className="font-headline font-bold text-base text-black">
                      "{result.transcript}"
                    </p>
                  </div>
                )}

                {/* Grounded Answer or Voice Notice */}
                {result.refused && !result.transcript ? (
                  <div className="bg-amber-50 p-4 border-3 border-amber-500 text-amber-900">
                    <span className="font-mono text-xs font-bold text-amber-700 block mb-1 uppercase tracking-wider flex items-center gap-1">
                      <span className="material-symbols-outlined text-sm">mic_off</span>
                      Speech Recognition Notice:
                    </span>
                    <p className="font-mono text-sm leading-relaxed">
                      {result.answer}
                    </p>
                  </div>
                ) : (
                  <div className="bg-surface-container-low p-4 border-3 border-black">
                    <span className="font-mono text-xs font-bold text-primary block mb-2 uppercase tracking-wider flex items-center gap-1">
                      <span className="material-symbols-outlined text-sm">verified</span>
                      {result.refused ? 'Knowledge Guardrail (Grounded Refusal):' : `Grounded Answer (${result.modelUsed || 'Gemini 2.5 Flash'}):`}
                    </span>
                    <p className="font-body text-base leading-relaxed text-slate-900 whitespace-pre-wrap">
                      {result.answer}
                    </p>
                    {result.refusalReason && (
                      <p className="font-mono text-xs text-slate-500 mt-2 border-t border-slate-300 pt-2">
                        ℹ Guardrail Reason: {result.refusalReason}
                      </p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center text-slate-400 gap-3 my-auto">
                <span className="material-symbols-outlined text-6xl text-slate-300">graphic_eq</span>
                <p className="font-mono text-sm text-slate-500 font-bold">
                  &gt; AWAITING VOICE OR TEXT QUERY INPUT...
                </p>
                <p className="font-mono text-xs text-slate-400 max-w-sm">
                  Click the red mic button to ask your question in English.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
