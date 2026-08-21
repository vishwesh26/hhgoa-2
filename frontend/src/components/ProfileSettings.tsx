import React, { useState } from 'react';

export const ProfileSettings: React.FC = () => {
  const [sarvamModel, setSarvamModel] = useState('saarika:v2.5');
  const [geminiModel, setGeminiModel] = useState('gemini-2.5-flash');
  const [generationMode, setGenerationMode] = useState<'generative' | 'extractive_first'>('generative');
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="flex-1 flex flex-col p-6 md:p-10 overflow-y-auto max-w-6xl mx-auto w-full gap-8">
      
      {/* Header */}
      <div className="bg-white border-4 border-black p-6 hard-shadow flex justify-between items-center flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="material-symbols-outlined text-primary text-3xl">settings</span>
            <h2 className="font-headline font-black text-2xl tracking-tight uppercase">
              Engine Configuration & Profile
            </h2>
          </div>
          <p className="font-mono text-xs text-slate-600">
            Configure STT models, LLM parameters, generation modes, and vector storage.
          </p>
        </div>

        {saved && (
          <div className="bg-neon-green text-black font-mono text-xs font-bold px-3 py-1.5 border-2 border-black animate-bounce">
            ✓ CONFIGURATION SAVED
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: API & Model Settings */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col gap-5">
            <h3 className="font-headline font-black text-lg uppercase tracking-tight flex items-center gap-2 border-b-2 border-black pb-2">
              <span className="material-symbols-outlined text-primary">key</span>
              1. API Keys & Models
            </h3>

            {/* Sarvam STT */}
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700">
                Sarvam AI STT Model:
              </label>
              <select
                value={sarvamModel}
                onChange={(e) => setSarvamModel(e.target.value)}
                className="p-3 bg-surface border-2 border-black font-mono text-xs focus:outline-none focus:ring-2 focus:ring-primary font-bold"
              >
                <option value="saarika:v2.5">saarika:v2.5 (High-Accuracy English STT - Recommended)</option>
                <option value="saaras:v3">saaras:v3 (Streaming Low-Latency)</option>
                <option value="saarika:flash">saarika:flash (Ultra-Fast Batch)</option>
              </select>
              <span className="font-mono text-[11px] text-slate-500">
                Active Key: <code className="bg-slate-100 px-1 border border-black">sk_42oacb1s_...</code> (Configured in .env)
              </span>
            </div>

            {/* Google Gemini */}
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700">
                Google Gemini LLM Model:
              </label>
              <select
                value={geminiModel}
                onChange={(e) => setGeminiModel(e.target.value)}
                className="p-3 bg-surface border-2 border-black font-mono text-xs focus:outline-none focus:ring-2 focus:ring-primary font-bold"
              >
                <option value="gemini-2.5-flash">gemini-2.5-flash (High RPM, Fast Grounded Answering)</option>
                <option value="gemini-3.1-pro-preview">gemini-3.1-pro-preview (Deep Reasoning)</option>
                <option value="gemini-2.5-pro">gemini-2.5-pro (High Capacity)</option>
              </select>
              <span className="font-mono text-[11px] text-slate-500">
                Active Key: <code className="bg-slate-100 px-1 border border-black">AQ.Ab8RN6J...</code> (Configured in .env)
              </span>
            </div>

            {/* Answer Generation Mode */}
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs font-bold uppercase tracking-wider text-slate-700">
                Answer Generation Strategy:
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setGenerationMode('generative')}
                  className={`p-3 border-2 border-black font-mono text-xs font-bold text-left transition-colors ${
                    generationMode === 'generative' ? 'bg-black text-white' : 'bg-surface-container hover:bg-white text-black'
                  }`}
                >
                  <div className="font-black mb-0.5">GENERATIVE</div>
                  <div className="text-[10px] opacity-80">Gemini LLM conversational grounding</div>
                </button>

                <button
                  type="button"
                  onClick={() => setGenerationMode('extractive_first')}
                  className={`p-3 border-2 border-black font-mono text-xs font-bold text-left transition-colors ${
                    generationMode === 'extractive_first' ? 'bg-black text-white' : 'bg-surface-container hover:bg-white text-black'
                  }`}
                >
                  <div className="font-black mb-0.5">EXTRACTIVE-FIRST</div>
                  <div className="text-[10px] opacity-80">&lt;2ms zero-token dataset answers</div>
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="mt-2 py-3 bg-primary text-white font-mono text-xs font-bold border-3 border-black hard-shadow hover:bg-primary-container"
            >
              SAVE SETTINGS & RELOAD PIPELINE
            </button>
          </div>
        </div>

        {/* Right Column: Storage & System Diagnostics */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          
          {/* Storage Details */}
          <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col gap-4">
            <h3 className="font-headline font-black text-lg uppercase tracking-tight flex items-center gap-2 border-b-2 border-black pb-2">
              <span className="material-symbols-outlined text-primary">database</span>
              2. Vector Storage & Corpus
            </h3>

            <div className="space-y-3 font-mono text-xs">
              <div className="flex justify-between border-b border-slate-200 pb-1.5">
                <span className="text-slate-600">Storage Engine:</span>
                <span className="font-bold">Qdrant Embedded</span>
              </div>
              <div className="flex justify-between border-b border-slate-200 pb-1.5">
                <span className="text-slate-600">Embedding Model:</span>
                <span className="font-bold">MiniLM-L12-v2 Transformer (FastEmbed ONNX)</span>
              </div>
              <div className="flex justify-between border-b border-slate-200 pb-1.5">
                <span className="text-slate-600">Indexed Collections:</span>
                <span className="font-bold">4 (Sentence, Sliding, Semantic, Combined)</span>
              </div>
              <div className="flex justify-between border-b border-slate-200 pb-1.5">
                <span className="text-slate-600">BM25 Tokenizer:</span>
                <span className="font-bold">Language-Aware Indic Tokenizer</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-600">Reranker:</span>
                <span className="font-bold">FlashRank ms-marco-MiniLM-L-12-v2</span>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};
