import React from 'react';

export const DocumentationView: React.FC = () => {
  return (
    <div className="flex-1 flex flex-col p-6 md:p-10 overflow-y-auto max-w-6xl mx-auto w-full gap-8">
      
      {/* Header */}
      <div className="bg-white border-4 border-black p-6 hard-shadow">
        <div className="flex items-center gap-3 mb-2">
          <span className="material-symbols-outlined text-primary text-3xl">menu_book</span>
          <h2 className="font-headline font-black text-2xl tracking-tight uppercase">
            VAANI — Voice RAG Engine Documentation
          </h2>
        </div>
        <p className="font-mono text-xs text-slate-600">
          Complete production architecture, offline multi-strategy chunking, low-latency search pipeline, and benchmark metrics.
        </p>
      </div>

      {/* Section 1: End-to-End Pipeline Architecture */}
      <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col gap-4">
        <h3 className="font-headline font-black text-lg uppercase tracking-tight flex items-center gap-2 border-b-2 border-black pb-2">
          <span className="material-symbols-outlined text-primary">account_tree</span>
          1. System Pipeline Architecture
        </h3>
        <div className="bg-slate-900 text-white p-4 font-mono text-xs border-2 border-black overflow-x-auto leading-relaxed">
          <pre>{`Voice Input (Web Audio / MediaRecorder @ 250ms chunks)
    ↓
Sarvam AI STT (High-accuracy Speech-to-Text)
    ↓
Security & Injection Gate (Heuristic Pattern & Delimiter Sanitizer: 0.02ms)
    ↓
Query Analyzer & Adaptive Router (Intent Classification & Strategy Weighting: 0.05ms)
    ↓
Parallel Hybrid Retrieval (asyncio.gather: ~3.2ms)
    ├── Qdrant Dense Vector Search (ONNX Transformer Embeddings)
    └── BM25 Lexical Matching (Token-level BM25 Indexing)
    ↓
Reciprocal Rank Fusion (RRF k=60: 0.04ms)
    ↓
FlashRank Cross-Encoder Reranking (ms-marco-MiniLM-L-12-v2: ~28ms)
    ↓
Retrieval Confidence & Off-Topic Guardrail (Threshold Gate: 0.01ms)
    ↓
Grounded LLM Generation (Google Gemini 2.5 Flash / Strict XML Sandbox)
    ↓
Lexical Factuality & Grounding Verifier (0.34ms)
    ↓
Final Grounded Answer & Nanosecond Stage Telemetry`}</pre>
        </div>
      </div>

      {/* Section 2: 4 Offline Chunking Strategies */}
      <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col gap-4">
        <h3 className="font-headline font-black text-lg uppercase tracking-tight flex items-center gap-2 border-b-2 border-black pb-2">
          <span className="material-symbols-outlined text-primary">segment</span>
          2. Four Distinct Offline Chunking Strategies
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-surface border-2 border-black">
            <h4 className="font-headline font-bold text-sm uppercase text-primary mb-1">1. Sentence Chunking</h4>
            <p className="font-mono text-xs text-slate-700 leading-relaxed">
              Splits text by full stops (<code className="bg-white px-1 border border-black font-bold">.</code>) and question marks (<code className="bg-white px-1 border border-black font-bold">?</code>, <code className="bg-white px-1 border border-black font-bold">!</code>). Ideal for concise factual QA like dates, numbers, and definitions.
            </p>
          </div>

          <div className="p-4 bg-surface border-2 border-black">
            <h4 className="font-headline font-bold text-sm uppercase text-primary mb-1">2. Sliding Window Chunking</h4>
            <p className="font-mono text-xs text-slate-700 leading-relaxed">
              Fixed word chunk sizes (120 words) with 25% overlap (30 words) ensuring boundary facts and cross-sentence dependencies are never severed across chunks.
            </p>
          </div>

          <div className="p-4 bg-surface border-2 border-black">
            <h4 className="font-headline font-bold text-sm uppercase text-primary mb-1">3. Semantic Topic Chunking</h4>
            <p className="font-mono text-xs text-slate-700 leading-relaxed">
              Groups coherent paragraphs by topic continuity and semantic similarity. Best for complex conceptual queries like photosynthesis mechanisms and historical impact.
            </p>
          </div>

          <div className="p-4 bg-surface border-2 border-black">
            <h4 className="font-headline font-bold text-sm uppercase text-primary mb-1">4. Hierarchical Chunking</h4>
            <p className="font-mono text-xs text-slate-700 leading-relaxed">
              Preserves parent document title and child passage hierarchy, enabling multi-level context retrieval with parent document enrichment.
            </p>
          </div>
        </div>
      </div>

      {/* Section 3: REST API Specifications */}
      <div className="bg-white border-4 border-black p-6 hard-shadow flex flex-col gap-4">
        <h3 className="font-headline font-black text-lg uppercase tracking-tight flex items-center gap-2 border-b-2 border-black pb-2">
          <span className="material-symbols-outlined text-primary">api</span>
          3. REST API Endpoints
        </h3>

        <div className="space-y-4 font-mono text-xs">
          <div className="p-3 bg-surface-container border-2 border-black">
            <div className="flex items-center gap-2 mb-1">
              <span className="bg-primary text-white font-bold px-2 py-0.5 border border-black">POST</span>
              <span className="font-bold text-black">/api/voice/query</span>
            </div>
            <p className="text-slate-600 mb-2">Accepts multipart audio file, runs Sarvam STT, and executes complete RAG pipeline.</p>
            <span className="text-slate-500">Payload: <code className="bg-white px-1 border border-black">file: UploadFile (.wav, .webm, .mp3)</code></span>
          </div>

          <div className="p-3 bg-surface-container border-2 border-black">
            <div className="flex items-center gap-2 mb-1">
              <span className="bg-neon-green text-black font-bold px-2 py-0.5 border border-black">POST</span>
              <span className="font-bold text-black">/api/rag/query</span>
            </div>
            <p className="text-slate-600 mb-2">Executes hybrid RAG for text queries.</p>
            <span className="text-slate-500">Payload: <code className="bg-white px-1 border border-black">{`{"query": "string"}`}</code></span>
          </div>

          <div className="p-3 bg-surface-container border-2 border-black">
            <div className="flex items-center gap-2 mb-1">
              <span className="bg-secondary-container text-black font-bold px-2 py-0.5 border border-black">GET</span>
              <span className="font-bold text-black">/api/benchmark/results</span>
            </div>
            <p className="text-slate-600">Retrieves the latest verified benchmark evaluation metrics and latency percentiles.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
