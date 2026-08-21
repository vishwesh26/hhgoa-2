# Multilingual Embedding Model Selection

## Executive Summary
For the **HH Goa 2026 Polyglot Voice RAG Engine**, our system must support high-accuracy cross-lingual and monolingual semantic retrieval across **English, Hindi, and Marathi** under a strict **sub-200ms end-to-end target**.

---

## 1. Candidate Comparison Matrix

| Model | Architecture | Params / Dim | Indic Quality (Hi/Mr) | Cross-Lingual Alignment | CPU Latency (batch=1) | ONNX Support | Memory Footprint | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`paraphrase-multilingual-mpnet-base-v2`** | MPNet | 278M / 768 | **Excellent** | **High** | **~25-35 ms** | **Yes** | **~1.1 GB** | **SELECTED** |
| `paraphrase-multilingual-MiniLM-L12-v2` | MiniLM | 118M / 384 | Good | Moderate | ~15-20 ms | Yes | ~470 MB | Fallback (Low-Memory) |
| `BAAI/bge-m3` | Multi-Vector | 567M / 1024 | State-of-the-Art | Very High | ~90-140 ms | Partial | ~2.3 GB | Too slow for CPU 200ms |
| `intfloat/multilingual-e5-base` | RoBERTa | 278M / 768 | Very Good | High | ~40-55 ms | Yes | ~1.1 GB | Strong alternative |
| `ai4bharat/indic-bert` | ALBERT | 33M / 768 | Good (Indic-only) | Poor (Cross-lingual) | ~15 ms | Experimental | ~135 MB | Insufficient for cross-lingual |

---

## 2. Rationale for Selecting `paraphrase-multilingual-mpnet-base-v2`

1. **Native Devanagari & Latin Space Alignment**:
   - `paraphrase-multilingual-mpnet-base-v2` is trained across 50+ languages with teacher-student distillation from English MPNet, creating a unified vector space where Hindi (`hin_Deva`), Marathi (`mar_Deva`), and English (`eng_Latn`) concepts share close cosine proximity.
2. **Deterministic Cross-Lingual Retrieval**:
   - Allows a user asking in Marathi (*"प्रकाश संश्लेषण म्हणजे काय?"*) to retrieve both Marathi passages and aligned English ground-truth scientific passages.
3. **ONNX Runtime Acceleration**:
   - Supports ONNX quantization reducing CPU embedding inference latency from ~45ms to ~18ms, fitting within the 200ms RAG target.
4. **Dimension & Storage Efficiency**:
   - 768-dimensional float32 vectors require ~3 KB per point in Qdrant, allowing 100,000 chunks to fit in ~300 MB RAM.

---

## 3. Runtime Strategy

- **Primary**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` with ONNX / PyTorch.
- **Batched Ingestion**: Batch size = 32-64 with exponential backoff and resumable checkpointing.
- **Deterministic Lightweight Fallback**: Fast character n-gram hashing vectorizer for zero-dependency test/offline execution.
