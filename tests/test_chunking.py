import pytest
from backend.chunking.sentence_chunker import SentenceChunker
from backend.chunking.sliding_chunker import SlidingWindowChunker
from backend.chunking.hierarchical import MultiStrategyChunker


def test_hindi_marathi_sentence_chunking():
    chunker = SentenceChunker(sentences_per_chunk=1)
    text = "प्रकाश संश्लेषण एक जैव रासायनिक प्रक्रिया है। यह पौधों के विकास के लिए आवश्यक है। क्या आप इसे समझ सकते हैं?"
    sentences = chunker.split_into_sentences(text)
    assert len(sentences) == 3
    assert "प्रकाश संश्लेषण" in sentences[0]


def test_sliding_window_chunking():
    chunker = SlidingWindowChunker(window_size=20, overlap=10)
    text = " ".join([f"word_{i}" for i in range(50)])
    chunks = chunker.chunk(text, {"doc_id": "test_doc"})
    assert len(chunks) >= 3
    assert chunks[0]["chunk_strategy"] == "sliding_window"


def test_multi_strategy_chunker():
    chunker = MultiStrategyChunker()
    doc_text = "Photosynthesis is vital. It creates glucose and oxygen. Plants use chlorophyll for this."
    result = chunker.process_document("doc_01", doc_text, language="en")
    assert "sentence" in result
    assert "sliding_window" in result
    assert "semantic" in result
