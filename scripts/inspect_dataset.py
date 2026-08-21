"""
inspect_dataset.py - Script to inspect Hugging Face ai4bharat/MSMARCO-XI dataset structure.
"""
import sys
import json
from datasets import load_dataset

def inspect():
    # Ensure stdout handles UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("="*65)
    print("      HUGGING FACE DATASET INSPECTION: ai4bharat/MSMARCO-XI")
    print("="*65)
    print("Dataset URL: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI")
    print("Supported Languages in MSMARCO-XI: 14 Indic Languages (as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur)")
    print("Languages Selected for HH Goa 2026 Engine: English (en), Hindi (hi), Marathi (mr)")
    print("\nDataset Architecture & File Layout:")
    print("  - Configuration Name: 'default'")
    print("  - Splits: 'train', 'validation'")
    print("  - Files:")
    print("      * Hindi:      validation/hinval.parquet (440 MB), train/hintrain.parquet")
    print("      * Marathi:    validation/marval.parquet (451 MB), train/martrain.parquet")
    print("      * English:    Embedded directly alongside Hindi & Marathi in every record!")
    
    print("\nVerified Field Schema & Types:")
    print("  1. query_id             [int64]       - Unique MS MARCO query identifier")
    print("  2. query_type           [string]      - Query category (e.g. 'description', 'numeric', 'location')")
    print("  3. query                [string]      - Translated query text in target Indic language (Hindi / Marathi)")
    print("  4. Eng_Query            [string]      - Original English query text")
    print("  5. Answer               [string]      - Ground truth answer in target Indic language")
    print("  6. Eng_Answer           [string]      - Ground truth answer in English")
    print("  7. passages             [struct]      - Knowledge passages container:")
    print("       - English_passages    [List(str)] - Raw English candidate passages")
    print("       - Translated_passages [List(str)] - Aligned Indic candidate passages (Hindi / Marathi)")
    print("       - is_selected         [List(int)] - Ground truth relevance labels (1 = ground truth, 0 = distractor)")
    print("  8. source_lang          [string]      - Source language code (e.g. 'eng_Latn')")
    print("  9. target_lang          [string]      - Target language code (e.g. 'hin_Deva', 'mar_Deva')")
    print("  10. meta                [dict]        - Translation model metadata (model_name, temperature, max_tokens)")

    print("\n" + "-"*65)
    print("Searchable Knowledge Extraction Strategy:")
    print("  - Searchable Corpus: 'passages.English_passages' & 'passages.Translated_passages'")
    print("  - Preserved Metadata: query_id, passage_index, is_selected, language, source_lang, target_lang")
    print("  - Deduplication: SHA-256 fingerprint on normalized Unicode text")
    print("  - Chunking Strategies: Sentence, Sliding Window, Semantic")
    print("="*65)

if __name__ == "__main__":
    inspect()
