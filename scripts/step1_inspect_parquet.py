import pyarrow.parquet as pq
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

def inspect_file(path, name):
    pf = pq.ParquetFile(path)
    print("=" * 80)
    print(f"1. INSPECTING PARQUET FILE: {name} ({path})")
    print("=" * 80)
    print(f"Dataframe Shape / Metadata: {pf.metadata.num_rows:,} rows, {pf.metadata.num_columns} columns, {pf.num_row_groups} row group(s)")
    print("\nColumns and Data Types:")
    for field in pf.schema_arrow:
        print(f"  - {field.name}: {field.type}")

    # Read first 5 records with all columns
    batch = next(pf.iter_batches(batch_size=5))
    records = batch.to_pylist()
    
    print(f"\n5 Representative Records from {name}:")
    for i, r in enumerate(records):
        print(f"\n--- [Record {i+1}] (query_id: {r.get('query_id')}) ---")
        print(f"  Query Field (Hindi)   ['query']:      {r.get('query')}")
        print(f"  Query Field (English) ['Eng_Query']:  {r.get('Eng_Query')}")
        print(f"  Query Type            ['query_type']: {r.get('query_type')}")
        print(f"  Answer Field (Hindi)  ['Answer']:     {str(r.get('Answer'))[:120]}...")
        print(f"  Answer Field (English)['Eng_Answer']: {str(r.get('Eng_Answer'))[:120]}...")
        print(f"  Meta Field            ['meta']:       {r.get('meta')}")
        print(f"  Languages             ['source_lang'] -> ['target_lang']: {r.get('source_lang')} -> {r.get('target_lang')}")
        
        passages = r.get('passages', {})
        if isinstance(passages, dict):
            hi_passages = passages.get('Translated_passages', [])
            en_passages = passages.get('English_passages', [])
            is_sel = passages.get('is_selected', [])
            print(f"  Passages Field:")
            print(f"    - Number of Hindi Translated Passages: {len(hi_passages)}")
            print(f"    - Number of English Passages:          {len(en_passages)}")
            print(f"    - is_selected labels:                 {is_sel}")
            if hi_passages:
                print(f"    - Sample Passage 1 (HI): {str(hi_passages[0])[:140]}...")
                if any(is_sel):
                    sel_idx = is_sel.index(1) if 1 in is_sel else -1
                    if sel_idx >= 0 and sel_idx < len(hi_passages):
                        print(f"    - Selected Ground Truth Passage (Index {sel_idx}): {str(hi_passages[sel_idx])[:140]}...")

inspect_file("hintrain.parquet", "HINTRAIN")
inspect_file("hinval.parquet", "HINVAL")
