import pyarrow.parquet as pq
import sys

sys.stdout.reconfigure(encoding='utf-8')

def search_corporation(path, name):
    print("=" * 80)
    print(f"2. SEARCHING DATASET FOR 'कॉर्पोरेशन' in {name} ({path})")
    print("=" * 80)
    pf = pq.ParquetFile(path)
    matches = []
    
    for batch in pf.iter_batches(batch_size=100, columns=['query_id', 'query', 'Eng_Query', 'Answer', 'Eng_Answer', 'passages', 'query_type']):
        df = batch.to_pandas()
        for idx, row in df.iterrows():
            q_hi = str(row['query'])
            q_en = str(row['Eng_Query'])
            ans_hi = str(row['Answer'])
            passages = row['passages']
            
            hi_passages = passages.get('Translated_passages', []) if isinstance(passages, dict) else []
            is_sel = passages.get('is_selected', []) if isinstance(passages, dict) else []
            
            found = False
            # Check query or passages
            if "कॉर्पोरेशन" in q_hi or "corporation" in q_en.lower():
                found = True
            elif any("कॉर्पोरेशन" in str(p) for p in hi_passages):
                found = True
                
            if found:
                matches.append({
                    "query_id": row['query_id'],
                    "query_hi": q_hi,
                    "query_en": q_en,
                    "query_type": row['query_type'],
                    "ans_hi": ans_hi,
                    "hi_passages": hi_passages,
                    "is_sel": is_sel
                })
                if len(matches) >= 10:
                    break
        if len(matches) >= 10:
            break
            
    print(f"Found {len(matches)} matching records in {name}:\n")
    for i, m in enumerate(matches):
        print(f"--- MATCH {i+1} (Query ID: {m['query_id']}, Type: {m['query_type']}) ---")
        print(f"  Hindi Query:   {m['query_hi']}")
        print(f"  English Query: {m['query_en']}")
        print(f"  Ground Answer: {m['ans_hi']}")
        print(f"  Passages count: {len(m['hi_passages'])}")
        for p_idx, (p_text, sel) in enumerate(zip(m['hi_passages'], m['is_sel'])):
            if "कॉर्पोरेशन" in str(p_text) or sel == 1:
                flag = " [GROUND TRUTH SELECTED]" if sel == 1 else ""
                print(f"    Passage {p_idx+1}{flag}: {str(p_text)[:160]}...")
        print()

search_corporation("hinval.parquet", "HINVAL")
