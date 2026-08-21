import pyarrow.parquet as pq
import sys

sys.stdout.reconfigure(encoding='utf-8')

pf = pq.ParquetFile('hinval.parquet')
batch = next(pf.iter_batches(batch_size=15, columns=['query_id', 'query', 'Eng_Query', 'Answer']))
df = batch.to_pandas()

print("--- 15 QUERIES PRESENT IN CURRENTLY INDEXED HINVAL DATASET ---")
for idx, r in df.iterrows():
    qid = r['query_id']
    q_hi = r['query']
    q_en = r['Eng_Query']
    ans = str(r['Answer'])[:80]
    print(f"[{idx+1}] ID: {qid}")
    print(f"    Hindi:   {q_hi}")
    print(f"    English: {q_en}")
    print(f"    Answer:  {ans}\n")
