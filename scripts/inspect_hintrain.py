import pyarrow.parquet as pq
import sys

sys.stdout.reconfigure(encoding='utf-8')

pf = pq.ParquetFile('hintrain.parquet')
print("=" * 70)
print("             HINTRAIN.PARQUET FILE SUMMARY")
print("=" * 70)
print(f"Total Rows (Records / QA Pairs): {pf.metadata.num_rows:,}")
print(f"Schema Columns: {[field.name for field in pf.schema_arrow]}")
print("=" * 70)

# Read query metadata columns (very fast)
cols = ['query_id', 'query', 'Eng_Query', 'query_type', 'Eng_Answer', 'Answer']
batch_iter = pf.iter_batches(batch_size=25, columns=cols)
first_batch = next(batch_iter)
df = first_batch.to_pandas()

print("\nLIST OF SAMPLE QUERIES IN HINTRAIN.PARQUET:\n")
for idx, row in df.iterrows():
    qid = row['query_id']
    q_hi = row['query']
    q_en = row['Eng_Query']
    q_type = row.get('query_type', 'N/A')
    ans_hi = str(row.get('Answer', ''))[:100]
    print(f"[{idx+1}] ID: {qid} | Type: {q_type}")
    print(f"    Hindi Query:   {q_hi}")
    print(f"    English Query: {q_en}")
    if ans_hi and ans_hi != 'None':
        print(f"    Ground Answer: {ans_hi}...")
    print("-" * 70)
