import zipfile
import os
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# -------------------------------
# 1. Unzip if needed
# -------------------------------
zip_path = "../corpora/passage_level/clapnq.jsonl.zip"
extract_dir = "../corpora/passage_level/"
target_file = os.path.join(extract_dir, "clapnq.jsonl")

if not os.path.exists(target_file):
    print("Extracting clapnq.jsonl.zip...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print("✓ Extracted!")
else:
    print("clapnq.jsonl already exists, skipping unzip.")


# -------------------------------
# 2. Load corpus
# -------------------------------
print("Loading passages...")
passage_ids = []
passage_texts = []

with open(target_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        passage_ids.append(obj["_id"])
        passage_texts.append(obj["text"])

print(f"✓ Loaded {len(passage_ids)} passages")


# -------------------------------
# 3. Load embedding model
# -------------------------------
print("Loading MiniLM model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✓ Model loaded")


# -------------------------------
# 4. Embed and build FAISS index
# -------------------------------
DIM = 384  # MiniLM embedding dim

# Choose index
index = faiss.IndexFlatIP(DIM)  # cosine similarity (use normalized vectors)

# Metadata will be saved separately
metadata = []

BATCH = 256
print("Embedding and adding to FAISS index...")

for i in tqdm(range(0, len(passage_texts), BATCH)):
    batch_texts = passage_texts[i:i+BATCH]
    batch_ids = passage_ids[i:i+BATCH]

    embeddings = model.encode(batch_texts, convert_to_numpy=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    for pid, txt in zip(batch_ids, batch_texts):
        metadata.append({"id": pid, "text": txt})

print("✓ Finished building FAISS index!")


# -------------------------------
# 5. Save FAISS index + metadata
# -------------------------------
faiss.write_index(index, "faiss_index.bin")

with open("passages.jsonl", "w") as f:
    for item in metadata:
        f.write(json.dumps(item) + "\n")

print("✓ faiss_index.bin and passages.jsonl saved!")
print("All done!")
