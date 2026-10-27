
# ==================================
# 1. IMPORT LIBRARIES
# ==================================

from sentence_transformers import SentenceTransformer
import numpy as np
import json


# ==================================
# 2. READ DOCUMENT
# ==================================

with open("document1.txt", "r", encoding="utf-8") as file:
    text = file.read()


# ==================================
# 3. SPLIT INTO CHUNKS
# ==================================

chunk_size = 100
overlap = 20

chunks = []
start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]

    chunks.append(chunk)

    start += chunk_size - overlap

print("Total chunks:", len(chunks))


# ==================================
# 4. LOAD EMBEDDING MODEL
# ==================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==================================
# 5. CREATE EMBEDDINGS
# ==================================

embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)


# ==================================
# 6. SAVE CHUNKS AND EMBEDDINGS
# ==================================

np.save("embeddings.npy", embeddings)

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Embeddings saved!")
print("Chunks saved!")
