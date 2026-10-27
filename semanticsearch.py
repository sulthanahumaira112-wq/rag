
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load saved chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# 2. Load saved embeddings
embeddings = np.load("embeddings.npy")

# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Ask a question
question = input("Ask a question: ")

# 5. Convert the question into an embedding
question_embedding = model.encode([question])

# 6. Compare the question with every chunk
similarities = cosine_similarity(
    question_embedding,
    embeddings
)[0]

# 7. Find the most relevant chunk
best_index = np.argmax(similarities)

print("\nMost relevant information:")
print(chunks[best_index])

print("\nSimilarity score:")
print(round(float(similarities[best_index]), 3))
