
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ==================================
# 1. LOAD DOCUMENT
# ==================================

with open("document1.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("DOCUMENT LOADED SUCCESSFULLY!")
print("Total characters:", len(text))


# ==================================
# 2. SPLIT INTO CHUNKS
# ==================================

# Split by paragraphs for this small document
chunks = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

print("\nTotal chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nCHUNK {i + 1}")
    print("-" * 40)
    print(chunk)


# ==================================
# 3. LOAD EMBEDDING MODEL
# ==================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================
# 4. CREATE EMBEDDINGS
# ==================================

embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

print("\nEmbedding shape:", embeddings.shape)


# ==================================
# 5. SAVE CHUNKS AND EMBEDDINGS
# ==================================

with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

np.save("embeddings.npy", embeddings)

print("Chunks and embeddings saved!")


# ==================================
# 6. LOAD TEXT-GENERATION MODEL
# ==================================

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


# ==================================
# 7. ASK A QUESTION
# ==================================

question = input("\nAsk a question: ")


# ==================================
# 8. EMBED THE QUESTION
# ==================================

question_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True
)


# ==================================
# 9. RETRIEVE TOP 3 CHUNKS
# ==================================

scores = cosine_similarity(
    question_embedding,
    embeddings
)[0]

top_indices = np.argsort(scores)[::-1][:min(3, len(chunks))]

context = "\n".join(
    chunks[int(index)]
    for index in top_indices
)

print("\nRetrieved context:")
print(context)


# ==================================
# 10. BUILD THE PROMPT
# ==================================

prompt = f"""
Answer the question using only the information in the context.
If the context does not contain the answer, say:
I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:
"""


# ==================================
# 11. GENERATE THE ANSWER
# ==================================

result = generator(
    prompt,
    max_new_tokens=100,
    do_sample=False
)

print("\nFinal Answer:")
print(result[0]["generated_text"])
