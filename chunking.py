
with open("document1.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("DOCUMENT LOADED SUCCESSFULLY!")
print("Total characters:", len(text))


# ==================================
# 2. CHUNK SETTINGS
# ==================================

chunk_size = 100
overlap = 20


# ==================================
# 3. SPLIT INTO CHUNKS
# ==================================

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start += chunk_size - overlap


# ==================================
# 4. DISPLAY CHUNKS
# ==================================

print("\nTOTAL CHUNKS:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\nCHUNK {i + 1}")
    print("-" * 40)
    print(chunk)


# ==================================
# 5. SAVE CHUNKS TO A TEXT FILE
# ==================================

with open("chunks.txt", "w", encoding="utf-8") as file:

    for i, chunk in enumerate(chunks):

        file.write(f"CHUNK {i + 1}\n")
        file.write(chunk)
        file.write("\n\n")


print("\nChunks saved to chunks.txt!")
