from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def create_index(chunks):
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))
    return index, embeddings


def retrieve(query, chunks, index, k=2):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]
    return results
