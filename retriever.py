# Import necessary libraries
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load domain data
dataset = load_dataset("squad_v2", split="train[:2000]")  # same small sample
corpus = []
for example in dataset:
    if example["answers"]["text"]:
        corpus.append(example["answers"]["text"][0])


# Create embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(corpus, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "faiss_index.bin")

# Save corpus
with open("corpus.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        f.write(line.strip() + "\n")

print("✅ FAISS index and corpus saved.")
