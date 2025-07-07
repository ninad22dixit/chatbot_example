# Import necessary libraries
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


#Load retriever
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.bin")
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f]

# Load and fine-tune LLM
model_name_or_path = "./lora-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)


# Chat loop
def retrieve_context(query, k=1):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [corpus[i] for i in I[0]]

def generate_answer(user_query):
    context = retrieve_context(user_query)[0]
    prompt = f"Answer the question using this context:\n{context}\nQuestion: {user_query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Output
print("✅ Chatbot ready (RAG with LoRA fine-tuned LLM)")
while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = generate_answer(query)
    print(f"Bot: {answer}")

